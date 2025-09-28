# baselines/run_baseline_parallel_fast.py
from pathlib import Path
import os
import uuid
import re
import time

from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback

# ==== Tunables (kept modest to avoid huge RAM) ====
NUM_CPU = 16
EP_LENGTH = 2048 * 10       # env_config["max_steps"] (per-env horizon)
ROLLOUT_STEPS = 512         # << reduced rollout size to avoid multi-GB buffers
BATCH_SIZE = 256            # pair well with ROLLOUT_STEPS


# ----------------------------- helpers -----------------------------

def make_env(rank, env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


def create_vec_env(env_conf, num_cpu: int):
    return SubprocVecEnv([make_env(i, env_conf) for i in range(num_cpu)])


def close_vec_env(env):
    """Best-effort SubprocVecEnv cleanup."""
    try:
        env.close()
    except Exception:
        pass


def build_env_config(repo_root: Path, sess_path: Path, extra_buttons: bool = False):
    DEFAULT_ROM = repo_root / "PokemonRed.gb"
    DEFAULT_STATE = repo_root / "has_pokedex_nballs.state"

    gb_path = Path(os.getenv("GB_PATH", str(DEFAULT_ROM))).expanduser().resolve()
    init_state = Path(os.getenv("STATE_PATH", str(DEFAULT_STATE))).expanduser().resolve()

    if not gb_path.exists():
        raise FileNotFoundError(f"ROM not found at: {gb_path}")
    if not init_state.exists():
        raise FileNotFoundError(f"Init state not found at: {init_state}")

    return {
        "headless": True,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": str(init_state),
        "max_steps": EP_LENGTH,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": str(gb_path),
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "use_screen_explore": True,
        "reward_scale": 4,
        "extra_buttons": extra_buttons,   # 6 actions (False) vs 8 actions (True)
        "explore_weight": 3,
    }


def newest_checkpoint(repo_root: Path):
    cands = sorted(
        repo_root.rglob("poke_*_steps.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def safe_rebuild_env(env, env_config: dict, num_cpu: int, extra_buttons: bool):
    """
    Kill old env, pause briefly, rebuild with desired action-space.
    This prevents mismatch loops and stale subprocess issues.
    """
    close_vec_env(env)
    time.sleep(1.0)
    env_config["extra_buttons"] = extra_buttons
    return create_vec_env(env_config, num_cpu)


def safe_load_model(ckpt, env, env_config: dict, num_cpu: int):
    """
    Load PPO with small rollout/batch via custom_objects.
    If action-space mismatch (8!=6 or 6!=8), rebuild env accordingly and retry once.
    Returns (model, env) because env might be rebuilt.
    """
    try:
        model = PPO.load(str(ckpt), env=env, custom_objects=dict(
            n_steps=ROLLOUT_STEPS,
            batch_size=BATCH_SIZE,
        ))
        return model, env
    except ValueError as e:
        msg = str(e)
        m = re.search(r"Discrete\((\d+)\)\s*!=\s*Discrete\((\d+)\)", msg)
        if not m:
            raise
        ckpt_actions, env_actions = int(m.group(1)), int(m.group(2))
        print(f"[WARN] Action space mismatch: checkpoint={ckpt_actions} vs env={env_actions}")

        # Case 1: ckpt=8, env=6 → use 8-action env
        if ckpt_actions == 8 and env_actions == 6:
            print("[INFO] Rebuilding env with extra_buttons=True …")
            env = safe_rebuild_env(env, env_config, num_cpu, extra_buttons=True)

        # Case 2: ckpt=6, env=8 → use 6-action env
        elif ckpt_actions == 6 and env_actions == 8:
            print("[INFO] Rebuilding env with extra_buttons=False …")
            env = safe_rebuild_env(env, env_config, num_cpu, extra_buttons=False)
        else:
            # Unknown combo; re-raise
            raise

        # Retry once after rebuilding
        model = PPO.load(str(ckpt), env=env, custom_objects=dict(
            n_steps=ROLLOUT_STEPS,
            batch_size=BATCH_SIZE,
        ))
        return model, env


# ------------------------------ main ------------------------------

if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[1]
    sess_id = str(uuid.uuid4())[:8]
    sess_path = REPO_ROOT / f"session_{sess_id}"
    sess_path.mkdir(parents=True, exist_ok=True)

    # Start with 6-action env; safe_load_model will rebuild if checkpoint expects 8
    env_config = build_env_config(REPO_ROOT, sess_path, extra_buttons=False)
    print(env_config)

    env = create_vec_env(env_config, NUM_CPU)

    checkpoint_cb = CheckpointCallback(
        save_freq=EP_LENGTH,
        save_path=sess_path,
        name_prefix="poke",
    )
    callbacks = [checkpoint_cb, TensorboardCallback(log_dir=str(sess_path))]

    # Optional Weights & Biases (off by default)
    use_wandb_logging = False
    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(WandbCallback())

    # Auto-pick newest checkpoint
    ckpt = newest_checkpoint(REPO_ROOT)
    if ckpt:
        print(f"[INFO] Resuming from checkpoint: {ckpt}")
        model, env = safe_load_model(ckpt, env, env_config, NUM_CPU)
    else:
        print("[WARN] No checkpoints found — starting from scratch.")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=ROLLOUT_STEPS,      # reduced rollout size
            batch_size=BATCH_SIZE,      # reduced batch size
            n_epochs=3,
            gamma=0.998,
            tensorboard_log=str(sess_path),
        )

    # Train
    total_timesteps = EP_LENGTH * NUM_CPU * 5000
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
    )

    if use_wandb_logging:
        run.finish()
