from pathlib import Path
import uuid
import re
import time

from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback


# ----------------------------- helpers -----------------------------

def make_env(rank, env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


def build_env_config(repo_root: Path, sess_path: Path, extra_buttons: bool = False):
    gb_path = repo_root / "PokemonRed.gb"
    init_state = repo_root / "has_pokedex_nballs.state"

    if not gb_path.exists():
        raise FileNotFoundError(f"ROM not found at: {gb_path}")
    if not init_state.exists():
        raise FileNotFoundError(f"Init state not found at: {init_state}")

    return {
        "headless": True,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": str(init_state.resolve()),
        "max_steps": 2048 * 10,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": str(gb_path.resolve()),
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "use_screen_explore": True,
        "reward_scale": 4,
        "extra_buttons": extra_buttons,  # 6 actions (False) vs 8 actions (True)
        "explore_weight": 3,
    }


def create_vec_env(env_config: dict, num_cpu: int):
    return SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])


def newest_checkpoint(repo_root: Path):
    cands = sorted(
        repo_root.rglob("poke_*_steps.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def close_vec_env(env):
    # Best-effort SubprocVecEnv cleanup
    try:
        env.close()
    except Exception:
        pass
    time.sleep(0.1)


# ------------------------------ main ------------------------------

if __name__ == "__main__":
    # --- paths / session setup ---
    REPO_ROOT = Path(__file__).resolve().parents[1]
    sess_id = str(uuid.uuid4())[:8]
    sess_path = REPO_ROOT / f"session_{sess_id}"
    sess_path.mkdir(parents=True, exist_ok=True)

    # --- config & env (start with 6-action env; will flip to 8 if needed) ---
    num_cpu = 16
    env_config = build_env_config(REPO_ROOT, sess_path, extra_buttons=False)
    print(env_config)

    env = create_vec_env(env_config, num_cpu)

    # --- callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=env_config["max_steps"],
        save_path=sess_path,
        name_prefix="poke",
    )
    callbacks = [checkpoint_callback, TensorboardCallback(log_dir=str(sess_path))]

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

    # --- model: resume from newest checkpoint if present; else fresh ---
    ckpt = newest_checkpoint(REPO_ROOT)
    model = None

    if ckpt:
        print(f"[INFO] Resuming from checkpoint: {ckpt}")
        try:
            model = PPO.load(str(ckpt), env=env)
        except ValueError as e:
            msg = str(e)
            # Typical mismatch: "Action spaces do not match: Discrete(8) != Discrete(6)"
            m = re.search(r"Discrete\((\d+)\)\s*!=\s*Discrete\((\d+)\)", msg)
            if m:
                ckpt_actions = int(m.group(1))
                env_actions = int(m.group(2))
                print(f"[WARN] Action space mismatch: checkpoint={ckpt_actions} vs env={env_actions}")

                # Case 1: ckpt=8, env=6 → rebuild env with extra_buttons=True
                if ckpt_actions == 8 and env_actions == 6:
                    print("[INFO] Rebuilding env with extra_buttons=True …")
                    close_vec_env(env)
                    env_config = build_env_config(REPO_ROOT, sess_path, extra_buttons=True)
                    env = create_vec_env(env_config, num_cpu)
                    model = PPO.load(str(ckpt), env=env)

                # Case 2: ckpt=6, env=8 → rebuild env with extra_buttons=False
                elif ckpt_actions == 6 and env_actions == 8:
                    print("[INFO] Rebuilding env with extra_buttons=False …")
                    close_vec_env(env)
                    env_config = build_env_config(REPO_ROOT, sess_path, extra_buttons=False)
                    env = create_vec_env(env_config, num_cpu)
                    model = PPO.load(str(ckpt), env=env)

                else:
                    raise
            else:
                raise

        # align buffer sizes with our settings (optional but keeps rollout shape consistent)
        model.n_steps = env_config["max_steps"]
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = env_config["max_steps"]
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()

    if model is None:
        print("[WARN] No checkpoints found — starting new model")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=env_config["max_steps"] // 8,
            batch_size=128,
            n_epochs=3,
            gamma=0.998,
            tensorboard_log=str(sess_path),
        )

    # --- train ---
    total_timesteps = env_config["max_steps"] * num_cpu * 5000
    model.learn(
        total_timesteps=total_timesteps,
        callback=CallbackList(callbacks),
    )

    if use_wandb_logging:
        run.finish()
