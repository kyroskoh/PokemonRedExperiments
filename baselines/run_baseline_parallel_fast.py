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


# --- Utils -------------------------------------------------------------------

def make_env(rank, env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


def create_vec_env(env_conf, num_cpu):
    return SubprocVecEnv([make_env(i, env_conf) for i in range(num_cpu)])


def close_vec_env(env):
    """Ensure SubprocVecEnv is properly terminated"""
    try:
        env.close()
    except Exception:
        pass


def build_env_config(repo_root, sess_path, extra_buttons=False):
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
        "max_steps": 2048 * 10,  # training horizon
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": str(gb_path),
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "use_screen_explore": True,
        "reward_scale": 4,
        "extra_buttons": extra_buttons,
        "explore_weight": 3,
    }


def safe_rebuild_env(env, env_config, num_cpu, extra_buttons):
    """Kill old env, sleep briefly, rebuild with new extra_buttons setting"""
    close_vec_env(env)
    time.sleep(1)
    env_config["extra_buttons"] = extra_buttons
    return create_vec_env(env_config, num_cpu)


# --- Main --------------------------------------------------------------------

if __name__ == "__main__":
    THIS_FILE = Path(__file__).resolve()
    REPO_ROOT = THIS_FILE.parents[1]

    # Session folder
    sess_id = str(uuid.uuid4())[:8]
    sess_path = REPO_ROOT / f"session_{sess_id}"
    sess_path.mkdir(parents=True, exist_ok=True)

    num_cpu = 16
    env_config = build_env_config(REPO_ROOT, sess_path, extra_buttons=False)
    env = create_vec_env(env_config, num_cpu)

    checkpoint_callback = CheckpointCallback(
        save_freq=env_config["max_steps"],
        save_path=sess_path,
        name_prefix="poke"
    )
    callbacks = [checkpoint_callback, TensorboardCallback(log_dir=str(sess_path))]

    # --- Auto-pick newest checkpoint
    candidates = sorted(
        REPO_ROOT.rglob("poke_*_steps.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    model = None
    if candidates:
        ckpt = candidates[0]
        print(f"[INFO] Resuming from checkpoint: {ckpt}")
        try:
            model = PPO.load(str(ckpt), env=env)
        except ValueError as e:
            msg = str(e)
            m = re.search(r"Discrete\((\d+)\)\s*!=\s*Discrete\((\d+)\)", msg)
            if m:
                ckpt_actions, env_actions = int(m.group(1)), int(m.group(2))
                print(f"[WARN] Action space mismatch: checkpoint={ckpt_actions} vs env={env_actions}")

                if ckpt_actions == 8 and env_actions == 6:
                    print("[INFO] Rebuilding env with extra_buttons=True …")
                    env = safe_rebuild_env(env, env_config, num_cpu, True)
                    model = PPO.load(str(ckpt), env=env)

                elif ckpt_actions == 6 and env_actions == 8:
                    print("[INFO] Rebuilding env with extra_buttons=False …")
                    env = safe_rebuild_env(env, env_config, num_cpu, False)
                    model = PPO.load(str(ckpt), env=env)
                else:
                    raise
            else:
                raise
    else:
        print("[WARN] No checkpoints found — starting from scratch.")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=1024,        # reduce memory
            batch_size=256,      # reduce memory
            n_epochs=3,
            gamma=0.998,
            tensorboard_log=str(sess_path),
        )

    # --- Training loop
    model.learn(
        total_timesteps=env_config["max_steps"] * num_cpu * 5000,
        callback=CallbackList(callbacks)
    )
