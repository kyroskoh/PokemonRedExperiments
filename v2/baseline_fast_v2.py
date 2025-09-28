# v2/baseline_fast_v2.py
import sys
import os
import re
import time
import uuid
from pathlib import Path
from typing import Union

from red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback
import gymnasium as gym
import numpy as np

# ===================== Tunables (with safe defaults) =====================

# You can override these at runtime via environment variables:
#   set NUM_CPU=8
#   set ROLLOUT_STEPS=256
#   set BATCH_SIZE=256
#   set MAX_BUFFER_GB=3.0
# ===================== Tunables (with safe defaults) =====================

# Override with env vars if you want more
NUM_CPU = int(os.getenv("NUM_CPU", 8))          # default = 8 for lower specs
EP_LENGTH = 2048 * 80
ROLLOUT_STEPS = int(os.getenv("ROLLOUT_STEPS", 256))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 256))
MAX_BUFFER_GB = float(os.getenv("MAX_BUFFER_GB", 2.0))  # tighter cap

# Clamp to available logical cores
try:
    NUM_CPU = max(1, min(NUM_CPU, os.cpu_count() or NUM_CPU))
except Exception:
    NUM_CPU = max(1, NUM_CPU)

# ======================== VecEnv + helpers ================================

def make_env(rank, env_conf, seed=0):
    """Multiprocess env with StreamWrapper metadata."""
    def _init():
        env = StreamWrapper(
            RedGymEnv(env_conf),
            stream_metadata={
                "user": "v2-default",
                "env_id": rank,
                "color": "#447799",
                "extra": "",
            }
        )
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


def create_vec_env(env_conf, num_cpu: int):
    return SubprocVecEnv([make_env(i, env_conf) for i in range(num_cpu)])


def close_vec_env(env):
    try:
        env.close()
    except Exception:
        pass


def safe_rebuild_env(env, env_config: dict, num_cpu: int, extra_buttons: bool):
    """Tear down old env, rebuild with desired action space."""
    close_vec_env(env)
    time.sleep(1.0)
    env_config["extra_buttons"] = extra_buttons
    return create_vec_env(env_config, num_cpu)


def newest_checkpoint(repo_root: Path):
    cands = sorted(
        repo_root.rglob("poke_*_steps.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None

# ========================= Memory estimators ==============================

def _space_bytes(space: gym.Space) -> int:
    """Rough bytes per observation (float32) for SB3 on-policy buffer."""
    if isinstance(space, gym.spaces.Box):
        # SB3 stores float32 tensors (even if uint8 obs)
        return int(np.prod(space.shape)) * 4
    if isinstance(space, gym.spaces.Discrete):
        # Stored as int64 typically, but negligible; use 8 bytes
        return 8
    if isinstance(space, gym.spaces.MultiBinary) or isinstance(space, gym.spaces.MultiDiscrete):
        return int(np.prod(space.shape)) * 8
    if isinstance(space, gym.spaces.Dict):
        return sum(_space_bytes(s) for s in space.spaces.values())
    if isinstance(space, gym.spaces.Tuple):
        return sum(_space_bytes(s) for s in space.spaces)
    # Fallback conservative
    try:
        return int(np.prod(space.shape)) * 4
    except Exception:
        return 1024  # arbitrary small fallback


def estimate_buffer_gb(obs_space: gym.Space, n_steps: int, n_envs: int) -> float:
    """
    Very rough upper-bound for PPO on-policy buffer usage:
    observations dominate; we multiply by ~2.5 for obs+values+advantages etc.
    """
    obs_bytes = _space_bytes(obs_space)
    per_step = obs_bytes * 2.5  # obs + extras (advantages, returns, etc.)
    total_bytes = per_step * n_steps * n_envs
    return total_bytes / (1024 ** 3)


def adapt_rollout_steps(obs_space: gym.Space, n_steps: int, n_envs: int, cap_gb: float) -> int:
    est = estimate_buffer_gb(obs_space, n_steps, n_envs)
    if est <= cap_gb:
        return n_steps
    # Scale down proportionally
    scale = cap_gb / max(est, 1e-6)
    new_steps = max(64, int(n_steps * scale))
    print(f"[MEM-GUARD] Estimated buffer {est:.2f} GB > cap {cap_gb:.2f} GB; "
          f"scaling ROLLOUT_STEPS {n_steps} -> {new_steps}")
    return new_steps

# ====================== Checkpoint-safe loader ============================

def safe_load_model(ckpt: Union[str, Path], env, env_config: dict, num_cpu: int,
                    n_steps: int, batch_size: int):
    """
    Load PPO with custom_objects to clamp rollout sizes.
    Rebuild env on action-space mismatch (8 vs 6).
    Returns (model, env, n_steps) since we might further adapt n_steps after env rebuild.
    """
    try:
        model = PPO.load(str(ckpt), env=env, custom_objects=dict(
            n_steps=n_steps,
            batch_size=batch_size,
        ))
        return model, env, n_steps
    except ValueError as e:
        msg = str(e)
        m = re.search(r"Discrete\((\d+)\)\s*!=\s*Discrete\((\d+)\)", msg)
        if not m:
            raise
        ckpt_actions, env_actions = int(m.group(1)), int(m.group(2))
        print(f"[WARN] Action space mismatch: checkpoint={ckpt_actions} vs env={env_actions}")

        if ckpt_actions == 8 and env_actions == 6:
            print("[INFO] Rebuilding env with extra_buttons=True …")
            env = safe_rebuild_env(env, env_config, num_cpu, extra_buttons=True)
        elif ckpt_actions == 6 and env_actions == 8:
            print("[INFO] Rebuilding env with extra_buttons=False …")
            env = safe_rebuild_env(env, env_config, num_cpu, extra_buttons=False)
        else:
            raise

        # After rebuild, recompute adaptive steps against the (same) obs space
        # (Vec env observation space remains the same shape interface)
        new_n_steps = adapt_rollout_steps(env.observation_space, n_steps, num_cpu, MAX_BUFFER_GB)

        model = PPO.load(str(ckpt), env=env, custom_objects=dict(
            n_steps=new_n_steps,
            batch_size=min(batch_size, new_n_steps * num_cpu),
        ))
        return model, env, new_n_steps

# ============================= Main =======================================

if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[0]

    # Cap NUM_CPU to logical cores (nice-to-have)
    try:
        NUM_CPU = min(NUM_CPU, os.cpu_count() or NUM_CPU)
    except Exception:
        pass

    sess_id = str(uuid.uuid4())[:8]
    sess_path = REPO_ROOT / f"runs_{sess_id}"
    sess_path.mkdir(parents=True, exist_ok=True)

    env_config = {
        "headless": True,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": str((REPO_ROOT / "../init.state").resolve()),
        "max_steps": EP_LENGTH,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": str((REPO_ROOT / "../PokemonRed.gb").resolve()),
        "debug": False,
        "reward_scale": 0.5,
        "explore_weight": 0.25,
        "extra_buttons": False,   # default 6-action env
    }

    print(env_config)

    # Build a small temp env to get observation space? Not needed:
    # SubprocVecEnv exposes obs space of first env.
    env = create_vec_env(env_config, NUM_CPU)

    # ---- Memory guard: adapt rollout steps BEFORE allocating PPO buffers
    adaptive_steps = adapt_rollout_steps(env.observation_space, ROLLOUT_STEPS, NUM_CPU, MAX_BUFFER_GB)
    effective_batch = min(BATCH_SIZE, adaptive_steps * NUM_CPU)

    checkpoint_cb = CheckpointCallback(
        save_freq=EP_LENGTH // 2,
        save_path=sess_path,
        name_prefix="poke",
    )
    callbacks = [checkpoint_cb, TensorboardCallback(sess_path)]

    use_wandb_logging = False
    if use_wandb_logging:
        import wandb
        from wandb.integration.sb3 import WandbCallback
        wandb.tensorboard.patch(root_logdir=str(sess_path))
        run = wandb.init(
            project="pokemon-train",
            id=sess_id,
            name="v2-a",
            config=env_config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        callbacks.append(WandbCallback())

    # Auto-detect newest checkpoint
    ckpt = newest_checkpoint(REPO_ROOT)
    if ckpt:
        print(f"[INFO] Resuming from checkpoint: {ckpt}")
        model, env, adaptive_steps = safe_load_model(
            ckpt, env, env_config, NUM_CPU, adaptive_steps, effective_batch
        )
    else:
        print("[WARN] No checkpoints found — starting from scratch.")
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=adaptive_steps,
            batch_size=effective_batch,
            n_epochs=1,
            gamma=0.997,
            ent_coef=0.01,
            tensorboard_log=str(sess_path),
        )

    print(model.policy)

    model.learn(
        total_timesteps=EP_LENGTH * NUM_CPU * 10000,
        callback=CallbackList(callbacks),
        tb_log_name="poke_ppo",
    )

    if use_wandb_logging:
        run.finish()
