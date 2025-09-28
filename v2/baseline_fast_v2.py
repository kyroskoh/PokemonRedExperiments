# baseline_fast_v2.py
import sys
import os
import re
import time
import uuid
from os.path import exists
from pathlib import Path

from red_gym_env_v2 import RedGymEnv
from stream_agent_wrapper import StreamWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback

# ==== Tunables ====
NUM_CPU = 64
EP_LENGTH = 2048 * 80
ROLLOUT_STEPS = 512
BATCH_SIZE = 256


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


def safe_load_model(ckpt, env, env_config: dict, num_cpu: int):
    """Load PPO with small rollout/batch. Rebuild env on action mismatch."""
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

        if ckpt_actions == 8 and env_actions == 6:
            print("[INFO] Rebuilding env with extra_buttons=True …")
            env = safe_rebuild_env(env, env_config, num_cpu, extra_buttons=True)
        elif ckpt_actions == 6 and env_actions == 8:
            print("[INFO] Rebuilding env with extra_buttons=False …")
            env = safe_rebuild_env(env, env_config, num_cpu, extra_buttons=False)
        else:
            raise

        model = PPO.load(str(ckpt), env=env, custom_objects=dict(
            n_steps=ROLLOUT_STEPS,
            batch_size=BATCH_SIZE,
        ))
        return model, env


if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parents[0]
    sess_id = str(uuid.uuid4())[:8]
    sess_path = REPO_ROOT / f"runs_{sess_id}"
    sess_path.mkdir(parents=True, exist_ok=True)

    env_config = {
        "headless": True,
        "save_final_state": False,
        "early_stop": False,
        "action_freq": 24,
        "init_state": str(REPO_ROOT / "../init.state"),
        "max_steps": EP_LENGTH,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": str(REPO_ROOT / "../PokemonRed.gb"),
        "debug": False,
        "reward_scale": 0.5,
        "explore_weight": 0.25,
        "extra_buttons": False,   # default 6-action env
    }

    print(env_config)

    env = create_vec_env(env_config, NUM_CPU)

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
        model, env = safe_load_model(ckpt, env, env_config, NUM_CPU)
    else:
        print("[WARN] No checkpoints found — starting from scratch.")
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=ROLLOUT_STEPS,
            batch_size=BATCH_SIZE,
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
