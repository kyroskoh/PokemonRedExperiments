from pathlib import Path
import uuid
import re

from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback


def make_env(rank, env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


if __name__ == '__main__':
    use_wandb_logging = False
    ep_length = 2048 * 10
    sess_id = str(uuid.uuid4())[:8]

    REPO_ROOT = Path(__file__).resolve().parents[1]
    sess_path = REPO_ROOT / f"session_{sess_id}"
    sess_path.mkdir(parents=True, exist_ok=True)

    gb_path = REPO_ROOT / "PokemonRed.gb"
    init_state = REPO_ROOT / "has_pokedex_nballs.state"

    if not gb_path.exists():
        raise FileNotFoundError(f"ROM not found at: {gb_path}")
    if not init_state.exists():
        raise FileNotFoundError(f"Init state not found at: {init_state}")

    env_config = {
        "headless": True,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": str(init_state.resolve()),
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": str(gb_path.resolve()),
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "use_screen_explore": True,
        "reward_scale": 4,
        "extra_buttons": False,
        "explore_weight": 3,
    }

    print(env_config)

    num_cpu = 16
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length,
        save_path=sess_path,
        name_prefix="poke"
    )

    callbacks = [checkpoint_callback, TensorboardCallback(log_dir=str(sess_path))]

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

    # 🔎 Auto-pick newest checkpoint if exists
    candidates = sorted(
        REPO_ROOT.rglob("poke_*_steps.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if candidates:
        ckpt = candidates[0]
        print(f"[INFO] Resuming from checkpoint: {ckpt}")
        model = PPO.load(str(ckpt), env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        print("[WARN] No checkpoints found — starting new model")
        model = PPO(
            "CnnPolicy",
            env,
            verbose=1,
            n_steps=ep_length // 8,
            batch_size=128,
            n_epochs=3,
            gamma=0.998,
            tensorboard_log=str(sess_path),
        )

    model.learn(
        total_timesteps=(ep_length) * num_cpu * 5000,
        callback=CallbackList(callbacks)
    )

    if use_wandb_logging:
        run.finish()
