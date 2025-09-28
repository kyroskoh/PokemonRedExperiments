from os.path import exists
from pathlib import Path
import uuid

from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common import env_checker
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback


def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init


if __name__ == "__main__":

    use_wandb_logging = False
    ep_length = 2048 * 10
    sess_id = str(uuid.uuid4())[:8]
    sess_path = Path(f"session_{sess_id}")
    sess_path.mkdir(parents=True, exist_ok=True)

    env_config = {
        "headless": True,
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": str(Path("../has_pokedex_nballs.state").resolve()),
        "max_steps": ep_length,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,
        "gb_path": str(Path("../PokemonRed.gb").resolve()),
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "use_screen_explore": True,
        "reward_scale": 4,
        "extra_buttons": False,
        "explore_weight": 3,  # 2.5
    }

    print(env_config)

    num_cpu = 16  # number of environments
    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length, save_path=sess_path, name_prefix="poke"
    )

    callbacks = [
        checkpoint_callback,
        TensorboardCallback(log_dir=str(sess_path)),
    ]

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

    # If checkpoint exists, resume
    file_name = "session_e41c9eff/poke_38207488_steps"

    if exists(file_name + ".zip"):
        print("\n[INFO] Loading checkpoint…")
        model = PPO.load(file_name, env=env)
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
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

    # Train up to 5k episodes
    model.learn(
        total_timesteps=(ep_length) * num_cpu * 5000,
        callback=CallbackList(callbacks),
    )

    if use_wandb_logging:
        run.finish()
