from os.path import exists
from pathlib import Path
import os
import uuid

from red_gym_env import RedGymEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList
from tensorboard_callback import TensorboardCallback

def make_env(rank, env_conf, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    set_random_seed(seed)
    return _init

if __name__ == '__main__':
    # --- Resolve paths relative to this file (robust for SubprocVecEnv on Windows) ---
    THIS_FILE = Path(__file__).resolve()
    REPO_ROOT = THIS_FILE.parents[1]  # .../PokemonRedExperiments
    DEFAULT_ROM = REPO_ROOT / "PokemonRed.gb"
    DEFAULT_STATE = REPO_ROOT / "has_pokedex_nballs.state"

    # Allow environment overrides if you ever set GB_PATH/STATE_PATH in PowerShell
    gb_path = Path(os.getenv("GB_PATH", str(DEFAULT_ROM))).expanduser().resolve()
    init_state = Path(os.getenv("STATE_PATH", str(DEFAULT_STATE))).expanduser().resolve()

    if not gb_path.exists():
        raise FileNotFoundError(f"ROM not found at: {gb_path}")
    if not init_state.exists():
        raise FileNotFoundError(f"Init state not found at: {init_state}")

    use_wandb_logging = False
    ep_length = 2048 * 10
    sess_id = str(uuid.uuid4())[:8]
    sess_path = REPO_ROOT / f"session_{sess_id}"
    sess_path.mkdir(parents=True, exist_ok=True)

    env_config = {
        'headless': True,
        'save_final_state': True,
        'early_stop': False,
        'action_freq': 24,
        'init_state': str(init_state),          # absolute path
        'max_steps': ep_length,
        'print_rewards': True,
        'save_video': False,
        'fast_video': True,
        'session_path': sess_path,              # Path is fine; RedGymEnv should accept it. If not, str(sess_path)
        'gb_path': str(gb_path),                # absolute path
        'debug': False,
        'sim_frame_dist': 2_000_000.0,
        'use_screen_explore': True,
        'reward_scale': 4,
        'extra_buttons': False,
        'explore_weight': 3
    }

    print(env_config)

    # You can tune this; on Windows with spawn, avoid exceeding physical cores
    num_cpu = 16

    env = SubprocVecEnv([make_env(i, env_config) for i in range(num_cpu)])

    checkpoint_callback = CheckpointCallback(
        save_freq=ep_length,
        save_path=str(sess_path),   # ensure str for SB3 callback on Windows
        name_prefix='poke'
    )

    callbacks = [checkpoint_callback, TensorboardCallback()]

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

    # Put a checkpoint here you want to start from (relative to repo root)
    file_name = REPO_ROOT / 'session_e41c9eff' / 'poke_38207488_steps'

    if exists(str(file_name) + '.zip'):
        print('\nloading checkpoint')
        model = PPO.load(str(file_name), env=env)
        # Align buffer sizes with our settings
        model.n_steps = ep_length
        model.n_envs = num_cpu
        model.rollout_buffer.buffer_size = ep_length
        model.rollout_buffer.n_envs = num_cpu
        model.rollout_buffer.reset()
    else:
        model = PPO(
            'CnnPolicy',
            env,
            verbose=1,
            n_steps=ep_length // 8,
            batch_size=128,
            n_epochs=3,
            gamma=0.998,
            tensorboard_log=str(sess_path)
        )

    # Run up to 5k episodes
    model.learn(total_timesteps=(ep_length) * num_cpu * 5000, callback=CallbackList(callbacks))

    if use_wandb_logging:
        run.finish()
