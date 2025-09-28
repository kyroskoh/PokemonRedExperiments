from pathlib import Path
import os
import uuid

from red_gym_env import RedGymEnv
from stable_baselines3 import PPO


def make_env(rank, env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=(seed + rank))
        return env
    return _init


if __name__ == "__main__":
    # Resolve paths relative to this file (repo root = .../PokemonRedExperiments)
    THIS_FILE = Path(__file__).resolve()
    REPO_ROOT = THIS_FILE.parents[1]
    DEFAULT_ROM = REPO_ROOT / "PokemonRed.gb"
    DEFAULT_STATE = REPO_ROOT / "has_pokedex_nballs.state"

    gb_path = Path(os.getenv("GB_PATH", str(DEFAULT_ROM))).expanduser().resolve()
    init_state = Path(os.getenv("STATE_PATH", str(DEFAULT_STATE))).expanduser().resolve()

    if not gb_path.exists():
        raise FileNotFoundError(f"ROM not found at: {gb_path}")
    if not init_state.exists():
        raise FileNotFoundError(f"Init state not found at: {init_state}")

    # Session folder for logs/videos (optional)
    sess_id = str(uuid.uuid4())[:8]
    sess_path = REPO_ROOT / f"session_{sess_id}"
    sess_path.mkdir(parents=True, exist_ok=True)

    # Interactive usually wants a window; set headless=False if you want SDL2 window
    env_config = {
        "headless": False,                # set True if you really want headless
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": str(init_state),    # absolute
        "max_steps": 2048 * 4,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,        # Path is fine; RedGymEnv accepts Path
        "gb_path": str(gb_path),          # absolute
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "use_screen_explore": True,
        "reward_scale": 4,
        "extra_buttons": False,
        "explore_weight": 3,
    }

    # Build a single env (interactive)
    env = make_env(0, env_config)()

    # --- Auto-detect newest checkpoint; fall back to fresh model if none ---
    candidates = sorted(
        REPO_ROOT.rglob("poke_*_steps.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if candidates:
        ckpt = candidates[0]
        print(f"[INFO] Loading checkpoint: {ckpt}")
        model = PPO.load(str(ckpt), env=env)
    else:
        print("[WARN] No checkpoints found — starting with a fresh PPO policy.")
        # Minimal defaults for interactive run; tune as you like
        model = PPO("CnnPolicy", env, verbose=1)

    # Simple interactive loop: close window / Ctrl+C to stop
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)

        step_out = env.step(action)
        # Your env returns 5 items: (obs, reward, terminated, truncated, info)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated) or bool(truncated)
        else:
            # Fallback if the env returns classic 4-tuple
            obs, reward, done, info = step_out

        if done:
            obs, _ = env.reset()
