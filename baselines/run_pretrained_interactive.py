from pathlib import Path
import os
import uuid
import re

from red_gym_env import RedGymEnv
from stable_baselines3 import PPO


def make_env(env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=seed)
        return env
    return _init


def build_env_config(repo_root, headless=False):
    DEFAULT_ROM = repo_root / "PokemonRed.gb"
    DEFAULT_STATE = repo_root / "has_pokedex_nballs.state"

    gb_path = Path(os.getenv("GB_PATH", str(DEFAULT_ROM))).expanduser().resolve()
    init_state = Path(os.getenv("STATE_PATH", str(DEFAULT_STATE))).expanduser().resolve()

    if not gb_path.exists():
        raise FileNotFoundError(f"ROM not found at: {gb_path}")
    if not init_state.exists():
        raise FileNotFoundError(f"Init state not found at: {init_state}")

    sess_id = str(uuid.uuid4())[:8]
    sess_path = repo_root / f"session_{sess_id}"
    sess_path.mkdir(parents=True, exist_ok=True)

    return {
        "headless": headless,                # SDL2 window if False
        "save_final_state": True,
        "early_stop": False,
        "action_freq": 24,
        "init_state": str(init_state),       # absolute
        "max_steps": 2048 * 4,
        "print_rewards": True,
        "save_video": False,
        "fast_video": True,
        "session_path": sess_path,           # Path is ok in your env
        "gb_path": str(gb_path),             # absolute
        "debug": False,
        "sim_frame_dist": 2_000_000.0,
        "use_screen_explore": True,
        "reward_scale": 4,
        "extra_buttons": False,              # will flip to True automatically if needed
        "explore_weight": 3,
    }


if __name__ == "__main__":
    THIS_FILE = Path(__file__).resolve()
    REPO_ROOT = THIS_FILE.parents[1]

    # Base config (interactive window)
    env_config = build_env_config(REPO_ROOT, headless=False)

    # Build env
    env = make_env(env_config)()

    # Find newest checkpoint
    candidates = sorted(
        REPO_ROOT.rglob("poke_*_steps.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    model = None
    ckpt = None

    if candidates:
        ckpt = candidates[0]
        print(f"[INFO] Loading checkpoint: {ckpt}")
        try:
            model = PPO.load(str(ckpt), env=env)
        except ValueError as e:
            msg = str(e)
            # Detect action space mismatch like: "Discrete(8) != Discrete(6)"
            m = re.search(r"Discrete\((\d+)\)\s*!=\s*Discrete\((\d+)\)", msg)
            if m:
                ckpt_actions = int(m.group(1))
                env_actions = int(m.group(2))
                print(f"[WARN] Action space mismatch: checkpoint={ckpt_actions} vs env={env_actions}")

                # If checkpoint used 8 actions, rebuild env with extra_buttons=True
                if ckpt_actions == 8 and env_actions == 6:
                    print("[INFO] Rebuilding env with extra_buttons=True to match checkpoint…")
                    env_config["extra_buttons"] = True
                    env = make_env(env_config)()
                    model = PPO.load(str(ckpt), env=env)
                else:
                    raise
            else:
                raise
    else:
        print("[WARN] No checkpoints found — starting with a fresh PPO policy.")
        model = PPO("CnnPolicy", env, verbose=1)

    # Interactive loop (Gymnasium 5-tuple friendly)
    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated) or bool(truncated)
        else:
            obs, reward, done, info = step_out
        if done:
            obs, _ = env.reset()
