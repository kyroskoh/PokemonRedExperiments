from pathlib import Path
import os
import uuid
import re
import time

from red_gym_env import RedGymEnv
from stable_baselines3 import PPO


def make_env(env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=seed)
        return env
    return _init


def build_env_config(repo_root, headless=False, extra_buttons=False):
    DEFAULT_ROM = repo_root / "PokemonRed.gb"
    DEFAULT_STATE = repo_root / "has_pokedex_nballs.state"

    from pathlib import Path as _Path
    gb_path = _Path(os.getenv("GB_PATH", str(DEFAULT_ROM))).expanduser().resolve()
    init_state = _Path(os.getenv("STATE_PATH", str(DEFAULT_STATE))).expanduser().resolve()

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
        "extra_buttons": extra_buttons,      # we’ll flip this when needed
        "explore_weight": 3,
    }


def newest_checkpoint(repo_root: Path):
    """Return newest poke_*_steps.zip or None."""
    cands = sorted(
        repo_root.rglob("poke_*_steps.zip"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return cands[0] if cands else None


def try_load_model(ckpt_path: Path, env, env_config, repo_root: Path):
    """Load a PPO model; if action-space mismatch 6!=8, rebuild env with extra_buttons=True and retry."""
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    try:
        return PPO.load(str(ckpt_path), env=env), env
    except ValueError as e:
        msg = str(e)
        m = re.search(r"Discrete\((\d+)\)\s*!=\s*Discrete\((\d+)\)", msg)
        if m:
            ckpt_actions = int(m.group(1))
            env_actions = int(m.group(2))
            print(f"[WARN] Action space mismatch: checkpoint={ckpt_actions} vs env={env_actions}")
            if ckpt_actions == 8 and env_actions == 6:
                print("[INFO] Rebuilding env with extra_buttons=True to match checkpoint…")
                env_config = build_env_config(repo_root, headless=False, extra_buttons=True)
                env = make_env(env_config)()
                return PPO.load(str(ckpt_path), env=env), env
        raise


def explore_count(env: RedGymEnv) -> int:
    """How much we've explored (frame-KNN or coord set), best-effort."""
    try:
        if getattr(env, "use_screen_explore", False):
            return env.knn_index.get_current_count()
        return len(getattr(env, "seen_coords", {}))
    except Exception:
        return 0


def progress_score(env: RedGymEnv) -> float:
    """Single scalar for ‘progress’; we combine env.total_reward + small weight on exploration."""
    # total_reward is maintained in env.get_game_state_reward() usage
    tr = float(getattr(env, "total_reward", 0.0))
    # add a tiny exploration term to break ties
    ex = float(explore_count(env)) * 1e-3
    return tr + ex


def spawn_fresh_env(env, env_config, repo_root: Path):
    """Close existing env and create a new one (new session folder)."""
    try:
        env.close()
    except Exception:
        pass
    time.sleep(0.1)
    # new session root
    new_conf = build_env_config(
        repo_root,
        headless=env_config["headless"],
        extra_buttons=env_config.get("extra_buttons", False),
    )
    new_env = make_env(new_conf)()
    return new_env, new_conf


if __name__ == "__main__":
    THIS_FILE = Path(__file__).resolve()
    REPO_ROOT = THIS_FILE.parents[1]

    # --- base env (interactive SDL2 window) ---
    env_config = build_env_config(REPO_ROOT, headless=False, extra_buttons=False)
    env = make_env(env_config)()

    # --- checkpoint handling (auto) ---
    ckpt = newest_checkpoint(REPO_ROOT)
    if ckpt:
        model, env = try_load_model(ckpt, env, env_config, REPO_ROOT)
    else:
        print("[WARN] No checkpoints found — starting with a fresh PPO policy.")
        model = PPO("CnnPolicy", env, verbose=1)

    # --- anti-stuck watchdog settings ---
    STUCK_WINDOW = 3000            # steps without progress before we respawn
    REWARD_EPS = 1e-4              # minimal improvement threshold
    MIN_EXPL_DELTA = 3             # allow tiny exploration increments to count
    MAX_GENERATIONS = 50           # safety: avoid infinite respawns

    generation = 1
    last_progress = progress_score(env)
    last_expl = explore_count(env)
    steps_since_progress = 0

    # --- interactive loop (Gymnasium 5-tuple friendly) ---
    obs, _ = env.reset()
    while generation <= MAX_GENERATIONS:
        action, _ = model.predict(obs, deterministic=True)
        step_out = env.step(action)

        # tuple compatibility
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated) or bool(truncated)
        else:
            obs, reward, done, info = step_out

        # progress tracking
        cur_score = progress_score(env)
        cur_expl = explore_count(env)

        improved = (cur_score - last_progress) > REWARD_EPS or (cur_expl - last_expl) >= MIN_EXPL_DELTA
        if improved:
            last_progress = cur_score
            last_expl = cur_expl
            steps_since_progress = 0
        else:
            steps_since_progress += 1

        # episode boundary
        if done:
            obs, _ = env.reset()
            # reset stuck window a little on episode reset
            steps_since_progress = max(0, steps_since_progress - 256)

        # watchdog: respawn if stuck too long
        if steps_since_progress >= STUCK_WINDOW:
            print(f"\n[WATCHDOG] No progress in {STUCK_WINDOW} steps. Respawning env (generation {generation+1})…")

            env, env_config = spawn_fresh_env(env, env_config, REPO_ROOT)
            # reuse the same model (weights), just bind it to the new env
            # SB3 models keep env reference; we can just set a new vec env:
            model.set_env(env)

            # reset trackers for the new generation
            obs, _ = env.reset()
            last_progress = progress_score(env)
            last_expl = explore_count(env)
            steps_since_progress = 0
            generation += 1

    print("[INFO] Max generations reached; exiting.")
    try:
        env.close()
    except Exception:
        pass
