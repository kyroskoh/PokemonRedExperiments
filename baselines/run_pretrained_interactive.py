from pathlib import Path
import os
import uuid
import re
import time
import random
from collections import deque

from red_gym_env import RedGymEnv
from stable_baselines3 import PPO


# ----------------------------- Helpers -----------------------------

def make_env(env_conf, seed=0):
    def _init():
        env = RedGymEnv(env_conf)
        env.reset(seed=seed)
        return env
    return _init


def build_env_config(repo_root, headless=False, extra_buttons=False):
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
        "extra_buttons": extra_buttons,      # may flip to True to match checkpoint
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
    """Load PPO model; if action-space mismatch 6!=8, rebuild env with extra_buttons=True and retry."""
    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    try:
        return PPO.load(str(ckpt_path), env=env), env, env_config
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
                return PPO.load(str(ckpt_path), env=env), env, env_config
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
    """Single scalar for ‘progress’; combine reward + tiny exploration term."""
    tr = float(getattr(env, "total_reward", 0.0))
    ex = float(explore_count(env)) * 1e-3  # tiny tie-breaker
    return tr + ex


def spawn_fresh_env(env, env_config, repo_root: Path):
    """Close existing env and create a new one (new session folder)."""
    try:
        env.close()
    except Exception:
        pass
    time.sleep(0.1)
    new_conf = build_env_config(
        repo_root,
        headless=env_config["headless"],
        extra_buttons=env_config.get("extra_buttons", False),
    )
    new_env = make_env(new_conf)()
    return new_env, new_conf


# ----------------------- Micro anti-oscillation -----------------------

class OscillationGuard:
    """
    Detect quick ‘ping-pong’ loops (e.g., UP/DOWN) or low tile variety,
    then enter a short ‘escape’ mode with randomized actions that avoid
    repeating the last action and its opposite.
    """
    def __init__(self, window=160, min_unique=6, escape_steps=96):
        self.window = window
        self.min_unique = min_unique
        self.escape_steps = escape_steps
        self.pos_hist = deque(maxlen=window)
        self.escape_left = 0
        self.last_action = None

    @staticmethod
    def _is_abab(seq):
        # Detect A,B,A,B,... pattern on positions
        if len(seq) < 6:
            return False
        a = seq[-1]
        b = seq[-2]
        if a == b:
            return False
        # check last 6 positions alternate a,b,a,b,a,b
        return all((seq[-i] == (a if i % 2 == 1 else b)) for i in range(1, 7))

    def update(self, env):
        # pull live position from env’s memory
        try:
            x = int(env.read_m(globals()["X_POS_ADDRESS"]))
            y = int(env.read_m(globals()["Y_POS_ADDRESS"]))
            m = int(env.read_m(globals()["MAP_N_ADDRESS"]))
        except Exception:
            # if anything fails, don’t break control loop
            return

        self.pos_hist.append((m, x, y))

        # already escaping
        if self.escape_left > 0:
            self.escape_left -= 1
            return

        # evaluate micro-stuck: too few unique tiles or ABAB bounce
        unique_cnt = len(set(self.pos_hist))
        if unique_cnt <= self.min_unique or self._is_abab(list(self.pos_hist)):
            # trigger escape
            self.escape_left = self.escape_steps

    def choose_action(self, proposed_action: int, env):
        """
        Return the final action to execute.
        If escaping, sample an action that is not the last action nor its opposite.
        Otherwise return the model's proposed_action.
        """
        if self.escape_left <= 0:
            self.last_action = proposed_action
            return proposed_action

        num_actions = env.action_space.n
        candidates = list(range(num_actions))

        # avoid repeating the same action
        if self.last_action is not None and self.last_action in candidates:
            candidates.remove(self.last_action)

        # avoid immediate opposite (up<->down, left<->right)
        opposite = {0: 1, 1: 0, 2: 3, 3: 2}
        if self.last_action in opposite and opposite[self.last_action] in candidates:
            candidates.remove(opposite[self.last_action])

        if not candidates:
            candidates = list(range(num_actions))
        act = random.choice(candidates)
        self.last_action = act
        return act


# ------------------------------ Main ------------------------------

if __name__ == "__main__":
    THIS_FILE = Path(__file__).resolve()
    REPO_ROOT = THIS_FILE.parents[1]

    # Base env (interactive SDL2 window)
    env_config = build_env_config(REPO_ROOT, headless=False, extra_buttons=False)
    env = make_env(env_config)()

    # Checkpoint handling (auto)
    ckpt = newest_checkpoint(REPO_ROOT)
    if ckpt:
        model, env, env_config = try_load_model(ckpt, env, env_config, REPO_ROOT)
    else:
        print("[WARN] No checkpoints found — starting with a fresh PPO policy.")
        model = PPO("CnnPolicy", env, verbose=1)

    # Long-horizon watchdog (no progress => respawn)
    STUCK_WINDOW   = 3000   # steps without progress before respawn
    REWARD_EPS     = 1e-4   # minimal reward improvement to count
    MIN_EXPL_DELTA = 3      # minimal exploration increase to count
    MAX_GENERATIONS = 50    # safety cap

    # Short-horizon anti-oscillation guard
    osc_guard = OscillationGuard(window=160, min_unique=6, escape_steps=96)

    generation = 1
    last_progress = progress_score(env)
    last_expl = explore_count(env)
    steps_since_progress = 0

    # Interactive loop (Gymnasium 5-tuple friendly)
    obs, _ = env.reset()
    while generation <= MAX_GENERATIONS:
        # Update oscillation state before picking action
        osc_guard.update(env)

        # Be exploratory only during escape mode
        deterministic = not (osc_guard.escape_left > 0)
        proposed_action, _ = model.predict(obs, deterministic=deterministic)

        # Let guard override action during escape
        action = osc_guard.choose_action(proposed_action, env)

        step_out = env.step(action)
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
            # ease the watchdog a bit on episode reset
            steps_since_progress = max(0, steps_since_progress - 256)

        # watchdog: respawn if stuck too long
        if steps_since_progress >= STUCK_WINDOW:
            print(f"\n[WATCHDOG] No progress in {STUCK_WINDOW} steps. "
                  f"Respawning env (generation {generation+1})…")

            env, env_config = spawn_fresh_env(env, env_config, REPO_ROOT)
            model.set_env(env)  # reuse weights with new env

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
