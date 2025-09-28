# Train RL agents to play Pokemon Red

### New 10-19-24! Updated & Simplified V2 Training Script - See V2 below
### New 1-29-24! - [Multiplayer Live Training Broadcast](https://github.com/pwhiddy/pokerl-map-viz/)  🎦 🔴 [View Here](https://pwhiddy.github.io/pokerl-map-viz/)
Stream your training session to a shared global game map using the [Broadcast Wrapper](/baselines/stream_agent_wrapper.py)  

See how in [Training Broadcast](#training-broadcast) section
  
## Baseline Updates (v1 & v2)

### v1 (`baselines/run_baseline_parallel_fast.py`, `RunBaseline.ps1`)
- Resolves ROM and state paths relative to the repo root (`PokemonRed.gb`, `has_pokedex_nballs.state`).
- Auto-selects the newest checkpoint via `rglob("poke_*_steps.zip")`.
- Rebuilds the environment when action spaces mismatch (6<->8 buttons) and tears down subprocesses safely first.
- Watchdog now grants a 512-step grace period after respawn, resets anti-oscillation, and improves SDL2 teardown.
- PPO rollout buffer reduced (`n_steps=512`, `batch_size=256`) to prevent 15GBmemory spikes while keeping TensorBoard logging intact.

### v2 (`v2/baseline_fast_v2.py`, `RunBaselineV2.ps1`)
- Built on the `StreamWrapper` so map streaming is on by default and lighter.
- Faster, memory-savvy training with lower reward scale, coordinate rewards, and trimmed rollout size.
- Keeps checkpoint auto-detect, action-space repair (6<->8), and safe `SubprocVecEnv` teardown from v1.
- Compatible with TensorBoard and optional Weights & Biases logging.
- Convenience PowerShell launcher targets Python 3.11 by default.

## Baseline Training

### v1 Baseline
Run the parallel baseline trainer:
```powershell
.\RunBaseline.ps1
```
+
**Changes in v1 (`baselines/run_baseline_parallel_fast.py`):**
- Fixed ROM/state paths (`REPO_ROOT / "PokemonRed.gb"`).
- Auto-detects newest checkpoint with `rglob("poke_*_steps.zip")`.
- Handles action-space mismatch (`Discrete(8) != Discrete(6)` and inverse) by rebuilding env with/without `extra_buttons`.
- Safe SubprocVecEnv teardown before rebuild (prevents subprocess leaks).
- Watchdog improvements:
  - Grace period after respawn (ignores "no progress" for first 512 steps).
  - Resets anti-oscillation guard after respawn.
  - More robust SDL2 teardown with sleep/guard checks.
- Reduced PPO rollout size (`n_steps=512`, `batch_size=256`) to avoid huge (15GB+) buffers.
- Preserves SB3 TensorBoard logging and custom `TensorboardCallback`.

### v2 Baseline
Run the new v2 trainer:
```powershell
.\RunBaselineV2.ps1
```
**Changes in v2 (`v2/baseline_fast_v2.py`):**
- Uses `StreamWrapper` to stream env state to map viewer.
- Faster and leaner training:
  - Coordinate-based exploration reward (replaces frame KNN).
  - Lower reward scale tuned exploration.
  - Optimized rollout size for speed and reduced memory.
- Includes all fixes from v1 (checkpoint auto-detect, action-space handling, safe teardown, reduced rollout).
- Supports TensorBoard and optional WandB logging.
- Adds helper script `RunBaselineV2.ps1` to launch with Python 3.11.
✅ **Impact:** Training is now faster, more stable, and uses less memory. Envs won’t get stuck oscillating, and v2 baseline provides improved exploration real-time streaming.


**Impact**
- Training stability improves, oscillations resolve automatically, and memory usage stays manageable.
- v2 delivers richer rewards plus real-time streaming without extra setup, making it the recommended path going forward.
## Watch the Video on Youtube! 

<p float="left">
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/youtube.jpg?raw=true" height="192">
  </a>
  <a href="https://youtu.be/DcYLT37ImBY">
    <img src="/assets/poke_map.gif?raw=true" height="192">
  </a>
</p>

## Join the discord server
[![Join the Discord server!](https://invidget.switchblade.xyz/RvadteZk4G)](http://discord.gg/RvadteZk4G)
  
## Running the Pretrained Model Interactively 🎮  
🐍 Python 3.10is recommended. Other versions may work but have not been tested.   
You also need to install ffmpeg and have it available in the command line.

### Windows Setup
Refer to this [Windows Setup Guide](windows-setup-guide.md)

### For AMD GPUs
Follow this [guide to install pytorch with ROCm support](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/howto_wsl.html)

### Linux / MacOS

V2 is now recommended over the original version. You may follow all steps below but replace `baselines` with `v2`.

1. Copy your legally obtained Pokemon Red ROM into the base directory. You can find this using google, it should be 1MB. Rename it to `PokemonRed.gb` if it is not already. The sha1 sum should be `ea9bcae617fdf159b045185467ae58b2e4a48b9a`, which you can verify by running `shasum PokemonRed.gb`. 
2. Move into the `baselines/` directory:  
 ```cd baselines```  
3. Install dependencies:  
```pip install -r requirements.txt```  
It may be necessary in some cases to separately install the SDL libraries.
For V2 MacOS users should use ```macos_requirements.txt``` instead of ```requirements.txt```
4. Run:  
```python run_pretrained_interactive.py```
  
Interact with the emulator using the arrow keys and the `a` and `s` keys (A and B buttons).  
You can pause the AI's input during the game by editing `agent_enabled.txt`

Note: the Pokemon.gb file MUST be in the main directory and your current directory MUST be the `baselines/` directory in order for this to work.

## Training the Model 🏋️ 

<img src="/assets/grid.png?raw=true" height="156">


### V2

- Trains faster and with less memory
- Reaches Cerulean
- Streams to map by default
- Other improvements

Replaces the frame KNN with a coordinate based exploration reward, as well as some other tweaks.
1. Previous steps but in the `v2` directory instead of `baselines`
2. Run:
```python baseline_fast_v2.py```

## Tracking Training Progress 📈

### Training Broadcast
Stream your training session to a shared global game map using the [Broadcast Wrapper](/baselines/stream_agent_wrapper.py) on your environment like this:
```python
env = StreamWrapper(
            env, 
            stream_metadata = { # All of this is part is optional
                "user": "super-cool-user", # choose your own username
                "env_id": id, # environment identifier
                "color": "#0033ff", # choose your color :)
                "extra": "", # any extra text you put here will be displayed
            }
        )
```

Hack on the broadcast viewing client or set up your own local stream with this repo:  
  
https://github.com/pwhiddy/pokerl-map-viz/

### Local Metrics
The current state of each game is rendered to images in the session directory.   
You can track the progress in tensorboard by moving into the session directory and running:  
```tensorboard --logdir .```  
You can then navigate to `localhost:6006` in your browser to view metrics.  
To enable wandb integration, change `use_wandb_logging` in the training script to `True`.

## Static Visualization 🐜
Map visualization code can be found in `visualization/` directory.

## Follow up work  
 
Check out our follow up projects & papers!  
  
### [Pokemon Red via Reinforcement Learning 🔗](https://arxiv.org/abs/2502.19920)
```  
  @misc{pleines2025pokemon,
    title={Pokemon Red via Reinforcement Learning},
    author={Marco Pleines and Daniel Addis and David Rubinstein and Frank Zimmer and Mike Preuss and Peter Whidden},
    year={2025},
    eprint={2502.19920},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
  }
```
### [Pokemon RL Edition 🔗](https://drubinstein.github.io/pokerl/)
### [PokeGym 🔗](https://github.com/PufferAI/pokegym)

## Supporting Libraries
Check out these awesome projects!
### [PyBoy](https://github.com/Baekalfen/PyBoy)
<a href="https://github.com/Baekalfen/PyBoy">
  <img src="/assets/pyboy.svg" height="64">
</a>

### [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3)
<a href="https://github.com/DLR-RM/stable-baselines3">
  <img src="/assets/sblogo.png" height="64">
</a>


