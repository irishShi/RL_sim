# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

RL_sim is a reinforcement learning system for optimizing cellular network handover parameters in high-speed rail scenarios. A train travels from cell A to cell B along a 1D track (3000m), and a Rainbow DQN agent learns optimal handover hysteresis (Hys) and time-to-trigger (TTT) settings to minimize outages and unnecessary handovers.

## Language

The codebase, documentation, comments, and commit messages are primarily in **Chinese (中文)**. Follow this convention.

## Commands

```bash
# Install dependencies (PyTorch must be installed separately with CUDA support)
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Collect offline training data
python scripts/data/collect_data.py --num_episodes 100 --output_path data/datasets/offline_dataset.npz

# Train offline (with optional CQL regularization)
python scripts/train/train_rainbow_offline.py --dataset_path data/datasets/offline_dataset.npz --use_cql

# Train online (real-time environment interaction)
python scripts/train/train_rainbow.py

# Evaluate: single episode comparison (RL vs traditional A3)
python scripts/eval/test_simple.py

# Evaluate: batch statistics with plots
python scripts/eval/test_batch_comparison.py
```

## Architecture

The system follows a three-stage offline RL pipeline: **collect data → train offline → evaluate**.

### Environment (`envs/`)
- **`train_ho_env.py`** — Gymnasium-compatible `TrainHandoverEnv`. Each step: apply Hys/TTT action → check A3 event → check TTT timer → execute handover if triggered → compute RSRP/SINR → compute reward. Tracks KPIs (outage time, HO count, ping-pong events).
- **`channel_model.py`** — RSRP/SINR calculation: path loss + correlated AR(1) shadowing + weather loss + co-channel interference.
- **`ho_logic.py`** — A3 event detection and TTT timer management with protection time between handovers.
- **`weather_model.py`** — Temperature, humidity, PM2.5 effects on additional path loss.

### Models (`models/`)
- **`rainbow_model.py`** — `RainbowWithForecast`: GRU encoder (128 hidden) → shared FC (256) → Rainbow DQN head (C51 distributional + dueling + NoisyLinear) + auxiliary ΔRSRP forecast head. Forecast loss is weighted at 0.3× the Rainbow loss.
- **`action_space.py`** — Maps 48 discrete actions to (Hys, TTT) pairs: 8 Hys values × 6 TTT values.
- **`observation_builder.py`** — Maintains a 15-step sliding window of 10 features per timestep, producing a `[15, 10]` tensor for the GRU.

### Utils (`utils/`)
- **`replay_buffer.py`** — Prioritized Experience Replay (PER) buffer.
- **`c51_projection.py`** — C51 distributional Bellman projection onto 51-atom support.
- **`dataset_loader.py`** — Loads offline `.npz` datasets into the replay buffer.
- **`scenario_generator.py`** — Pre-generates and pickles scenarios for reproducible evaluation.

### Configs (`configs/`)
- **`default_env_config.yaml`** — Environment parameters: track geometry, speed range, power, channel model, reward weights, outage thresholds.
- **`model_config.yaml`** — Network architecture, training hyperparameters (lr=6.25e-5, batch=32, n_steps=3, gamma=0.99), PER settings, C51 atoms.

## Key Domain Concepts

- **A3 Event**: Handover trigger condition — neighboring cell RSRP exceeds serving cell RSRP by more than Hys (hysteresis) for longer than TTT (time-to-trigger).
- **Action space**: 48 actions = 8 Hys values {1.5–5.0 dB} × 6 TTT values {0–650 ms}.
- **Observation**: 15-step window × 10 features (RSRP serving/neighbor, ΔRSRP, SINR, velocity, position, time since last HO, current Hys/TTT, temperature).
- **Reward (R3 multi-stage)**: Base = normalized SINR. Penalties: outage (-15.0), severe degradation (-4.0), minor degradation (-1.5), interruption per slot (-6.0), handover (-0.2).
- **Outage/RLF**: SINR below -6 dB for 200ms continuous triggers an outage event.

## Evaluation Pattern

`test_batch_comparison.py` compares the learned policy against multiple traditional A3 parameter combinations using pre-generated scenarios (via `ScenarioGenerator`) for fair comparison. Metrics: average reward, HO count, average/min SINR, outage ratio. Results are saved as PNG plots and JSON statistics.
