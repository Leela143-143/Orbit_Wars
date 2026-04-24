# HTMRL — Hierarchical Temporal Memory Reinforcement Learning

An HTM (Hierarchical Temporal Memory) based reinforcement learning agent designed for the **Orbit Wars** Kaggle simulation environment.

The agent combines a **Spatial Pooler** (for sparse encoding of game state) with a **Temporal Memory** module (for sequence prediction), applying HTM theory to multi-agent competitive strategy.

---

## Project Structure

```
HTMRL/
├── HTMRL/                      # Core HTM library
│   ├── __init__.py
│   ├── spatial_pooler.py       # Sparse Distributed Representation encoder
│   ├── temporal_memory.py      # Sequence learning via temporal context
│   └── log.py                  # Logging utilities
│
├── htmrl_agent.py              # HTMRLAgent — game-aware agent wrapper
├── train.py                    # ELO-based self-play population training
├── play_single_game.py         # Run a single game to test the agent
├── build_kaggle_submission.py  # Package agent into a Kaggle submission
├── best_bot.py                 # Standalone bot for Kaggle submission
└── README.md
```

---

## How It Works

### 1. State Encoding
Each planet's local view is encoded into a **40-bit binary vector**:
- Bits 0–11: Enemy planets in each of 12 directional sectors
- Bits 12–23: Neutral planets per sector
- Bits 24–35: Friendly planets per sector
- Bits 36–39: Own ship count (discretized into 4 bins)

### 2. Spatial Pooler
The binary observation is fed through a **Spatial Pooler** that produces a **Sparse Distributed Representation (SDR)** — a small set of active columns from a 2048-column pool. This provides noise robustness and semantic similarity.

### 3. Temporal Memory (per-planet timelines)
Each planet maintains its **own temporal context** (`tm_states` dictionary). The Temporal Memory module learns sequential patterns by maintaining predictive cell states, enabling the agent to anticipate future game states.

### 4. Action Selection
The active SDR columns are mapped to one of **25 discrete actions**:
- `0`: Do nothing
- `1–12`: Send 50% of ships in one of 12 angular directions
- `13–24`: Send 100% of ships in one of 12 angular directions

### 5. Population Training
`train.py` runs **ELO-rated self-play** across a population of 100 bots:
- Bots are matched against close-ELO opponents
- Matches run in parallel across all CPU cores
- The best bot by ELO is saved as `best_bot.pkl`

---

## Setup

### Prerequisites
- Python 3.8+
- `kaggle-environments` (install separately)

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/HTMRL.git
cd HTMRL

# Install dependencies
pip install numpy scipy tqdm

# Install kaggle-environments
pip install kaggle-environments
```

---

## Usage

### Train a population of bots (self-play)
```bash
python train.py
```
Runs `MATCHES` matches across `NUM_BOTS` bots in parallel. Saves best bot to `best_bot.pkl`.

### Play a single game
```bash
python play_single_game.py
```

### Build a Kaggle submission
```bash
python build_kaggle_submission.py
```

---

## Configuration

Key constants in `train.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_BOTS` | 100 | Population size |
| `MATCHES` | 500 | Total training matches |

Key constants in `htmrl_agent.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INPUT_SIZE` | 40 | Observation encoding size |
| `NUM_ACTIONS` | 25 | Action space size |
| `cell_count` | 2048 | Spatial Pooler columns |
| `active_count` | 40 | Active columns per SDR |

---

## References

- [Numenta HTM Theory](https://numenta.com/resources/htm-school/)
- [Orbit Wars — Kaggle Simulation](https://www.kaggle.com/competitions/kore-2022)
- Original HTM codebase by [JakobStruye](https://github.com/JakobStruye/HTMRL)
