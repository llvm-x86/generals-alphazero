<div align="center">

<p align="center">
  <img src="https://raw.githubusercontent.com/strakam/generals-bots/master/generals/assets/images/game1.webp" width="250" alt="Self-play game 1" />
  <img src="https://raw.githubusercontent.com/strakam/generals-bots/master/generals/assets/images/game2.webp" width="250" alt="Self-play game 2" />
  <img src="https://raw.githubusercontent.com/strakam/generals-bots/master/generals/assets/images/game3.webp" width="250" alt="Self-play game 3" />
</p>

## **Generals.io Bots**

[Installation](#-installation) • [Getting Started](#-getting-started) • [Environment](#-environment) • [Deployment](#-deployment)
</div>

A high-performance JAX-based simulator for [generals.io](https://generals.io), designed for reinforcement learning research.

**Highlights:**
* ⚡ **10M+ steps/second** — fully JIT-compiled JAX simulator with vectorized `vmap` for massive parallelism
* 🎯 **Pure functional design** — immutable state, reproducible trajectories
* 🚀 **Live deployment** — deploy agents to [generals.io](https://generals.io) servers
* 🎮 **Built-in GUI** — visualize games and debug agent behavior

## 📦 Installation

```bash
git clone https://github.com/llvm-x86/generals-alphazero
cd generals-alphazero
pip install -e '.[alphazero]'
```

> [!Note]
> This repository is based on the [generals.io](https://generals.io) game.
> The goal is to provide a fast bot development platform for reinforcement learning research.
> The simulator follows the current rules of the game: 10,000 ranked games played in 2026 replay
> through it with a board identical to generals.io's own engine after every turn.

## AlphaZero-style research baseline

`examples/az_selfplay.py` trains a pixel-level spacetime transformer (one token per cell per frame
over the last 4 own observations, a global token for value, no pooling or
convolution) on Gumbel-root policy-improvement targets and self-play; interior search uses PUCT.
Search samples hidden terrain, enemy territory/armies and general position **from each player's
observation and public totals**, not the true hidden board. Root candidates share sampled
worlds/replies; interior opponent replies are sampled. Start with a small smoke run:

```bash
PYTHONPATH=. python examples/az_selfplay.py --games 1 --sims 2 --size 4 --max-steps 12 --output checkpoint.pt
# Continue to a total of 2 games, including optimizer, replay buffer and RNG states:
PYTHONPATH=. python examples/az_selfplay.py --games 2 --sims 2 --size 4 --max-steps 12 --output checkpoint.pt --resume
```

Training boards come from `make_env(size, max_steps)`: about 10 castles where the board
allows (`size*size//6` on tiny boards), `min_generals_distance = max(3, size//2)`. Search's hidden
castle prior (`CASTLE_FRACTION`: 0.65 for 6x6, 0.63 for 8x8) was measured as true castles over
`structures_in_fog` cells in ~400 Expander-vs-Expander states per size on those boards.

Evaluate against built-in agents (random, expander, hunter, harvester) or a checkpoint
(`ckpt:<path>`, policy argmax, or `--sims N` for search), on side-swapped map pairs.
Real wins/losses/draws have Wilson intervals and match score has a paired-map bootstrap interval;
material adjudication of draws is reported separately and is **not** a real win:

```bash
PYTHONPATH=. python examples/eval_agents.py expander random --size 8 --games 200 --max-steps 500
```

Memory and belief. Every frame carries 5 memory planes (ever-seen, last-seen enemy cells, last-seen
mountains, last-seen castles, sticky enemy-general sighting; `CHANNELS = 20`), updated from the player's
own observations and carried through the frame history, search nodes and self-play. Checkpoints record
`channels`; loading one with a different count (e.g. from before memory planes) raises a clear error.
An auxiliary belief head predicts the enemy general's cell, masked to still-plausible unseen
cells; it is trained (weight 0.1) with cross-entropy against the true cell only until that general
has been seen, and the true cell is never a net input. `--belief on` (also `eval_agents.py
--belief`) samples its hidden location from that head when not already known; remembered
terrain and a re-hidden general remain fixed, but enemy territory can reoccupy seen cells.
Held-out check
(`examples/belief_experiment.py`, 8x8 Expander-vs-Expander, split by game): 400 train / 100 validation / 300 held-out games
(798 train after adding 400 more), 3 epochs chosen on validation: mean P(true general) 0.136 learned vs 0.027
uniform over the plausible cells (5.05x) on 8,433 held-out rows. In a 112-game side-swapped A/B vs Expander
(6x6, untrained net, 8 sims) `--belief` on and off gave identical results, so search impact is unmeasured.

Behaviour-cloning warm start (`examples/bc_warmstart.py`): clones Expander (half the games with 15% random-move noise,
clean label) from both seats' histories, split by game. To train with those weights, use
`az_selfplay.py --init-weights bc.pt --output az.pt` with matching `--size/--attn/--frames`; this starts
fresh replay/optimizer state. Legacy BC replay cannot be resumed as AlphaZero data. Measured on 6x6, 150 steps, 150 games (29k train samples, 10 epochs):
held-out teacher-action accuracy 0.317 (Expander samples its move, so accuracy is capped); policy-argmax vs Random,
100 games: 14 W / 13 L / 73 D (Expander itself: 74/2/24) -- the 90% bar was NOT met.

BC diagnosis (6x6, seeds 5000.. validation / 9000.. test, train maps 100000+, 100 side-swapped games each): the 150-step
limit is the first bottleneck -- Expander vs Random is 75/1/24 at 150 steps but 98/2/0 at 300 and 500. Second, Expander's
move is a score-proportional *sample* (about uniform over owned->owned moves when nothing can be captured), so labels are
noise (held-out accuracy 0.31) and the policy's argmax is a much weaker agent than the sampler: the BC student shuffles
armies (39% of its moves reverse the previous one vs 16% for Expander; it holds 4-20 cells vs 14-26). Deterministic
labels (`--teacher argmax`) are learnable (accuracy 0.80) but that teacher is itself weak (32% wins at 150, 64% at 300).
No variant reached the 90% bar; best was a *sampled* policy from a 300-step sampled-teacher net (test 49 W / 15 L / 36 D).
`--dagger-rounds N` relabels student-visited states with the teacher.

Expander does not hunt the general: on 8x8 at `--max-steps 200` roughly 44% of its games vs Random
are truncated draws, so use a longer limit when you want real captures.

Real wins/losses train the value head with +1/-1; unfinished games at `--max-steps` are
**draws (0)** in the default finite-horizon objective. `--truncation proxy` is opt-in and
changes that objective. The default net sees 4 frames of memory; sampled hidden-world trees
are not an information-set solver. Checkpoints store versioned replay, raw outcomes, search
semantics and code identity; incompatible legacy replay is evaluation-only. Load only checkpoints
you trust. A corrected 6x6, 200-step, 50-game self-play control did **not** show strength gains:
g50 vs g0 drew 40/40 prior-only games and 20/20 eight-simulation games on side-swapped maps.
On 24 held-out games the g50 value MSE was 0.306 versus 0.278 for constant zero.
This is not a ranked-ready agent; evaluate on held-out maps and stronger opponents before live use.
`--attn pisa` swaps dense attention for a pure-PyTorch implementation of
[PISA](https://arxiv.org/abs/2609.31093) (pyramid top-K block-sparse attention;
mean-pooled key pyramid, LogSumExp-scored coarse-to-fine block selection).
Tokens are ordered frame-major, so PISA also spans the history axis:
`--frames N` sets how many past own-observation frames the net attends over
(default 4). Tests confirm PISA equals dense attention when all blocks are kept
and that the pyramid finds a planted key. Measured on CPU (1 thread, 4 layers,
width 64, batch 1, single cold forward pass, so noisy), 30x30 board:

| history | tokens | dense | PISA |
|---|---|---|---|
| 4 frames | 3,601 | 1.32s | 3.65s |
| 8 frames | 7,201 | 4.90s | 7.11s |
| 16 frames | 14,401 | 20.63s | 11.91s |

PISA is slower than dense for short sequences and wins only for long
board-times-history sequences (here from about 16 frames at 30x30). It has no
fused Triton GPU kernel, and replay storage grows linearly with `--frames`
(16 frames at 30x30 is about 0.9 MB per sample). Dense and PISA checkpoints, and
checkpoints with different `--frames`, are not interchangeable.

Two opt-in training options (defaults unchanged): `--value-target mix` trains the
value head on `--value-lambda` (default 0.5) times the outcome (or truncation
score proxy) plus the remainder times the root search value (mean backed-up Q
at the roots); the mixed target is what the replay stores, so checkpoints keep
their format. `--hybrid-order GPGP` sets the `--attn hybrid` layer types
(G = GDN-2, P = PISA; length 4, at least one of each; default alternates G,P).
Both are recorded in the checkpoint and must match on `--resume`.

`--attn hybrid` interleaves [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)
layers with PISA layers (GDN-2, PISA, GDN-2, PISA). GDN-2 runs causally over the
frame axis independently for every cell, with a fixed-size delta-rule state
(channel-wise decay plus separate erase and write gates, eq. 9 of the paper), so
the last frame's cell token summarises the whole history. PISA mixes across
space and recent frames. Tests check the scan against the explicit matrix form
of the paper's update, that repeated writes to one key overwrite rather than
accumulate, that the write gate is channel-selective, and that the oldest frame
reaches the current cell's logits. On the 30x30, 16-frame benchmark above the
hybrid took 4.61s, but it has only two attention layers against four in the
dense/PISA rows, so that is not a like-for-like speed comparison, and nothing
here shows it plays better. Simplifications versus the paper: no short
convolution or output gate, and a plain PyTorch loop over frames instead of the
chunkwise Triton kernels. By default the state is rebuilt from the frame window on every
call; see "Incremental hybrid" below for the opt-in streaming mode.

### Incremental hybrid (`--attn hybrid --incremental`, opt-in)

`Net(..., stream=True)` is a *streaming* variant: `forward` folds the window frame by
frame from an empty state, and `Net.step(frame, state)` does the same for one new frame
from a carried state. GDN layers carry `S` (fixed size, O(1) in history); PISA layers carry
the last `frames-1` frames' cell keys/values and only the newest frame's tokens are queries
(so, unlike the default net, old frames never see new ones: causal in time, all frames share
one temporal embedding). Search nodes store the batch-2 (me, opponent) state, so a child costs
one step; the root opponent state starts empty. Training uses `forward` on stored windows
(a rebuild, zero initial state); self-play/search use the carried state.

- Equal (`test_incremental_equals_window_rebuild_when_window_covers_game`, atol 1e-4 on
  logits, value, belief): whenever the window holds every frame stepped so far.
- **Not equal** once the game outlives the window (`test_incremental_state_outlives_the_window`):
  the carried GDN `S` and cached PISA keys still contain pre-window frames (and everything
  computed after them depends on that), while training and `eval_agents` rebuild from the
  window only. So self-play/search with `--incremental` run on a different input distribution
  than training once a game is longer than `--frames` turns (including the 4-turn-padded start, where
  `stack_with` repeats frame 0 in the window but the carried state saw it once). Trained-only-on-windows
  weights are not guaranteed to use long-range state well; this is not measured here.
- The stream net is a different function from the default hybrid, so checkpoints are not interchangeable
  (`args["incremental"]` is recorded; `bc_warmstart.py` does not support it).

Per-step time, 1 thread, B=1, default width, median of 5 after one excluded cold call
(throwaway script timing `net(window)` vs `net.step(frame, state)`):

| board | frames | window rebuild | incremental step |
|---|---|---|---|
| 8x8 | 16 | 191.4 ms | 11.4 ms |
| 8x8 | 64 | 980.4 ms | 15.9 ms |
| 12x12 | 16 | 426.8 ms | 27.0 ms |
| 12x12 | 64 | 2048.5 ms | 29.8 ms |

`uv run proofs/information_flow.py` runs symengine checks of the layers' information
flow (GDN-2 memory dynamics and contractivity, softmax logit gap dense vs PISA,
exhaustive reachability depth over layer orderings, state-size vs KV-cache
break-even). They prove properties of the layer geometry, not that a model plays
better; `test_gdn2_scan_matches_symbolic_closed_form` ties the GDN result to the
shipped `gdn2_scan`.

Earlier one-off checkpoints lack optimizer/RNG state and use a different input
shape; start a fresh run rather than using `--resume` on them.

The simulator, assets, documentation and original examples come from
[strakam/generals-bots](https://github.com/strakam/generals-bots) by Matej
Straka, under the [MIT license](LICENSE). Its authors report tile-for-tile
validation on [10,000 ranked replays](paper/validation/README.md); the
AlphaZero-style trainer is an independent addition, not an official
generals.io implementation.

## 🌱 Getting Started

### Basic Game Loop

```python
import jax.numpy as jnp
import jax.random as jrandom

from generals import GeneralsEnv, get_observation
from generals.agents import RandomAgent, ExpanderAgent

# Create environment (customize grid size and truncation)
env = GeneralsEnv(grid_dims=(10, 10), truncation=500)

# Create agents
agent_0 = RandomAgent()
agent_1 = ExpanderAgent()

# Initialize — reset returns the auto-reset pool plus the first state
key = jrandom.PRNGKey(42)
pool, state = env.reset(key)

# Game loop
while True:
    # Get observations
    obs_0 = get_observation(state, 0)
    obs_1 = get_observation(state, 1)

    # Get actions
    key, k1, k2 = jrandom.split(key, 3)
    action_0 = agent_0.act(obs_0, k1)
    action_1 = agent_1.act(obs_1, k2)
    actions = jnp.stack([action_0, action_1])

    # Step environment (auto-resets from the pre-generated pool)
    timestep, state = env.step(state, actions, pool)

    if timestep.terminated or timestep.truncated:
        break

print(f"Winner: Player {int(timestep.info.winner)}")
```

### ⚡Vectorized Parallel Environments

Run **thousands** of games in parallel using `jax.vmap`:

```python
import jax
import jax.random as jrandom
from generals import GeneralsEnv, get_observation

# Create single environment
env = GeneralsEnv(grid_dims=(10, 10), truncation=500)

# Generate state pool once, then create per-env starting states
NUM_ENVS = 1024
key = jrandom.PRNGKey(0)
key, pool_key = jrandom.split(key)
pool, _ = env.reset(pool_key)  # generates the shared pool

keys = jrandom.split(key, NUM_ENVS)
states = jax.vmap(env.init_state)(keys)  # Batched states

# Step all environments in parallel (auto-resets from the shared pool)
# ... get batched observations and actions ...
step_vmap = jax.vmap(lambda s, a: env.step(s, a, pool))
timesteps, states = step_vmap(states, actions)
```

See `examples/vectorized_example.py` for a complete example.

### 👥 Teams and Free-For-All

The same env plays 1v1 (the default), N-player free-for-all, and team games:

```python
env = GeneralsEnv(grid_dims=(15, 15))                   # 1v1
env = GeneralsEnv(grid_dims=(15, 15), num_players=4)    # 4-player free-for-all
env = GeneralsEnv(grid_dims=(15, 15), teams=[0, 0, 1, 1])   # 2v2: players 0+1 vs 2+3
```

With N players, actions are `(N, 5)`, `state.ownership` is `(N, H, W)`, and
observations and rewards are stacked `(N, ...)`. Rules beyond 1v1:

* Moving onto a **teammate's** cell pools the armies and hands the cell to the mover.
* **Capturing a general** transfers all of the victim's cells to the capturer with
  every army halved (rounded up); the general becomes a castle and the victim is
  eliminated (their actions are ignored from then on). The game goes on while
  another team is alive.
* A team loses only when **every** one of its generals has fallen; the **last
  team standing** wins. `info.winner` is the winning team id (the player index
  in 1v1 / free-for-all), and every player on that team gets reward `+1`,
  everyone else `-1`.
* **Sight is shared** within a team. Observations carry `allied_cells`,
  `allied_land_count` and `allied_army_count` (all zero when you have no
  teammate); `opponent_*` covers every enemy team together.

See `examples/multiplayer_example.py` for batched 2v2 / FFA / 1v1 games under `jax.jit`.

## 🌍 Environment

### Observation

Each player receives an `Observation` with these fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `armies` | `(H,W)` | Army counts in visible cells |
| `generals` | `(H,W)` | Mask of visible generals |
| `castles` | `(H,W)` | Mask of visible castles (formerly `cities` — a deprecated alias remains) |
| `mountains` | `(H,W)` | Mask of visible mountains |
| `owned_cells` | `(H,W)` | Mask of cells you own |
| `opponent_cells` | `(H,W)` | Mask of opponent's visible cells |
| `neutral_cells` | `(H,W)` | Mask of neutral visible cells |
| `fog_cells` | `(H,W)` | Mask of fog (unexplored) cells |
| `structures_in_fog` | `(H,W)` | Mask of castles/mountains in fog |
| `owned_land_count` | scalar | Total cells you own |
| `owned_army_count` | scalar | Total armies you have |
| `opponent_land_count` | scalar | Opponent's cell count |
| `opponent_army_count` | scalar | Opponent's army count |
| `timestep` | scalar | Current game step |
| `allied_cells` | `(H,W)` | Mask of teammates' visible cells (team games; all-False otherwise) |
| `allied_land_count` | scalar | Teammates' cell count |
| `allied_army_count` | scalar | Teammates' army count |

`obs.as_tensor()` stacks the first 14 fields into a `(14, H, W)` tensor;
`obs.as_tensor(include_allied=True)` appends the three allied channels.

### Action

Actions are arrays of 5 integers: `[pass, row, col, direction, split]`

| Index | Field | Values |
|-------|-------|--------|
| 0 | `pass` | `1` to pass, `0` to move |
| 1 | `row` | Source cell row |
| 2 | `col` | Source cell column |
| 3 | `direction` | `0`=up, `1`=down, `2`=left, `3`=right |
| 4 | `split` | `1` to send half army, `0` to send all-1 |

Use `compute_valid_move_mask` to get legal moves:

```python
from generals import compute_valid_move_mask

mask = compute_valid_move_mask(obs.armies, obs.owned_cells, obs.mountains)
# mask shape: (H, W, 4) - True where move from (i,j) in direction d is valid
```

## Live UI and local protocol server

The live human site is [generals.io](https://generals.io), backed by
`https://ws.generals.io/`. WebBridge can open the **real browser UI** under
your existing session:

```bash
loom webbridge call navigate --args '{"url":"https://generals.io","newTab":true}'
```

The DOM board is `.game-cursor-table`; inspect visible cells/leaderboard with
WebBridge `evaluate`. For board moves, get a cell's `getBoundingClientRect()`
center and send CDP `Input.dispatchMouseEvent` `mousePressed` then
`mouseReleased` at that position: synthetic DOM `click` did not move armies
in the live tutorial, but CDP mouse events did (land increased from 1 to 2).
PLAY on a fresh account starts the tutorial.
The [2025 research paper](https://arxiv.org/html/2507.06825v2) reports
thousands of matches against humans, but the current
[official API FAQ](https://dev.generals.io/api) says bots are allowed only on
the bot server. Public-server automation is **explicit opt-in** and may be
disallowed or penalized. Do not put your user ID in source control:

```bash
PYTHONPATH=. python examples/client_example.py --user_id "$GENERALS_USER_ID" --lobby_id my_private_game --endpoint https://ws.generals.io/
# Ranked queue (the example ExpanderAgent is not competitive):
PYTHONPATH=. python examples/client_example.py --user_id "$GENERALS_USER_ID" --endpoint https://ws.generals.io/ --ranked
```

For offline private 1v1 protocol tests, run:

```bash
python -m generals.remote.local_server --host 127.0.0.1 --port 8080 --seed 0 --size 10
PYTHONPATH=. python examples/client_example.py --user_id alice --lobby_id test --endpoint http://127.0.0.1:8080
# In another process, use --user_id bob with the same lobby and endpoint.
```

The same `GeneralsIOClient` defaults to the bot server and accepts
`endpoint=` for local or explicitly chosen public-server use. The local server
speaks Socket.IO Engine.IO v4 over HTTP polling and implements private 1v1
join, force-start, queued moves, half-army attacks, fogged `game_update`
diffs, current simultaneous general trades, and game results. State
transitions use the upstream JAX engine. It does **not** emulate ranked
matchmaking, browser assets, maps/modifiers, team modes, WebSocket transport,
or the unpublished production server's queue and RNG behavior. Full
replacement diffs decode to the same arrays but are not byte-for-byte
identical to production deltas. There is no captured live-match trace proving
1:1 protocol or outcome parity. Keep this limitation explicit before using
the local server for training or evaluation.

## 📄 Citation

```bibtex
@misc{generals_rl,
      author    = {Matej Straka, Martin Schmid},
      title     = {Artificial Generals Intelligence: Mastering Generals.io with Reinforcement Learning},
      year      = {2025},
      eprint    = {2507.06825},
      archivePrefix = {arXiv},
      primaryClass = {cs.LG},
}
```
