# Pick the Lock — standalone + NEAT trainer

A standalone recreation of the Dota 2 Dark Carnival minigame **Pick the Lock**
(internal name *Lockpicking*), plus a NEAT neuroevolution trainer that learns
to play it. All tuning values were decompiled from the local Dota 2 install
(`pak01_dir.vpk` → `scripts/events/dark_carnival/lockpicking/game.vdata`) with
Source 2 Viewer; textures and sounds come from the same VPK and live in
`extracted/` (personal local use).

## Quick start

```
.venv\Scripts\python.exe smoke_test.py                 # sanity-check the sim
.venv\Scripts\python.exe play.py                       # play it yourself
.venv\Scripts\python.exe train_neat.py --smoke         # verify training pipeline
.venv\Scripts\python.exe train_neat.py --generations 100
.venv\Scripts\python.exe play.py --ai                  # watch the AI (infers the genome)
```

Controls: **LMB** pick, **RMB (hold)** speed boost, **F9/Esc** pause,
**F3** debug overlay (hidden variables, AI inputs/outputs, AI target ghost),
**Space** start/restart.

### Model files (`models/`)

Training artifacts are organised into three parallel trees, each keyed by the
I/O schema and the human knobs
`<schema>/reaction_time_ms/reaction_time_standard_deviation/inaccuracy`
(defaults → `0/0.0/0.05/0.0`); see `pickthelock/paths.py`.

| Tree | Contents |
|---|---|
| `models/saved/<schema>/<knobs>/<index>_<timestamp>_<score>_best_genome.pkl` | promoted best genomes for a parameter set; the index auto-increments so runs never overwrite |
| `models/temp/<schema>/<knobs>/<run_id>/` | per-run scratch: `best_genome.pkl`, `winner_genome.pkl`, `fitness_history.csv`, `checkpoints/` |
| `models/graphs/<schema>/<knobs>/…​.html` | `graph_genome.py` renders here, mirroring the saved filename |

A training run writes into its own `temp/…/<run_id>/` (the run id is the process
id, so parallel runs never collide); on termination — normal or Ctrl+C — the
best genome is promoted into `saved/` under a unique
`<index>_<timestamp>_<score>_best_genome.pkl` name (a `--smoke` run skips this).
`play.py --ai` with no path infers the genome from the human knobs, picking the
one with `--index N` (default 0); pass an explicit path to `--ai` or to
`graph_genome.py` to target a specific file. So the usual loop is just
`train_neat.py …` then `play.py --ai` with the same knobs.

## Layout

| Path | Purpose |
|---|---|
| `pickthelock/config.py` | Every `game.vdata` value + documented interpretation assumptions |
| `pickthelock/sim.py` | Headless fixed-timestep game logic (single source of truth) |
| `pickthelock/controller.py` | Cancellable "click after X° with Y boost" scheduler |
| `pickthelock/observations.py` | NEAT input vector (18) / output decoding (3) |
| `pickthelock/game.py`, `assets.py` | pygame client using the extracted Dota art/sounds |
| `pickthelock/paths.py` | `models/` layout: the `saved`/`temp`/`graphs` trees keyed by human knobs |
| `train_neat.py`, `neat_config.txt` | NEAT training (parallel, checkpointed) |
| `smoke_test.py` | Determinism + heuristic-bot sanity checks |
| `extracted/` | Decompiled Dota assets (`game.vdata`, PNGs, MP3s, layout XML/CSS) |

## The rules (datamined values + binary-verified semantics)

Semantics were originally fitted from frame-by-frame tracking of retail
footage (beam angle + marker wedges across 2,259 frames at 30 fps), then
**read directly out of `client.dll`** — class `CDOTALockpickingGame`, from
`src/game/client/dota/panorama/dark_carnival/lockpicking/dota_lockpicking_schema.cpp`.
Method virtual addresses are cited in `config.py`. The disassembly overturned
several of the footage fits; where they disagree, the binary wins.

- 30 s time limit; pick rotates at 40°/s base, up to ×3.2 with boost.
- **Update is per rendered frame with wall-clock `dt`** — there is no fixed
  timestep. `SimTuning.tick_rate` is the *assumed client framerate*, and it
  really does change the physics (below).
- **Boost (RMB)**: `speed += m_flSpeedBoostRate` (6) **per Update call, with
  no `× dt`** — so the ramp is framerate dependent: 360°/s² at 60 fps,
  864°/s² at 144 fps. Full boost takes 15 frames regardless. On release,
  speed bleeds down by `m_flDecelerationRate × banked_boost × dt` until it
  reaches base. `m_flSpeedBoostPercentage` (0.12) is confirmed dead code.
- **Miss**: speed is set to **exactly 0** and the pick is latched off until
  it climbs back at `m_flRecoverRate` (60°/s²) — i.e. `base/recover` =
  **0.667 s**, not a fixed 0.55 s. Clicks in that window are swallowed
  without even registering a miss, and RMB is ignored. Banked boost survives
  the miss. The missed bar is not consumed.
- **Direction reverses on every successful pick**; misses do not reverse.
- **Bars**: half-width is `atan(40/180)` = **12.53°** (full **25.06°**), not
  the small-angle 25.46°. Each edge closes at 3°/s (width −6°/s) and the bar
  is dropped once the half-width hits 0 → **4.18 s** life. Max 6 on board.
  The hit test is inclusive with **no slack** —
  `dota_lockpicking_unlock_marker_display_buffer` is rendering-only. When
  bars overlap, the most recently spawned one is picked.
- **Spawning**: the countdown and the interval are **separate fields**, and
  the interval is *not* `1.4 − 0.04 × unlocks`:
  ```
  timer -= dt
  if timer < 0:
      if picked_since_last_spawn:
          interval -= 0.04          # at most once per cycle
          picked_since_last_spawn = False
      timer = interval              # reload, nothing carries over
      if len(bars) < 6: spawn()
  ```
  So picking a lock **never shortens the countdown already running**; three
  picks inside one interval still shave 0.04 only once; a cycle with no picks
  shaves nothing; and the interval keeps shrinking on cycles where the board
  is full and nothing spawns. Retail applies no floor.
- **Spawn angles are rejection-sampled**: uniform on [0, 360), retried while
  within `m_flMinDegreesBetweenUnlocks` (20°) of an existing bar, giving up
  after 20 tries and using that conflicting angle anyway — so the 20°
  spacing is a soft constraint. Exactly one bar exists at t=0, placed at
  `uniform(20, 120)`, and it never rolls blue. An empty board does **not**
  refill early; it waits for the timer.
- Blue bars: rolled at spawn, `15 + 4 × spawns_since_blue ≥ rand(0, 100)`;
  a blue pick grants +1.5 s.
- Score 1000 per pick (menu confirms; rewards at 6k/12k/18k/24k).
- `m_nNumUnlocks = 1000` = win threshold (unreachable in practice ⇒ endless).

## NEAT model interface

Prompted at episode start, on every **new bar spawn**, and on **target
reached**. Between prompts the cancellable controller drives the sim.

**Inputs (20):** for each of the 3 nearest bars — sorted by *travel needed
until a click would hit* (0 while inside the hit zone, so a just-passed but
still hittable bar stays in slot 0 instead of jumping to "360° away"):
forward distance/360, reverse distance/360, is-blue, width/initial width;
then in-zone flag (click now would hit), time remaining/30, boost
multiplier (norm.), penalty factor, pick-disabled flag, spawn interval/1.4,
blue pity chance/100, current speed/max speed. Missing bar slots
pad with (1, 1, 0, 0).

**Outputs (3):** target distance (×360°), average speed = fraction of the
travel with RMB held (held first, released after — 0.5 ⇒ boost for the first
half), and click-at-target (>0.5).

**Fitness:** 7 headless sims per genome (same seed set for all genomes in a
generation, rotating each generation):
`0.5·avg + 0.25·worst + 0.25·best` — weights are constants atop
`train_neat.py`.

## Re-extracting assets

Source 2 Viewer CLI, e.g.:

```
Source2Viewer-CLI.exe -i "C:\...\dota 2 beta\game\dota\pak01_dir.vpk" ^
  --vpk_filepath "scripts/events/dark_carnival/lockpicking" -d -o extracted
```
