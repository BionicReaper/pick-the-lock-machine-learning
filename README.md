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
.venv\Scripts\python.exe play.py --ai models/best_genome.pkl   # watch the AI
```

Controls: **LMB** pick, **RMB (hold)** speed boost, **F9/Esc** pause,
**F3** debug overlay (hidden variables, AI inputs/outputs, AI target ghost),
**Space** start/restart.

## Layout

| Path | Purpose |
|---|---|
| `pickthelock/config.py` | Every `game.vdata` value + documented interpretation assumptions |
| `pickthelock/sim.py` | Headless fixed-timestep game logic (single source of truth) |
| `pickthelock/controller.py` | Cancellable "click after X° with Y boost" scheduler |
| `pickthelock/observations.py` | NEAT input vector (18) / output decoding (3) |
| `pickthelock/game.py`, `assets.py` | pygame client using the extracted Dota art/sounds |
| `train_neat.py`, `neat_config.txt` | NEAT training (parallel, checkpointed) |
| `smoke_test.py` | Determinism + heuristic-bot sanity checks |
| `extracted/` | Decompiled Dota assets (`game.vdata`, PNGs, MP3s, layout XML/CSS) |

## The rules (datamined values + footage-verified semantics)

Semantics were verified by frame-by-frame tracking of retail gameplay footage
(beam angle + marker wedges across 2,259 frames at 30 fps):

- 30 s time limit; pick rotates at 40°/s base (measured ~36–37 at game
  start), up to ×3.2 with boost (measured ~120–140°/s sustained).
- **Boost (RMB)**: multiplier ramps at `m_flSpeedBoostRate` = 6×/s → full
  boost in ~0.37 s, decays equally fast on release (measured ±240°/s²).
  `m_flSpeedBoostPercentage` (0.12) has no observable mapping yet — unused.
- **Miss**: pick turns red and is disabled ~0.55 s (fitted) while speed
  decays exponentially (`m_flDecelerationRate` = 3/s, never a full stop);
  boost is lost; then speed recovers (`m_flRecoverRate` = 60°/s², plus boost
  re-ramp). The missed bar is not consumed.
- **Direction reverses on every successful pick** (observed repeatedly);
  misses do not reverse. `SimTuning.reverse_on_success` to toggle.
- **Bars**: `m_nUnlockRadius` = 40 is a half-arc in px at the 180 px board
  radius → initial width 2×40/180 rad ≈ **25.5°** (measured 22–24 at spawn).
  Both edges close at 3°/s (width −6°/s), despawn at ≈3° → ~3–3.7 s life.
  Max 6 on board (rarely binds — good players clear them), ≥20° apart.
- **Spawning**: interval timer; interval = `1.4 s − 0.04 s × unlocks`
  (measured gaps ~1.35 s early → ~0.6 s after ~24 picks). The raw formula
  hits zero at 35 unlocks; the sim clamps at `min_spawn_interval` = 0.1 s
  (the real game's floor is unknown). A bar spawns at game start, and an
  empty board refills within ~0.1 s.
- Blue bars: 15% chance, +4% pity per orange spawn (reset on blue — pity
  direction is still an assumption); a blue pick grants +1.5 s.
- Score 1000 per pick (menu confirms; rewards at 6k/12k/18k/24k).
- `m_nNumUnlocks = 1000` = win threshold (unreachable in practice ⇒ endless).
- Hit test gets ±1° slack (`dota_lockpicking_unlock_marker_display_buffer`).

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
