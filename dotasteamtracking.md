# Pick the Lock — Dota 2 Dark Carnival Minigame Datamine

Internal name: **Lockpicking** · Event: Dark Carnival (June 2026) · Compiled: 2026-07-07

Sources scanned: [SteamDatabase/GameTracking-Dota2](https://github.com/SteamDatabase/GameTracking-Dota2), [spirit-bear-productions/dota_vpk_updates](https://github.com/spirit-bear-productions/dota_vpk_updates)

---

## 1. Architecture — where the logic lives

- The game loop is **compiled C++ in the Dota 2 client** (`client.dll`), exposed as the Panorama panel class `DOTADarkCarnivalEncounterLockpickingPopup`. There is no Lua or Panorama JS for it — no repo contains the literal loop code.
- It is **data-driven**: the client reads tuning values from `scripts/events/dark_carnival/lockpicking/game.vdata_c` (2,099 bytes, inside `pak01_dir.vpk`). **This file is not tracked by any public repo** — it is the only missing piece.
- Scoring/leaderboards are server-acknowledged (friends high-score leaderboard in the UI), but all minigame simulation is client-side.

## 2. The parameter model (from GameTracking-Dota2 schema dumps)

### [CDOTALockpickingGameDefinition.h](https://github.com/SteamDatabase/GameTracking-Dota2/blob/master/DumpSource2/schemas/client/CDOTALockpickingGameDefinition.h) — top-level game

| Field | Meaning |
|---|---|
| `m_vecStages` | List of stage definitions |
| `m_successEffect` / `m_failEffect` | Particle systems on pick success/fail |
| `m_nScorePerUnlock` | Points per successful pick |

### [CDOTALockpickingStageDefinition.h](https://github.com/SteamDatabase/GameTracking-Dota2/blob/master/DumpSource2/schemas/client/CDOTALockpickingStageDefinition.h) — per-stage tuning

Created in build 6839 (2026-06-27), never modified since.

**Pick rotation**

| Field | Meaning |
|---|---|
| `m_flInitialSpeed` | Starting rotation speed |
| `m_flSpeedIncrementPerUnlock` | Speed-up per successful pick |
| `m_flMaxSpeedMultiplier` | Speed cap |

**Speed boost (right-click hold)**

| Field | Meaning |
|---|---|
| `m_flSpeedBoostRate` | Boost ramp rate |
| `m_flSpeedBoostPercentage` | Boost magnitude |

**Miss penalty ("briefly disables the pick")**

| Field | Meaning |
|---|---|
| `m_flDecelerationRate` | Slowdown on miss |
| `m_flRecoverRate` | Recovery back to speed |

**Bars (unlock markers) on the board**

| Field | Meaning |
|---|---|
| `m_nNumUnlocks` | Bar count baseline (initial/maintained bars — the "minimum bars" mechanic; exact semantics live in compiled code) |
| `m_nMaxUnlocksOnBoard` | Max bars at once |
| `m_flBaseUnlockAppearRate` / `m_flUnlockAppearIncreaseRate` | Spawn rate of new bars, accelerating |
| `m_flMinDegreesBetweenUnlocks` | Minimum angular separation between bars |
| `m_nUnlockRadius` | Bar size |
| `m_nBoardRadius` | Board size |
| `m_flUnlockDegreeDecreaseRate` | Bars shrink over time |

**Timer**

| Field | Meaning |
|---|---|
| `m_flTimeLimit` | Game length |
| `m_flTimerIncreasePerUnlock` | Time granted by blue bars |
| `m_flTimerIncreaseUnlockChance` / `m_flTimerIncreaseUnlockEscalatingChance` | Chance a spawned bar is blue, with pity escalation |

**Scoring**

| Field | Meaning |
|---|---|
| `m_nScorePerUnlock` | Per-stage score override |

### Supporting types

- [ELockpickingStageMode.h](https://github.com/SteamDatabase/GameTracking-Dota2/blob/master/DumpSource2/schemas/client/ELockpickingStageMode.h): `INVALID = 0`, `TIME_ATTACK = 1` — only one real mode shipped
- `CDOTA_Lockpicking_EffectsEntity` — empty model entity for the FX scene
- Client-only: no server-side schema classes exist for the minigame

## 3. UI & engine wiring (from dota_vpk_updates)

From the decompiled [popup_dark_carnival_encounter_lockpicking.xml](https://github.com/spirit-bear-productions/dota_vpk_updates/blob/master/panorama/layout/events/dark_carnival/popup_dark_carnival_encounter_lockpicking.xml):

- **Engine commands**: `DOTALockpickingSetMode(time_attack)` + `DOTAEncounterMinigameStart()` (Play / Play Again), `DOTALockpickingTogglePaused()` (F9 / Esc), `DOTAEncounterMinigameExit()`
- **Board structure**: `LockpickingBoard` → `UnlockMarkerContainer` (populated with `UnlockMarker` / `UnlockMarkerRange` snippets from code), `LockpickPanel` with `LockpickStick`, particle overlay `lockpick_2.vpcf`
- **HUD bindings**: `+{f:1:time_increase}` popup label, `{f:1:time_remaining}`, incrementing score label, `{d:score_per_unlock}` in the scoring help, `{d:unlock_count} x {d:score_per_unlock}` in post-game
- **Screens**: intro menu (how-to-play, scoring, friends leaderboard, rewards list with `{d:reward_score_required}` thresholds), pause menu, post-game (score breakdown, new-high-score flair, rewards, leaderboard)
- **3D FX**: `DOTALockpickingScenePanel` rendering map `maps/scenes/dark_carnival/lockpicking_fx.vpk`; Slark head is an `AnimatedImageStrip` (30 ms/frame, 420×500 frames)

## 4. Rules as documented in localization (`resource/localization/dota_english.txt`)

- Title: *"Pick the Lock"* — *"Help Slark break out of his cell!"*
- *"Successfully pick the lock as many times as you can before time runs out."*
- **LMB**: *"Activate the lock pick when it reaches the highlighted bar. Missing the bar briefly disables the pick. Blue bars grant additional time."*
- **RMB (hold)**: *"Hold for continuous speed boost."*
- Score = Successful Picks × score-per-unlock; Win / Lose / Game Over states exist

## 5. Convars & commands (GameTracking `convars.txt` / `commands.txt`)

| Convar / command | Notes |
|---|---|
| `dota_lockpicking_use_frame_update false` | **The framerate-bug artifact**: at launch the pick position updated per rendered frame, making the game harder at high FPS (players cheesed it by capping FPS). Valve switched to fixed-timestep updates; this toggle remains, default off. |
| `dota_lockpicking_unlock_marker_display_buffer 1` | Visual buffer on marker hit zones |
| `dota_dark_carnival_encounter_lockpicking` | Dev command to launch the minigame directly |
| `dota_allow_single_player_minigames false` | Gate for minigames in single-player games |

All are marked `developmentonly clientdll`.

## 6. Assets (from `pak01_dir.txt` VPK listing)

- **Particles** (~26 systems in `particles/events/dark_carnival/lockpicking/`): pick trail (`lockpick_fine_line*`), success sparks (`lockpick_spark_*`), boost FX (`lockpick_speed_left/right*`), and a full `lockpicking_failed_explosion_*` suite — the lock blows up on failure
- **Sounds**: `lockpick_boost/break/success.vsnd` + Slark voice lines (`slark_lockpick_start_02–06`, `success_01–07`, `gameover_01–05`, `story_01–06`)
- **Images**: lock dial/shank/background, Slark arm + animated head, in `panorama/images/events/dark_carnival/lockpicking/`

## 7. What is still unobtainable publicly

The **actual numbers** in `game.vdata_c` (initial speed, time limit, blue-bar chance, etc.). Getting them requires decompiling that one file from a real Dota 2 install's `pak01_dir.vpk` with [Source 2 Viewer](https://valveresourceformat.github.io/) — its KV3 keys are exactly the schema field names above with live values filled in.

## Related context

- The event's existence leaked pre-launch via the variable `m_bHackAllPlayersLoadedForDarkCarnivalPreLaunchModifierApplication`, which Valve renamed to `m_bHackWhyAreYouGuysReadingOurVariableNames` in build 6815 ([commit 082b68f](https://github.com/SteamDatabase/GameTracking-Dota2/commit/082b68fb81efc708476401c902d6eb3f5aca99fa), 2026-06-20) as a jab at dataminers.
- Framerate-bug coverage: [Escorenews minigame guide](https://escorenews.com/en/dota-2/article/79123-how-to-beat-pick-the-lock-and-boot-breaker-guide-to-winning-mini-games-in-dota-2-dark-carnival-event)
