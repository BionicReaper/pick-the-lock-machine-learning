"""Tuning values for Pick the Lock.

Every field in StageConfig mirrors a key from the decompiled
  scripts/events/dark_carnival/lockpicking/game.vdata
(extracted from the local Dota 2 pak01_dir.vpk with Source 2 Viewer).
Values below are the live retail numbers as of 2026-07-07.

SimTuning holds *interpretation* constants: places where the compiled
client behaviour had to be inferred (units, ramp shapes, spawn model).
Each one is documented with the assumption made so it can be corrected
against the real game without touching sim logic.

BINARY-VERIFIED (2026-07-23): the game loop was recovered by disassembling
`game/dota/bin/win64/client.dll` (built from
`src/game/client/dota/panorama/dark_carnival/lockpicking/dota_lockpicking_schema.cpp`).
The relevant `CDOTALockpickingGame` methods, by virtual address:

  0x182ada830  Init(stage*)   - seeds every field, then tail-calls Reset
  0x182aec770  Reset()        - timer := interval, spawns the opening bar
  0x182aecdb0  Update(dt)     - the whole game loop
  0x182ac0d40  TryUnlock(&bonus) - the click handler
  0x182acd670  PickSpawnAngle(bool opening)
  0x1828cb490  popup OnThink  - calls Update(dt) with wall-clock frame dt

Comments tagged BINARY-VERIFIED below are read directly off that code and
outrank the older FOOTAGE-VERIFIED readings where the two disagree.
"""

from dataclasses import dataclass, field


@dataclass
class StageConfig:
    # --- pick rotation ---
    initial_speed: float = 40.0            # m_flInitialSpeed        (deg/s)
    speed_increment_per_unlock: float = 0.0  # m_flSpeedIncrementPerUnlock (deg/s per pick)
    max_speed_multiplier: float = 3.2      # m_flMaxSpeedMultiplier  (cap incl. boost)

    # --- speed boost (RMB hold) ---
    speed_boost_rate: float = 6.0          # m_flSpeedBoostRate
    speed_boost_percentage: float = 0.12   # m_flSpeedBoostPercentage

    # --- miss penalty ---
    deceleration_rate: float = 3.0         # m_flDecelerationRate
    recover_rate: float = 60.0             # m_flRecoverRate

    # --- bars (unlock markers) ---
    num_unlocks: int = 1000                # m_nNumUnlocks (win threshold; unreachable in 30s => endless)
    max_unlocks_on_board: int = 6          # m_nMaxUnlocksOnBoard
    base_unlock_appear_rate: float = 1.4   # m_flBaseUnlockAppearRate
    unlock_appear_increase_rate: float = 0.04  # m_flUnlockAppearIncreaseRate
    min_degrees_between_unlocks: float = 20.0  # m_flMinDegreesBetweenUnlocks
    unlock_radius: float = 40.0            # m_nUnlockRadius (marker size, board-space px; see SimTuning)
    board_radius: int = 180                # m_nBoardRadius (px, visual only)
    unlock_degree_decrease_rate: float = 3.0  # m_flUnlockDegreeDecreaseRate (deg/s shrink)

    # --- timer ---
    time_limit: float = 30.0               # m_flTimeLimit (s)
    timer_increase_per_unlock: float = 1.5  # m_flTimerIncreasePerUnlock (s, blue bars)
    timer_increase_unlock_chance: float = 15.0   # m_flTimerIncreaseUnlockChance (%)
    timer_increase_unlock_escalating_chance: float = 4.0  # m_flTimerIncreaseUnlockEscalatingChance (%/pity step)

    # --- scoring ---
    score_per_unlock: int = 1000           # m_nScorePerUnlock (stage override; top level was 10)


@dataclass
class SimTuning:
    """Interpretation constants — assumptions, not datamined facts."""

    # BINARY-VERIFIED: there is NO fixed timestep. The popup's OnThink
    # (0x1828cb490) does `dt = Plat_FloatTime() - m_flLastThink` and passes
    # that straight to Update(dt), so the real game updates once per rendered
    # frame with a variable wall-clock dt. `tick_rate` is therefore the
    # *assumed client framerate* we are emulating, and it genuinely changes
    # the physics — see boost_is_per_tick below.
    tick_rate: int = 60

    # BINARY-VERIFIED (Update @ 0x182aecef7): while RMB is held,
    #     delta = min(m_flSpeedBoostRate, max_speed - speed)
    #     speed += delta ; boost_accum += delta
    # There is NO `* dt` on that add — boost is m_flSpeedBoostRate deg/s
    # added *per Update call*, i.e. per rendered frame. At 60 fps that is
    # 360 deg/s^2; at 144 fps, 864 deg/s^2. The ramp really is framerate
    # dependent in retail.
    #
    # This contradicts the older footage reading (~240 deg/s^2, which would
    # correspond to ~40 fps capture). Set False to fall back to the
    # framerate-independent `rate * dt` interpretation.
    boost_is_per_tick: bool = True

    # BINARY-VERIFIED (Update @ 0x182aecf2a): released boost does NOT decay at
    # the boost rate. It decays proportionally to how much boost was banked:
    #     speed = max(base_speed, speed - m_flDecelerationRate * boost_accum * dt)
    # and boost_accum resets to 0 once speed is back at base.

    # BINARY-VERIFIED (TryUnlock @ 0x182ac0ded, Update @ 0x182aecebc):
    # a miss sets speed to *exactly zero* and latches a disabled flag. Speed
    # then climbs back at m_flRecoverRate deg/s^2 and the flag clears the
    # instant speed reaches base_speed again. So the disable is not a fixed
    # 0.55 s: it lasts base_speed / recover_rate = 40/60 = 0.667 s, and it
    # scales if m_flSpeedIncrementPerUnlock is ever non-zero. Clicks during
    # the whole window are swallowed (no miss is even registered), and RMB is
    # ignored — the recover branch takes priority over the boost branch.
    # Banked boost_accum is NOT cleared by the miss.
    #
    # Retained only so calibration.py's REACT_BAR_LIFETIME_S maths keeps
    # working; the sim no longer reads it.
    miss_disable_duration: float = 0.55   # s, DEPRECATED - see above

    # BINARY-VERIFIED (Update @ 0x182aed023): the spawn interval is *not*
    # 1.4 - 0.04*unlocks. There are two separate fields:
    #     m_flTimeUntilNextUnlock (timer)  and  m_flCurrentAppearRate (interval)
    # and the interval only shrinks when the timer FIRES, gated on a
    # "picked at least once since the last spawn" latch:
    #     timer -= dt
    #     if timer < 0:
    #         if picked_since_last_spawn:
    #             interval -= m_flUnlockAppearIncreaseRate   # once, not per pick
    #             picked_since_last_spawn = False
    #         timer = interval                                # reload, no carry
    #         if len(bars) < m_nMaxUnlocksOnBoard: spawn()
    # Consequences: picking a lock never shortens the countdown already in
    # flight; three picks inside one interval still only shave 0.04 s once;
    # and a cycle with no picks shaves nothing. The interval also keeps
    # shrinking on cycles where the board is full and nothing spawns.
    #
    # Retail applies no floor. 0.0 reproduces it (interval 0 => the timer
    # fires every frame), while a positive value clamps for experiments.
    min_spawn_interval: float = 0.0        # s

    # BINARY-VERIFIED (Reset @ 0x182aec770): exactly one bar exists at game
    # start, placed at uniform(m_flMinDegreesBetweenUnlocks, 120) deg rather
    # than uniformly around the dial, and the spawn timer starts at a FULL
    # interval (1.4 s), not at zero.
    spawn_immediately_at_start: bool = True
    opening_bar_angle_max: float = 120.0

    # BINARY-VERIFIED: no such rule exists. Update only ever spawns off the
    # interval timer, so an empty board simply stays empty until it fires.
    keep_board_nonempty: bool = False

    # BINARY-VERIFIED (TryUnlock @ 0x182ac0e1c): m_nDirection is stored as
    # 1/2 and flipped on every successful pick (`dir = (dir == 1) ? 2 : 1`);
    # Update negates the step when it is 2. Misses do not flip it.
    reverse_on_success: bool = True

    # BINARY-VERIFIED (TryUnlock @ 0x182ac0d90): the hit test is a plain
    # inclusive `center - half <= needle <= center + half` against the live
    # arc, with no slack. dota_lockpicking_unlock_marker_display_buffer is
    # never consulted by the hit test — it is a rendering-only convar.
    hit_buffer_deg: float = 0.0

    # BINARY-VERIFIED (Update @ 0x182aed0ab): the blue-bar roll happens at
    # SPAWN time, not on pick:
    #     chance = m_flTimerIncreaseUnlockChance
    #            + m_flTimerIncreaseUnlockEscalatingChance * misses_since_blue
    #     blue = chance >= RandomFloat(0, 100)     # note >=, not >
    #     misses_since_blue = 0 if blue else misses_since_blue + 1
    # Minimum scheduled travel distance accepted from the model (deg).
    min_target_distance_deg: float = 1.0

    # BINARY-VERIFIED (Init @ 0x182ada891): the bar's half-width is
    #     degrees(atan(m_nUnlockRadius / m_nBoardRadius)) = atan(40/180)
    #                                                     = 12.5288 deg
    # computed as the angle between (0, -180) and (40, -180) via
    # acos(dot/|a||b|) * 57.2957763671875. Full width is therefore
    # 25.0576 deg, not the 25.46 the small-angle reading gave.
    # Each edge closes at m_flUnlockDegreeDecreaseRate (3 deg/s, so 6 deg/s
    # of width) and the bar is dropped once the half-width goes <= 0, giving
    # a 12.5288/3 = 4.176 s lifetime.
    min_bar_width_deg: float = 0.0

    # BINARY-VERIFIED (board update @ 0x1828c6c70): the convar
    # `dota_lockpicking_unlock_marker_display_buffer` (default 1.0) is a
    # DISPLAY SHRINK, not a hit-test buffer:
    #     drawn_half = bar.half_width - buffer
    #     if drawn_half <= 0: the marker is not drawn at all
    #     panel gets (start = center - drawn_half, sweep = 2 * drawn_half)
    # So retail renders each bar 1 deg narrower per side than the arc that
    # actually registers a pick — you get 1 deg of forgiveness past the
    # visible edge on both sides. The hit test itself has no slack.
    #
    # This also pins the visible lifetime: (12.5288 - 1.0)/3 = 3.843 s, i.e.
    # 115.3 frames at 30 fps — matching the 115 counted on
    # youtu.be/1FuZkn8iLhA. The bar stays hittable for the full 4.176 s.
    #
    # Rendering only (game.py); the sim's hit test uses the true width.
    marker_display_buffer_deg: float = 1.0


DEFAULT_STAGE = StageConfig()
DEFAULT_TUNING = SimTuning()
