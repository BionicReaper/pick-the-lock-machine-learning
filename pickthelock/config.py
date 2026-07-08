"""Tuning values for Pick the Lock.

Every field in StageConfig mirrors a key from the decompiled
  scripts/events/dark_carnival/lockpicking/game.vdata
(extracted from the local Dota 2 pak01_dir.vpk with Source 2 Viewer).
Values below are the live retail numbers as of 2026-07-07.

SimTuning holds *interpretation* constants: places where the compiled
client behaviour had to be inferred (units, ramp shapes, spawn model).
Each one is documented with the assumption made so it can be corrected
against the real game without touching sim logic.
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
    unlock_radius: float = 40.0            # m_nUnlockRadius (interpreted: full arc width, deg)
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

    # Fixed timestep, Hz. Valve moved the minigame to fixed-timestep updates
    # (dota_lockpicking_use_frame_update defaults false); actual rate unknown.
    tick_rate: int = 60

    # FOOTAGE-VERIFIED: while RMB is held the speed multiplier ramps linearly
    # at m_flSpeedBoostRate = 6 x/s toward max_speed_multiplier (3.2 => full
    # boost in ~0.37 s) and decays at the same rate on release (measured
    # ~+-240 deg/s^2 both ways at base speed 40).
    # m_flSpeedBoostPercentage (0.12) is not yet mapped to an observable
    # effect; it is currently unused.

    # FOOTAGE-VERIFIED: miss penalty — the pick turns red and is disabled for
    # ~0.55 s while speed decays exponentially at deceleration_rate (never a
    # full stop), then clicks re-enable and speed recovers at recover_rate
    # deg/s^2 (with the boost re-ramping on top if RMB is held). Boost is
    # lost on the miss.
    miss_disable_duration: float = 0.55   # s, fitted from red-beam frames

    # FOOTAGE-VERIFIED (fitted): spawns are on an interval timer; the
    # frequency starts at 1/m_flBaseUnlockAppearRate (1/1.4 = 0.71 Hz) and
    # increases by m_flUnlockAppearIncreaseRate per successful unlock:
    #   f = 1/1.4 + 0.04 * unlocks
    # (measured spawn gaps: ~1.35 s early game -> ~0.6 s after ~24 unlocks)

    # FOOTAGE-VERIFIED: one bar spawns right at game start, and when the
    # board empties a new bar appears almost immediately (~0.1 s).
    spawn_immediately_at_start: bool = True
    keep_board_nonempty: bool = True

    # FOOTAGE-VERIFIED: the pick's travel direction REVERSES on every
    # successful pick (misses do not reverse). Set False to disable.
    reverse_on_success: bool = True

    # dota_lockpicking_unlock_marker_display_buffer 1 — hit zone is given
    # this many degrees of slack beyond the visible arc, per side.
    hit_buffer_deg: float = 1.0

    # ASSUMPTION: blue-bar pity escalates per *spawned* orange bar and resets
    # when a blue spawns.
    # Minimum scheduled travel distance accepted from the model (deg).
    min_target_distance_deg: float = 1.0

    # FOOTAGE-VERIFIED: m_nUnlockRadius=40 is the bar's half-arc in *pixels*
    # at m_nBoardRadius=180 px => initial width = 2*40/180 rad ~= 25.46 deg
    # (measured 22-24 deg at spawn with conservative color thresholds).
    # Both edges close in at m_flUnlockDegreeDecreaseRate (3 deg/s each,
    # width shrinks 6 deg/s) and the bar despawns at this floor width,
    # giving the observed ~3-3.7 s lifetime.
    min_bar_width_deg: float = 3.0


DEFAULT_STAGE = StageConfig()
DEFAULT_TUNING = SimTuning()
