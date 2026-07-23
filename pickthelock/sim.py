"""Headless fixed-timestep simulation of Pick the Lock.

Tuned against (a) the decompiled game.vdata values and (b) frame-by-frame
analysis of retail gameplay footage (see README "Interpretation" section).

Coordinate conventions:
  - Angles in degrees, [0, 360). 0 = straight up, increasing clockwise.
  - `direction` is +1 (clockwise) or -1; it FLIPS on every successful pick
    (observed in footage; SimTuning.reverse_on_success to disable).
  - `total_traveled` accumulates absolute degrees moved regardless of
    direction and is the basis for scheduled clicks ("click after X deg").

The sim knows nothing about rendering or neural nets. Inputs each tick:
  - `rmb_held` attribute (speed boost)
  - `click()` calls (pick attempt)
Outputs: event tuples returned from tick()/click().
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .config import StageConfig, SimTuning, DEFAULT_STAGE, DEFAULT_TUNING

# Event type constants (first element of event tuples)
EV_BAR_SPAWNED = "bar_spawned"
EV_BAR_EXPIRED = "bar_expired"
EV_PICKED = "picked"
EV_MISSED = "missed"
EV_TIMER_BONUS = "timer_bonus"
EV_GAME_OVER = "game_over"
EV_TARGET_REACHED = "target_reached"   # emitted by the controller, not the sim

# Penalty phases
PEN_NONE = 0
PEN_DECEL = 1      # pick disabled (red), speed decays exponentially
PEN_RECOVER = 2    # pick usable again, speed climbing back


def ang_norm(a: float) -> float:
    return a % 360.0


def ang_diff(a: float, b: float) -> float:
    """Smallest absolute angular difference."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


@dataclass
class Bar:
    uid: int
    center: float          # deg
    width: float           # full arc width, deg (shrinks over time)
    is_blue: bool
    born_at: float         # sim time
    # Seconds added to the clock when this bar is picked. The client stores
    # this per-bar and rolls it at spawn time; blue bars carry
    # m_flTimerIncreasePerUnlock, orange bars carry nothing.
    time_bonus: float = 0.0
    # False while a reaction delay hides this bar from the model's
    # observations (set by the controller; the sim itself never reads it,
    # so an already-scheduled click can still land on an unperceived bar)
    perceived: bool = True


class LockpickingSim:
    def __init__(self, stage: StageConfig = DEFAULT_STAGE,
                 tuning: SimTuning = DEFAULT_TUNING,
                 seed: int | None = None):
        self.stage = stage
        self.tuning = tuning
        self.rng = random.Random(seed)
        self.dt = 1.0 / tuning.tick_rate

        # Init @ 0x182ada891: half-width is the angle between (0, -boardRadius)
        # and (unlockRadius, -boardRadius) => atan(40/180) = 12.5288 deg.
        self.initial_bar_width = 2.0 * math.degrees(
            math.atan(stage.unlock_radius / stage.board_radius))

        # dynamic state (names mirror CDOTALockpickingGame fields)
        self.t = 0.0
        self.time_remaining = stage.time_limit     # +0x34
        self.pick_angle = 0.0                      # +0x30
        self.direction = 1                         # +0x2c, 1 => cw, -1 => ccw
        self.total_traveled = 0.0
        self.rmb_held = False                      # +0x40
        self.speed = stage.initial_speed           # +0x38, absolute deg/s
        self.boost_accum = 0.0                     # +0x3c, banked boost
        self.disabled = False                      # +0x41, latched by a miss
        self.bars: list[Bar] = []                  # +0x18
        self.unlock_count = 0                      # +0x08
        self.score = 0
        self._blue_pity = 0                        # +0x50, spawns since blue
        self.spawn_interval = stage.base_unlock_appear_rate   # +0x48
        self.spawn_timer = self.spawn_interval                # +0x44
        self._picked_since_spawn = False           # +0x4c
        self.game_over = False
        self.won = False
        self._next_uid = 1

        # Reset @ 0x182aec770 places one bar before the first frame, in a
        # restricted arc, and leaves the timer at a full interval.
        if tuning.spawn_immediately_at_start:
            self._spawn_bar(self.rng.uniform(stage.min_degrees_between_unlocks,
                                             tuning.opening_bar_angle_max))

    # ------------------------------------------------------------------ #
    # derived quantities

    @property
    def base_speed(self) -> float:
        return (self.stage.initial_speed
                + self.stage.speed_increment_per_unlock * self.unlock_count)

    @property
    def current_speed(self) -> float:
        """Effective angular speed right now, deg/s (always >= 0)."""
        return self.speed

    @property
    def max_speed(self) -> float:
        return self.base_speed * self.stage.max_speed_multiplier

    @property
    def spawn_frequency(self) -> float:
        """Bar spawns per second (1 / spawn_interval)."""
        return 1.0 / max(1e-6, self.spawn_interval)

    @property
    def blue_chance(self) -> float:
        """Current blue-bar roll chance, %, including escalating pity."""
        s = self.stage
        return (s.timer_increase_unlock_chance
                + s.timer_increase_unlock_escalating_chance * self._blue_pity)

    @property
    def pick_disabled(self) -> bool:
        return self.disabled

    # -- compatibility shims -------------------------------------------- #
    # The sim now tracks absolute speed the way the client does; these keep
    # the older multiplier/penalty view working for observations, the HUD
    # and calibration.py.

    @property
    def boost_mult(self) -> float:
        return self.speed / max(1e-6, self.base_speed)

    @boost_mult.setter
    def boost_mult(self, value: float) -> None:
        self.speed = self.base_speed * value
        self.boost_accum = max(0.0, self.speed - self.base_speed)

    @property
    def penalty_factor(self) -> float:
        if not self.disabled:
            return 1.0
        return min(1.0, self.speed / max(1e-6, self.base_speed))

    @penalty_factor.setter
    def penalty_factor(self, value: float) -> None:
        self.disabled = value < 1.0
        if not self.disabled:
            self.speed = max(self.speed, self.base_speed)

    @property
    def penalty_phase(self) -> int:
        return PEN_DECEL if self.disabled else PEN_NONE

    @penalty_phase.setter
    def penalty_phase(self, value: int) -> None:
        self.disabled = value == PEN_DECEL

    # ------------------------------------------------------------------ #
    # main loop

    def tick(self, dt: float | None = None) -> list[tuple]:
        """Advance the simulation by one fixed step (or a partial `dt`)."""
        if self.game_over:
            return []
        if dt is None:
            dt = self.dt
        if dt <= 0.0:
            return []
        events: list[tuple] = []
        s = self.stage
        tu = self.tuning

        # The block order below mirrors CDOTALockpickingGame::Update
        # (0x182aecdb0) statement for statement.

        # --- move the pick, using the speed from the START of the frame ---
        # (Update reads m_flSpeed into xmm7 before touching it, so this frame's
        # boost/recovery does not apply until the next one.)
        step = self.speed * dt
        self.pick_angle = ang_norm(self.pick_angle + self.direction * step)
        self.total_traveled += step

        # --- countdown ---
        self.time_remaining = max(0.0, self.time_remaining - dt)

        # --- speed: recover > boost > decay, in that priority ---
        base = self.base_speed
        if self.disabled:
            # a miss zeroed the speed; climb back at m_flRecoverRate deg/s^2
            # and re-enable the pick the moment we are whole again. RMB is
            # ignored for the whole window.
            self.speed = min(base, self.speed + s.recover_rate * dt)
            if self.speed >= base:
                self.disabled = False
        elif self.rmb_held:
            # NOTE: no `* dt` — m_flSpeedBoostRate is added per Update call,
            # so the ramp is framerate dependent in retail (see SimTuning).
            # We scale by dt/self.dt rather than using a flat rate so that a
            # tick the controller SPLITS for sub-tick click precision still
            # contributes exactly one frame's worth of boost between them
            # (a flat add would apply the boost twice for that tick).
            gain = (s.speed_boost_rate * (dt / self.dt) if tu.boost_is_per_tick
                    else s.speed_boost_rate * dt)
            delta = min(gain, self.max_speed - self.speed)
            self.speed += delta
            self.boost_accum += delta
        else:
            # banked boost bleeds off proportionally to how much was banked
            self.speed = max(base, self.speed
                             - s.deceleration_rate * self.boost_accum * dt)
            if base >= self.speed:
                self.boost_accum = 0.0

        # --- shrink bars / expire (both edges close at the decrease rate) ---
        shrink = 2.0 * s.unlock_degree_decrease_rate * dt
        survivors = []
        for bar in self.bars:
            bar.width -= shrink
            if bar.width <= tu.min_bar_width_deg:
                events.append((EV_BAR_EXPIRED, bar))
            else:
                survivors.append(bar)
        self.bars = survivors

        # --- spawn timer ---
        # The interval shrinks at most once per firing, and only if a pick
        # landed since the last one; the timer is reloaded, never carried.
        self.spawn_timer -= dt
        if self.spawn_timer < 0.0:
            if self._picked_since_spawn:
                self.spawn_interval = max(
                    tu.min_spawn_interval,
                    self.spawn_interval - s.unlock_appear_increase_rate)
                self._picked_since_spawn = False
            self.spawn_timer = self.spawn_interval
            # note: the interval already shrank even if the board is full
            if len(self.bars) < s.max_unlocks_on_board:
                bar = self._spawn_bar(self._pick_spawn_angle())
                events.append((EV_BAR_SPAWNED, bar))

        self.t += dt
        if self.time_remaining <= 0.0:
            self.game_over = True
            self.won = False
            events.append((EV_GAME_OVER, self.score))

        return events

    def perturbed_angle(self, displacement: float = 0.0) -> float:
        """Pick angle after advancing `displacement` degrees in the current
        travel direction (same unit as current_speed).

        `displacement=0.0` returns the live pick angle. Positive values look
        *forward* along the direction of motion (reducing the forward distance
        to bars by exactly `displacement`); negative values look behind. Used to
        positionally perturb the observations fed to a model without moving the
        real pick.
        """
        if displacement == 0.0:
            return self.pick_angle
        return ang_norm(self.pick_angle + self.direction * displacement)

    def hittable_bar(self, displacement: float = 0.0) -> Bar | None:
        """The bar a click right now would hit, or None.

        With a non-zero `displacement` (degrees, travel-direction), answers as
        if the pick were at that offset from its current position.
        """
        angle = self.perturbed_angle(displacement)
        # TryUnlock walks the array back to front, so when two bars overlap
        # the most recently spawned one is the one that gets picked.
        for bar in reversed(self.bars):
            if ang_diff(angle, bar.center) <= bar.width / 2.0 + self.tuning.hit_buffer_deg:
                return bar
        return None

    def click(self) -> list[tuple]:
        """LMB pick attempt at the current pick angle."""
        # TryUnlock bails out immediately while the disabled flag is latched,
        # so a click during the recovery window is swallowed entirely — it is
        # not even counted as a miss.
        if self.game_over or self.pick_disabled:
            return []
        events: list[tuple] = []
        s = self.stage
        hit = self.hittable_bar()
        if hit is not None:
            self.bars.remove(hit)
            self.unlock_count += 1
            self.score += s.score_per_unlock
            self._picked_since_spawn = True    # latches the interval shrink
            if self.tuning.reverse_on_success:
                self.direction = -self.direction
            events.append((EV_PICKED, hit))
            if hit.time_bonus:
                self.time_remaining += hit.time_bonus
                events.append((EV_TIMER_BONUS, hit.time_bonus))
            if self.unlock_count >= s.num_unlocks:
                self.game_over = True
                self.won = True
                events.append((EV_GAME_OVER, self.score))
        else:
            # miss: speed drops to exactly zero and the pick is latched off
            # until it recovers. Banked boost survives the miss.
            self.speed = 0.0
            self.disabled = True
            events.append((EV_MISSED, self.pick_angle))
        return events

    # ------------------------------------------------------------------ #
    # spawning

    def _spawn_bar(self, center: float, roll_blue: bool = True) -> Bar:
        """Append a bar at `center`, rolling the blue-bar chance.

        The opening bar placed by Reset() skips the roll entirely — it is
        always orange and does not advance the pity counter.
        """
        s = self.stage
        is_blue = False
        if roll_blue:
            # `chance >= roll`, so a 15% chance genuinely beats a 15.0 roll
            is_blue = self.blue_chance >= self.rng.uniform(0.0, 100.0)
            self._blue_pity = 0 if is_blue else self._blue_pity + 1
        bar = Bar(self._next_uid, ang_norm(center), self.initial_bar_width,
                  is_blue, self.t,
                  time_bonus=s.timer_increase_per_unlock if is_blue else 0.0)
        self._next_uid += 1
        self.bars.append(bar)
        return bar

    def _pick_spawn_angle(self) -> float:
        """PickSpawnAngle @ 0x182acd670.

        Rejection sampling, NOT an exact draw from the allowed gaps: sample a
        uniform angle, retry while it lands within
        m_flMinDegreesBetweenUnlocks of an existing bar, and give up after 20
        samples — using that last, still-conflicting angle. So a crowded board
        really can place bars closer than the nominal minimum.

        (The client also burns one extra RandomFloat before the loop and
        discards it; that only matters for matching Valve's RNG stream, which
        we cannot do anyway, so it is not reproduced here.)
        """
        r = self.stage.min_degrees_between_unlocks
        # Intervals are clamped to [0, 360] and the wrapped remainder is added
        # as a second interval, exactly as the client builds them.
        blocked: list[tuple[float, float]] = []
        for bar in self.bars:
            lo, hi = bar.center - r, bar.center + r
            blocked.append((max(0.0, lo), min(360.0, hi)))
            if lo < 0.0:
                blocked.append((lo + 360.0, 360.0))
            if hi > 360.0:
                blocked.append((0.0, hi - 360.0))
        angle = 0.0
        for _ in range(20):
            angle = ang_norm(self.rng.uniform(0.0, 360.0))
            if not any(lo <= angle <= hi for lo, hi in blocked):
                break
        return angle

    # ------------------------------------------------------------------ #
    # helpers for observers / renderers

    def ang_fwd(self, to: float, displacement: float = 0.0) -> float:
        """Degrees of travel (in the current direction) to reach angle `to`.

        With a non-zero `displacement` (degrees, travel-direction), measures
        from the perturbed pick position instead of the live one.
        """
        angle = self.perturbed_angle(displacement)
        return ((to - angle) * self.direction) % 360.0

    def bars_by_forward_distance(self, displacement: float = 0.0) -> list[tuple[float, Bar]]:
        """Bars sorted by travel distance from the pick, in travel direction."""
        out = [(self.ang_fwd(b.center, displacement), b) for b in self.bars]
        out.sort(key=lambda p: p[0])
        return out
