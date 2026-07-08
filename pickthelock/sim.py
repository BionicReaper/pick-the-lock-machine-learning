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

        # m_nUnlockRadius is a half-arc in *pixels* at m_nBoardRadius px:
        # full width = 2 * 40/180 rad ~= 25.46 deg (matches footage: 22-24
        # measured with conservative thresholds)
        self.initial_bar_width = math.degrees(
            2.0 * stage.unlock_radius / stage.board_radius)

        # dynamic state
        self.t = 0.0
        self.time_remaining = stage.time_limit
        self.pick_angle = 0.0
        self.direction = 1                     # +1 cw, -1 ccw
        self.total_traveled = 0.0
        self.rmb_held = False
        self.boost_mult = 1.0                  # 1 .. max_speed_multiplier
        self.penalty_phase = PEN_NONE
        self.penalty_factor = 1.0              # 0 .. 1, scales speed
        self._penalty_t = 0.0
        self.bars: list[Bar] = []
        self.unlock_count = 0
        self.score = 0
        self.blue_chance = stage.timer_increase_unlock_chance  # %, escalates
        self.spawn_timer = 0.0 if tuning.spawn_immediately_at_start else self.spawn_interval
        self.game_over = False
        self.won = False
        self._next_uid = 1

    # ------------------------------------------------------------------ #
    # derived quantities

    @property
    def base_speed(self) -> float:
        return (self.stage.initial_speed
                + self.stage.speed_increment_per_unlock * self.unlock_count)

    @property
    def current_speed(self) -> float:
        """Effective angular speed right now, deg/s (always >= 0)."""
        return self.base_speed * self.boost_mult * self.penalty_factor

    @property
    def max_speed(self) -> float:
        return self.base_speed * self.stage.max_speed_multiplier

    @property
    def spawn_interval(self) -> float:
        """Seconds between spawns: m_flBaseUnlockAppearRate minus
        m_flUnlockAppearIncreaseRate per successful unlock (1.4 - 0.04*n).
        Floored, since the raw formula reaches zero at 35 unlocks."""
        return max(self.tuning.min_spawn_interval,
                   self.stage.base_unlock_appear_rate
                   - self.stage.unlock_appear_increase_rate * self.unlock_count)

    @property
    def spawn_frequency(self) -> float:
        """Bar spawns per second (1 / spawn_interval)."""
        return 1.0 / self.spawn_interval

    @property
    def pick_disabled(self) -> bool:
        return self.penalty_phase == PEN_DECEL

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

        # --- boost multiplier: ramps at m_flSpeedBoostRate per second
        # (footage: base->max in ~0.4 s, and back just as fast) ---
        if self.rmb_held and not self.pick_disabled:
            self.boost_mult = min(s.max_speed_multiplier,
                                  self.boost_mult + s.speed_boost_rate * dt)
        else:
            self.boost_mult = max(1.0, self.boost_mult - s.speed_boost_rate * dt)

        # --- miss penalty state machine ---
        if self.penalty_phase == PEN_DECEL:
            self._penalty_t += dt
            # exponential slowdown, never a full stop (footage)
            self.penalty_factor *= math.exp(-s.deceleration_rate * dt)
            if self._penalty_t >= tu.miss_disable_duration:
                self.penalty_phase = PEN_RECOVER
        elif self.penalty_phase == PEN_RECOVER:
            # speed climbs back at m_flRecoverRate deg/s^2 (boost may re-ramp
            # on top of this, which matches the fast recovery in footage)
            self.penalty_factor += (s.recover_rate / max(1e-6, self.base_speed)) * dt
            if self.penalty_factor >= 1.0:
                self.penalty_factor = 1.0
                self.penalty_phase = PEN_NONE

        # --- move the pick ---
        step = self.current_speed * dt
        self.pick_angle = ang_norm(self.pick_angle + self.direction * step)
        self.total_traveled += step

        # --- shrink bars / expire (both edges close at the decrease rate) ---
        shrink = 2.0 * s.unlock_degree_decrease_rate * dt
        survivors = []
        for bar in self.bars:
            bar.width -= shrink
            if bar.width <= self.tuning.min_bar_width_deg:
                events.append((EV_BAR_EXPIRED, bar))
            else:
                survivors.append(bar)
        self.bars = survivors

        # --- spawn new bars on the interval timer ---
        self.spawn_timer -= dt
        if self.spawn_timer <= 0.0:
            bar = self._try_spawn()
            if bar is not None:
                events.append((EV_BAR_SPAWNED, bar))
            self.spawn_timer = self.spawn_interval
        elif self.tuning.keep_board_nonempty and not self.bars:
            # footage: a new bar pops ~0.1s after the board empties
            bar = self._try_spawn()
            if bar is not None:
                events.append((EV_BAR_SPAWNED, bar))
                self.spawn_timer = self.spawn_interval

        # --- timer ---
        self.time_remaining -= dt
        self.t += dt
        if self.time_remaining <= 0.0:
            self.time_remaining = 0.0
            self.game_over = True
            self.won = False
            events.append((EV_GAME_OVER, self.score))

        return events

    def hittable_bar(self) -> Bar | None:
        """The bar a click right now would hit, or None."""
        for bar in self.bars:
            if ang_diff(self.pick_angle, bar.center) <= bar.width / 2.0 + self.tuning.hit_buffer_deg:
                return bar
        return None

    def click(self) -> list[tuple]:
        """LMB pick attempt at the current pick angle."""
        if self.game_over or self.pick_disabled:
            return []
        events: list[tuple] = []
        s = self.stage
        hit = self.hittable_bar()
        if hit is not None:
            self.bars.remove(hit)
            self.unlock_count += 1
            self.score += s.score_per_unlock
            if self.tuning.reverse_on_success:
                self.direction = -self.direction
            events.append((EV_PICKED, hit))
            if hit.is_blue:
                self.time_remaining += s.timer_increase_per_unlock
                events.append((EV_TIMER_BONUS, s.timer_increase_per_unlock))
            if self.unlock_count >= s.num_unlocks:
                self.game_over = True
                self.won = True
                events.append((EV_GAME_OVER, self.score))
        else:
            # miss: pick disabled briefly; boost is lost
            self.penalty_phase = PEN_DECEL
            self._penalty_t = 0.0
            events.append((EV_MISSED, self.pick_angle))
        return events

    # ------------------------------------------------------------------ #
    # spawning

    def _try_spawn(self) -> Bar | None:
        s = self.stage
        if len(self.bars) >= s.max_unlocks_on_board:
            return None
        width = self.initial_bar_width
        # forbidden arcs around existing bars: centers may not be closer than
        # half both widths plus the minimum gap
        forbidden: list[tuple[float, float]] = []
        for bar in self.bars:
            r = width / 2.0 + bar.width / 2.0 + s.min_degrees_between_unlocks
            forbidden.append((ang_norm(bar.center - r), 2.0 * r))
        center = self._sample_allowed_angle(forbidden)
        if center is None:
            return None
        # blue roll with escalating pity
        roll = self.rng.uniform(0.0, 100.0)
        is_blue = roll < self.blue_chance
        if is_blue:
            self.blue_chance = s.timer_increase_unlock_chance
        else:
            self.blue_chance += s.timer_increase_unlock_escalating_chance
        bar = Bar(self._next_uid, center, width, is_blue, self.t)
        self._next_uid += 1
        self.bars.append(bar)
        return bar

    def _sample_allowed_angle(self, forbidden: list[tuple[float, float]]) -> float | None:
        """Pick a uniform random angle outside all forbidden arcs
        (each given as (start_deg, length_deg), clockwise)."""
        if not forbidden:
            return self.rng.uniform(0.0, 360.0)
        # merge forbidden arcs on the circle
        arcs = sorted((start, min(length, 360.0)) for start, length in forbidden)
        merged: list[list[float]] = []
        for start, length in arcs:
            end = start + length
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        # wrap-around: last arc may spill past 360 into the first
        if len(merged) > 1 and merged[-1][1] >= 360.0 + merged[0][0]:
            merged[0][0] = 0.0
            merged[0][1] = max(merged[0][1], merged[-1][1] - 360.0)
            merged.pop()
        elif merged and merged[-1][1] > 360.0:
            spill = merged[-1][1] - 360.0
            merged[-1][1] = 360.0
            if merged[0][0] < spill:
                merged[0][0] = 0.0
                merged[0][1] = max(merged[0][1], spill)
            else:
                merged.insert(0, [0.0, spill])
        # allowed gaps between merged arcs
        gaps: list[tuple[float, float]] = []
        total = 0.0
        for i, (start, end) in enumerate(merged):
            nxt_start = merged[(i + 1) % len(merged)][0] + (360.0 if i == len(merged) - 1 else 0.0)
            gap_len = nxt_start - end
            if gap_len > 1e-9:
                gaps.append((end, gap_len))
                total += gap_len
        if total <= 1e-9:
            return None
        x = self.rng.uniform(0.0, total)
        for start, length in gaps:
            if x <= length:
                return ang_norm(start + x)
            x -= length
        return ang_norm(gaps[-1][0] + gaps[-1][1])

    # ------------------------------------------------------------------ #
    # helpers for observers / renderers

    def ang_fwd(self, to: float) -> float:
        """Degrees of travel (in the current direction) to reach angle `to`."""
        return ((to - self.pick_angle) * self.direction) % 360.0

    def bars_by_forward_distance(self) -> list[tuple[float, Bar]]:
        """Bars sorted by travel distance from the pick, in travel direction."""
        out = [(self.ang_fwd(b.center), b) for b in self.bars]
        out.sort(key=lambda p: p[0])
        return out
