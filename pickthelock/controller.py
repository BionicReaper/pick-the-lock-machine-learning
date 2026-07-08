"""Cancellable "click after X degrees at Y boost" scheduler.

This is the interface the NEAT model (and the AI playback mode) drives:

    ctrl.schedule(distance_deg=137.0, boost_hold_frac=0.5, do_click=True)

means: travel 137 degrees from wherever the pick is right now, holding RMB
for the first 50% of that distance, then click (or just report arrival if
do_click is False).  A new schedule() or cancel() at any point discards the
pending one — e.g. when a new bar spawns and the model changes its mind.

The controller owns sim stepping via step(): it splits the tick that would
overshoot the target so the click lands on the exact scheduled distance
(sub-tick precision), then finishes the remainder of the tick.

Human-imperfection knobs (all default to "perfect play"):
  inaccuracy         gaussian aim error added to every scheduled distance,
                     proportional to the current pick speed
  reaction_time_ms   delay before reacting to a new bar spawn; arm it with
                     schedule_prompt() and poll prompt_due each tick
  reaction_time_std  relative gaussian jitter on that delay
"""

from __future__ import annotations

import random

from .sim import LockpickingSim, EV_TARGET_REACHED


class ScheduledClickController:
    def __init__(self, sim: LockpickingSim, inaccuracy: float = 0.0,
                 reaction_time_ms: float = 0.0, reaction_time_std: float = 0.05):
        self.sim = sim
        self.inaccuracy = inaccuracy
        self.reaction_time_ms = reaction_time_ms
        self.reaction_time_std = reaction_time_std
        self.active = False
        self.target_dist = 0.0
        self.boost_dist = 0.0
        self.do_click = True
        self._start_traveled = 0.0
        self.tick_counter = 0
        self.scheduled_prompts: set[int] = set()

    # ------------------------------------------------------------------ #

    def schedule(self, distance_deg: float, boost_hold_frac: float, do_click: bool) -> None:
        """Arm a click `distance_deg` of travel from the current position.

        boost_hold_frac in [0, 1]: fraction of the distance during which RMB
        is held (held first, then released — per spec 0.5 = boost for the
        first half of the way).
        """
        if self.inaccuracy > 0.0:
            distance_deg += self.sim.current_speed * self.inaccuracy * random.gauss(0.0, 1.0)
        distance_deg = max(self.sim.tuning.min_target_distance_deg, float(distance_deg))
        boost_hold_frac = min(1.0, max(0.0, float(boost_hold_frac)))
        self.target_dist = distance_deg
        self.boost_dist = distance_deg * boost_hold_frac
        self.do_click = bool(do_click)
        self._start_traveled = self.sim.total_traveled
        self.active = True

    def cancel(self) -> None:
        """Discard the pending schedule (model changed its mind)."""
        self.active = False
        self.sim.rmb_held = False

    def schedule_prompt(self) -> None:
        """Request a prompt after a human-like reaction delay.

        Call on unforeseeable stimuli (a new bar spawn); the tick at which
        the reaction lands is added to scheduled_prompts and prompt_due
        turns True on that tick. Zero-valued knobs skip the random draw."""
        if self.reaction_time_ms <= 0.0:
            self.scheduled_prompts.add(self.tick_counter)
            return
        base = (self.reaction_time_ms / 1000.0) * self.sim.tuning.tick_rate
        if self.reaction_time_std == 0.0:
            self.scheduled_prompts.add(self.tick_counter + round(base))
            return
        delay = base * (1.0 + random.gauss(0.0, 1.0) * self.reaction_time_std)
        self.scheduled_prompts.add(self.tick_counter + max(0, round(delay)))

    @property
    def prompt_due(self) -> bool:
        """True while a reaction scheduled via schedule_prompt() lands on this tick."""
        return self.tick_counter in self.scheduled_prompts

    @property
    def progress(self) -> float:
        """Degrees traveled since the schedule was armed."""
        return self.sim.total_traveled - self._start_traveled

    @property
    def remaining(self) -> float:
        return self.target_dist - self.progress

    @property
    def target_angle(self) -> float:
        """Board angle where the scheduled click will land (for display).

        Note: if the direction flips mid-travel (a manual click, not possible
        via this controller), the landing angle changes; distance does not.
        """
        return (self.sim.pick_angle + self.sim.direction * self.remaining) % 360.0

    # ------------------------------------------------------------------ #

    def step(self) -> list[tuple]:
        """Advance the sim by one fixed tick, honoring the schedule.

        Returns all sim events raised during the tick, plus
        (EV_TARGET_REACHED, clicked_events) when the scheduled distance is hit.
        """
        # previous tick is over: its reaction entry (if any) has been consumed
        if self.scheduled_prompts:
            self.scheduled_prompts.discard(self.tick_counter)
        self.tick_counter += 1
        sim = self.sim
        if not self.active:
            return sim.tick()

        events: list[tuple] = []
        # hold RMB only while inside the boost portion of the travel
        sim.rmb_held = self.progress < self.boost_dist

        speed = sim.current_speed
        est_step = speed * sim.dt
        rem = self.remaining

        if speed > 0.0 and 0.0 < rem <= est_step:
            # split the tick: land exactly on the target distance
            dt1 = rem / speed
            dt1 = min(dt1, sim.dt)
            events += sim.tick(dt1)
            events += self._arrive()
            events += sim.tick(sim.dt - dt1)
        else:
            events += sim.tick()
            # guard: speed ramps within the tick can overshoot the estimate
            if self.active and self.remaining <= 0.0:
                events += self._arrive()
        return events

    def _arrive(self) -> list[tuple]:
        events: list[tuple] = []
        clicked: list[tuple] = []
        if self.do_click:
            clicked = self.sim.click()
            events += clicked
        self.active = False
        self.sim.rmb_held = False
        events.append((EV_TARGET_REACHED, clicked))
        return events
