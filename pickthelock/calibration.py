"""Calibration mode — measure the player's reaction time and aim inaccuracy.

Two controlled tests estimate the human-imperfection knobs that
train_neat.py / play.py accept, then print a ready-to-edit train command:

  Test 1 (reaction, 20 samples): after a random pause a full-width bar
  appears glued to the pick (it follows it, so aim plays no part). Click the
  instant it appears; the sample is the time from spawn to click. The bar
  declines away in REACT_BAR_LIFETIME_S — reacting slower than that discards
  the trial, and clicking before the bar appears (a false start) discards it
  too and restarts the random wait, so anticipatory spam never produces a
  sample.

  Test 2 (inaccuracy, 100 clicks): one small non-shrinking bar at a time at a
  random position; every click is a sample of (signed angular error from the
  bar center) / (pick speed). Speed control is locked: the first half of the
  attempts run at base speed, the second half at full boost.

Both estimates are gaussian MLE fits matching the controller's noise model
(see controller.py): aim error is zero-mean and speed-proportional, so
--inaccuracy = RMS of the error/speed samples; the reaction delay is
mean * (1 + N(0,1) * relative_std), so --reaction_time_ms = sample mean and
--reaction_time_standard_deviation = sample std / mean.

Everything here is additive: the sim is a stock LockpickingSim driven with a
calibration-local StageConfig/SimTuning (no natural spawns, no time limit,
no width shrink — the reaction bar's decline is applied by hand), and bars
are placed by hand. No other module is modified.
"""

from __future__ import annotations

import itertools
import math
import os
import random

import pygame

from .config import StageConfig, SimTuning
from .game import (GameApp, BOARD_CENTER, DESIGN_W, DESIGN_H, DIAL_SIZE,
                   RING_INNER, RING_OUTER, COL_BEAM,
                   COL_GOLD_LIGHT, COL_GOLD_DARK, COL_GREY, COL_RED,
                   COL_MARKER_BLUE, MENU, RUNNING, PAUSED,
                   angle_to_xy, FloatText)
from .sim import (LockpickingSim, Bar, PEN_NONE,
                  EV_PICKED, EV_MISSED, EV_BAR_EXPIRED)

COL_GREEN = (120, 255, 120)

# --------------------------------------------------------------------- #
# reaction test
REACTION_SAMPLES = 20
REACT_BAR_LIFETIME_S = 0.8   # the bar declines from full width to the expiry
                             # floor in this long; slower reactions discard
INTER_TRIAL_S = (0.75, 2.0)  # random empty-board wait before each bar

# inaccuracy test
INACC_SAMPLES = 100
INACC_BAR_WIDTH_DEG = 6.0
INACC_SPAWN_S = (1.2, 2.5)          # spawn distance = uniform(this) * speed
INACC_MIN_DEG, INACC_MAX_DEG = 30.0, 330.0

MSG_SECONDS = 1.8           # how long trial verdict messages linger

# printed train command defaults (mirror train_neat.py's own defaults)
TRAIN_GENERATIONS = 1000
TRAIN_SCHEMA = 0


def _calibration_stage() -> StageConfig:
    """Retail physics, but endless and inert: no clock, no natural spawns,
    and no width shrink (the reaction bar's decline is applied by hand)."""
    return StageConfig(time_limit=1e9,
                       base_unlock_appear_rate=1e9,
                       num_unlocks=10 ** 9,
                       unlock_degree_decrease_rate=0.0)


def _calibration_tuning() -> SimTuning:
    return SimTuning(spawn_immediately_at_start=False,
                     keep_board_nonempty=False)


# --------------------------------------------------------------------- #


class CalibrationApp(GameApp):
    """Pygame app for the two calibration tests.

    Reuses GameApp's window/assets/fonts/run loop and overrides the event,
    tick and draw hooks; the underlying sim is only ever driven from here.
    """

    def __init__(self, muted: bool = False, scale: float | None = None,
                 seed: int | None = None,
                 reaction_target: int = REACTION_SAMPLES,
                 inacc_target: int = INACC_SAMPLES):
        super().__init__(ai_genome_path=None, seed=None, muted=muted, scale=scale)
        pygame.display.set_caption("Pick the Lock — Calibration")
        self.rng = random.Random(seed)
        self._bar_uids = itertools.count(1)
        self.page = "intro"            # intro | interlude | results (state MENU)
        self.phase = None              # "react" | "inacc" while state RUNNING
        self.reaction_target = reaction_target
        self.inacc_target = inacc_target
        # reaction test state
        self.react_bar: Bar | None = None
        self.react_spawn_t = 0.0
        self._react_decline = 0.0      # deg/s of width, walks the bar to expiry
        self.next_trial_at = 0.0
        self.samples: list[float] = []     # reaction times, sim seconds
        self.fails = 0
        # inaccuracy test state
        self.inacc_bar: Bar | None = None
        self.inacc_samples: list[float] = []
        # results
        self.react_mu_ms = 0.0
        self.react_sigma_ms = 0.0
        self.react_rel_std = 0.0
        self.inaccuracy = 0.0
        self.train_cmd = ""
        self._msg: tuple[str, tuple] | None = None
        self._msg_until = 0.0

    # ------------------------------------------------------------------ #
    # phase transitions

    def _start_reaction(self):
        self.phase = "react"
        self.sim = LockpickingSim(_calibration_stage(), _calibration_tuning(),
                                  seed=self.rng.randrange(1 << 30))
        self.react_bar = None
        self.samples.clear()
        self.fails = 0
        self.next_trial_at = self.sim.t + self.rng.uniform(*INTER_TRIAL_S)
        self.state = RUNNING

    def _finish_reaction(self):
        n = len(self.samples)
        mu = sum(self.samples) / n
        sigma = math.sqrt(sum((x - mu) ** 2 for x in self.samples) / n)  # MLE (/n)
        self.react_mu_ms = mu * 1000.0
        self.react_sigma_ms = sigma * 1000.0
        self.react_rel_std = sigma / mu if mu > 0.0 else 0.0
        self.state = MENU
        self.page = "interlude"

    def _start_inaccuracy(self):
        self.phase = "inacc"
        self.sim = LockpickingSim(_calibration_stage(), _calibration_tuning(),
                                  seed=self.rng.randrange(1 << 30))
        self.inacc_samples.clear()
        self._spawn_inacc_bar()
        self.state = RUNNING

    def _finish(self):
        self.inaccuracy = math.sqrt(
            sum(x * x for x in self.inacc_samples) / len(self.inacc_samples))
        workers = max(1, (os.cpu_count() or 2) - 1)
        self.train_cmd = (
            f"python train_neat.py"
            f" --reaction_time_ms {self.react_mu_ms:.1f}"
            f" --reaction_time_standard_deviation {self.react_rel_std:.3f}"
            f" --inaccuracy {self.inaccuracy:.4f}"
            f" --workers {workers}"
            f" --generations {TRAIN_GENERATIONS}"
            f" --schema {TRAIN_SCHEMA}")
        print("\n=== Calibration complete ===")
        print(f"Reaction:   mean {self.react_mu_ms:.1f} ms, "
              f"std {self.react_sigma_ms:.1f} ms "
              f"(relative {self.react_rel_std:.3f}), "
              f"{len(self.samples)} samples")
        print(f"Inaccuracy: {self.inaccuracy:.4f} "
              f"({len(self.inacc_samples)} clicks)")
        print("\nTrain a net that plays like you:\n")
        print(f"  {self.train_cmd}\n")
        self.state = MENU
        self.page = "results"

    # ------------------------------------------------------------------ #
    # reaction test

    def _spawn_react_bar(self):
        sim = self.sim
        w = sim.initial_bar_width
        bar = Bar(next(self._bar_uids), sim.pick_angle, w,
                  self.rng.random() < 0.5, sim.t)   # color is cosmetic here
        sim.bars.append(bar)
        self.react_bar = bar
        self.react_spawn_t = sim.t
        self._react_decline = (w - sim.tuning.min_bar_width_deg) / REACT_BAR_LIFETIME_S
        # a false start just before the spawn must not gate this click
        self._reset_penalty()

    def _react_tick(self):
        sim = self.sim
        if self.react_bar is not None:
            # apply the decline before the tick so the sim's own expiry
            # pass emits the usual EV_BAR_EXPIRED at the width floor
            self.react_bar.width -= self._react_decline * sim.dt
        events = sim.tick()
        if self.react_bar is not None:
            self.react_bar.center = sim.pick_angle   # glued to the pick
        for ev in events:
            if ev[0] == EV_BAR_EXPIRED and ev[1] is self.react_bar:
                self.react_bar = None
                self.fails += 1
                self._flash("TOO SLOW — trial discarded", COL_RED)
                self.next_trial_at = sim.t + self.rng.uniform(*INTER_TRIAL_S)
        if self.react_bar is None and self.state == RUNNING:
            if len(self.samples) >= self.reaction_target:
                self._finish_reaction()
            elif sim.t >= self.next_trial_at:
                self._spawn_react_bar()

    def _react_click(self):
        sim = self.sim
        bar = self.react_bar
        if bar is None:
            # false start: clicked before the bar appeared — restart the wait
            self._fx_miss()
            self._reset_penalty()
            self.fails += 1
            self._flash("TOO EARLY — wait for the bar", COL_RED)
            self.next_trial_at = sim.t + self.rng.uniform(*INTER_TRIAL_S)
            return
        events = sim.click()
        picked = next((ev[1] for ev in events if ev[0] == EV_PICKED), None)
        if picked is None:
            return   # unreachable: the bar is glued to the pick
        self._fx_pick(picked)
        self.react_bar = None
        sample = sim.t - self.react_spawn_t
        self.samples.append(sample)
        self._flash(f"{sample * 1000.0:.0f} ms", COL_GREEN)
        self.next_trial_at = sim.t + self.rng.uniform(*INTER_TRIAL_S)

    def _reset_penalty(self):
        self.sim.penalty_phase = PEN_NONE
        self.sim.penalty_factor = 1.0

    # ------------------------------------------------------------------ #
    # inaccuracy test

    def _locked_boost(self) -> bool:
        """Full boost for the second half of the attempts, none before."""
        return len(self.inacc_samples) >= self.inacc_target // 2

    def _inacc_tick(self):
        sim = self.sim
        # lock the speed: pin boost to the block's level and void penalties
        if self._locked_boost():
            sim.rmb_held = True
            sim.boost_mult = sim.stage.max_speed_multiplier
        else:
            sim.rmb_held = False
            sim.boost_mult = 1.0
        self._reset_penalty()
        sim.tick()

    def _signed_forward_error(self, bar: Bar) -> float:
        """Angular click error, deg; positive = bar center still ahead."""
        err = ((bar.center - self.sim.pick_angle) * self.sim.direction) % 360.0
        return err - 360.0 if err >= 180.0 else err

    def _inacc_click(self):
        sim = self.sim
        bar = self.inacc_bar
        if bar is None:
            return
        self._reset_penalty()   # a same-frame earlier miss must not swallow this
        self.inacc_samples.append(self._signed_forward_error(bar) / sim.current_speed)
        events = sim.click()
        if any(ev[0] == EV_PICKED for ev in events):
            self._fx_pick(bar)
        else:
            sim.bars.remove(bar)    # miss: still one sample, still respawn
            self._reset_penalty()
            self._fx_miss()
        self.inacc_bar = None
        if len(self.inacc_samples) >= self.inacc_target:
            self._finish()
        else:
            if len(self.inacc_samples) == self.inacc_target // 2:
                self._flash("SPEED UNLOCKED TO FULL BOOST", COL_GOLD_LIGHT)
            self._spawn_inacc_bar()

    def _spawn_inacc_bar(self):
        sim = self.sim
        dist = self.rng.uniform(*INACC_SPAWN_S) * sim.current_speed
        dist = min(max(dist, INACC_MIN_DEG), INACC_MAX_DEG)
        bar = Bar(next(self._bar_uids),
                  (sim.pick_angle + sim.direction * dist) % 360.0,
                  INACC_BAR_WIDTH_DEG, False, sim.t)
        sim.bars.append(bar)
        self.inacc_bar = bar

    # ------------------------------------------------------------------ #
    # GameApp hooks

    def _sim_tick(self):
        if self.phase == "react":
            self._react_tick()
        elif self.phase == "inacc":
            self._inacc_tick()

    def _handle_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_F3:
                    self.debug = not self.debug
                elif self.state == MENU:
                    if self.page == "results":
                        return False              # any key exits
                    if ev.key == pygame.K_SPACE:
                        if self.page == "intro":
                            self._start_reaction()
                        elif self.page == "interlude":
                            self._start_inaccuracy()
                    elif ev.key == pygame.K_ESCAPE:
                        return False
                elif ev.key in (pygame.K_F9, pygame.K_ESCAPE):
                    if self.state == RUNNING:
                        self.state = PAUSED
                    elif self.state == PAUSED:
                        self.state = RUNNING
                elif ev.key == pygame.K_UP and self.state == RUNNING:
                    if self.phase == "react":
                        self._react_click()
                    else:
                        self._inacc_click()
            if (ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1
                    and self.state == RUNNING):
                if self.phase == "react":
                    self._react_click()
                else:
                    self._inacc_click()
        return True

    def _flash(self, text: str, color):
        self._msg = (text, color)
        self._msg_until = self.wall_time + MSG_SECONDS
        self.float_texts.append(FloatText(
            "OK" if color == COL_GREEN else "X",
            (BOARD_CENTER[0], BOARD_CENTER[1] - 84),
            color, life=0.6))

    # ------------------------------------------------------------------ #
    # drawing

    def _draw_board(self, c, center):
        sim = self.sim
        # marker wedges + beam, identical look to GameApp (which draws a
        # timer/score dial we replace with calibration progress)
        glow = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        for bar in sim.bars:
            if bar.is_blue:
                inner, outer = (60, 120, 160), COL_MARKER_BLUE
                pulse = 0.925 + 0.075 * math.sin(self.wall_time * math.tau / 0.6)
                inner = tuple(int(ch * pulse) for ch in inner)
                outer = tuple(int(ch * pulse) for ch in outer)
            else:
                inner, outer = (170, 105, 15), (255, 205, 70)
            layer = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
            self._ring_wedge(layer, inner, outer, center,
                             bar.center - bar.width / 2, bar.center + bar.width / 2)
            glow.blit(layer, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        c.blit(glow, (0, 0))

        beam_col = COL_RED if sim.pick_disabled else COL_BEAM
        p1 = angle_to_xy(center, sim.pick_angle, RING_INNER - 6)
        p2 = angle_to_xy(center, sim.pick_angle, RING_OUTER)
        pygame.draw.line(c, (*beam_col, 110), p1, p2, 6)
        pygame.draw.line(c, (*beam_col, 200), p1, p2, 3)
        core = COL_RED if sim.pick_disabled else (255, 255, 255)
        pygame.draw.line(c, core, p1, p2, 1)

        if self.dial:
            c.blit(self.dial, (center[0] - DIAL_SIZE // 2, center[1] - DIAL_SIZE // 2))
        else:
            pygame.draw.circle(c, (20, 18, 16), center, DIAL_SIZE // 2)

        # dial center: phase + progress instead of timer/score
        if self.phase == "react":
            head, prog = "REACTION TEST", f"{len(self.samples)}/{self.reaction_target}"
            sub = f"discarded {self.fails}"
        else:
            head = "ACCURACY TEST"
            prog = f"{len(self.inacc_samples)}/{self.inacc_target}"
            sub = "FULL BOOST" if self._locked_boost() else "NO BOOST"
        s = self.f_stat_title.render(head, True, COL_GOLD_DARK)
        c.blit(s, s.get_rect(center=(center[0], center[1] - 52)))
        s = self.f_big.render(prog, True, COL_GOLD_LIGHT)
        c.blit(s, s.get_rect(center=(center[0], center[1] - 8)))
        s = self.f_small.render(sub, True, COL_GREY)
        c.blit(s, s.get_rect(center=(center[0], center[1] + 40)))

        if self._msg and self.wall_time < self._msg_until:
            text, col = self._msg
            s = self.f_med.render(text, True, col)
            c.blit(s, s.get_rect(center=(DESIGN_W // 2, 340)))

    def _draw_hints(self, c):
        if self.phase == "react":
            left = self.f_small.render("[LMB / UP] PICK — click the instant the bar appears",
                                       True, COL_GREY)
        else:
            left = self.f_small.render("[LMB / UP] PICK — speed is locked", True, COL_GREY)
        c.blit(left, (30, DESIGN_H - 120))
        pause = self.f_small.render("[F9] PAUSE", True, (110, 110, 110))
        c.blit(pause, (DESIGN_W - 30 - pause.get_width(), 24))

    def _page_lines(self) -> list[tuple]:
        if self.page == "intro":
            return [
                ("CALIBRATION", COL_GOLD_LIGHT, self.f_title),
                ("Two short tests estimate how you play, then print", COL_GREY, self.f_small),
                ("the train_neat.py command for a net that plays like you.", COL_GREY, self.f_small),
                ("", None, None),
                (f"TEST 1 — REACTION ({self.reaction_target} samples)", COL_GOLD_DARK, self.f_stat_title),
                ("After a random pause a bar appears on your pick.", COL_GREY, self.f_small),
                ("Click the instant you see it.", COL_GOLD_LIGHT, self.f_small),
                ("Clicking early, or slower than the bar lasts,", COL_GREY, self.f_small),
                ("discards that trial.", COL_GREY, self.f_small),
                ("", None, None),
                (f"TEST 2 — ACCURACY ({self.inacc_target} clicks)", COL_GOLD_DARK, self.f_stat_title),
                ("Small bars appear one at a time: click each as precisely", COL_GREY, self.f_small),
                ("as you can. Misses are fine — they are data too.", COL_GREY, self.f_small),
                ("Speed is locked: first half slow, second half full boost.", COL_GREY, self.f_small),
                ("", None, None),
                ("PRESS SPACE TO START", COL_GOLD_LIGHT, self.f_med),
            ]
        if self.page == "interlude":
            return [
                ("TEST 1 COMPLETE", COL_GOLD_LIGHT, self.f_title),
                (f"Reaction: mean {self.react_mu_ms:.1f} ms, "
                 f"std {self.react_sigma_ms:.1f} ms", COL_GREY, self.f_med),
                (f"(relative std {self.react_rel_std:.3f}, "
                 f"{self.fails} trials discarded)", COL_GREY, self.f_small),
                ("", None, None),
                (f"TEST 2 — ACCURACY ({self.inacc_target} clicks)", COL_GOLD_DARK, self.f_stat_title),
                ("One small bar at a time; click it dead center.", COL_GREY, self.f_small),
                ("Speed control is locked — no boosting decisions,", COL_GREY, self.f_small),
                (f"just aim. {self.inacc_target // 2} clicks at base speed,", COL_GREY, self.f_small),
                (f"then {self.inacc_target - self.inacc_target // 2} at full boost.", COL_GREY, self.f_small),
                ("", None, None),
                ("PRESS SPACE TO START", COL_GOLD_LIGHT, self.f_med),
            ]
        # results
        lines = [
            ("CALIBRATION COMPLETE", COL_GOLD_LIGHT, self.f_title),
            (f"Reaction: mean {self.react_mu_ms:.1f} ms, "
             f"std {self.react_sigma_ms:.1f} ms "
             f"(relative {self.react_rel_std:.3f})", COL_GREY, self.f_small),
            (f"Inaccuracy: {self.inaccuracy:.4f}", COL_GREY, self.f_small),
            ("", None, None),
            ("Train a net that plays like you:", COL_GOLD_DARK, self.f_stat_title),
        ]
        # wrap the command at flag boundaries so it fits the panel
        line = ""
        for part in self.train_cmd.split(" --"):
            part = part if not line else "--" + part
            if line and len(line) + len(part) > 56:
                lines.append((line, COL_GREEN, self.f_mono))
                line = "  " + part
            else:
                line = part if not line else line + " " + part
        lines.append((line, COL_GREEN, self.f_mono))
        lines += [
            ("", None, None),
            ("(also printed to the console)", COL_GREY, self.f_small),
            ("PRESS ANY KEY TO EXIT", COL_GOLD_LIGHT, self.f_med),
        ]
        return lines

    def _draw_menu(self, c):
        self._dim(c)
        lines = self._page_lines()
        panel = pygame.Rect(60, 180, 630, 60 + 30 * len(lines))
        self._panel(c, panel)
        y = panel.top + 40
        for text, col, font in lines:
            if text:
                s = font.render(text, True, col)
                c.blit(s, s.get_rect(center=(DESIGN_W // 2, y)))
            y += 30

    def _draw_postgame(self, c):    # unreachable in calibration; keep menu look
        self._draw_menu(c)


# --------------------------------------------------------------------- #


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Calibrate your reaction time and aim inaccuracy, then "
                    "print the matching train_neat.py command")
    parser.add_argument("--mute", action="store_true", help="disable audio")
    parser.add_argument("--scale", type=float, default=None, help="window scale factor")
    parser.add_argument("--seed", type=int, default=None,
                        help="fixed RNG seed for a reproducible trial sequence")
    parser.add_argument("--reaction-samples", type=int, default=REACTION_SAMPLES,
                        help=f"reaction clicks to collect (default {REACTION_SAMPLES})")
    parser.add_argument("--inaccuracy-samples", type=int, default=INACC_SAMPLES,
                        help=f"click attempts to collect, split half no-boost / "
                             f"half full-boost (default {INACC_SAMPLES})")
    args = parser.parse_args(argv)
    if args.reaction_samples < 1:
        parser.error("--reaction-samples must be at least 1")
    if args.inaccuracy_samples < 2:
        parser.error("--inaccuracy-samples must be at least 2")
    CalibrationApp(muted=args.mute, scale=args.scale, seed=args.seed,
                   reaction_target=args.reaction_samples,
                   inacc_target=args.inaccuracy_samples).run()


if __name__ == "__main__":
    main()
