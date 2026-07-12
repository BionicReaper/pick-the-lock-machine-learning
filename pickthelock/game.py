"""Playable Pick the Lock — pygame client on top of the shared sim.

Layout mirrors popup_dark_carnival_encounter_lockpicking.xml/.css:
750x1000 design surface, dungeon background, Slark behind the bars,
600px board bottom-aligned with a 340px dial (the dial art's native
size; it seats on the matching dark disc baked into lock_background).

Modes:
  human play (default)      LMB/Up pick, RMB/Space hold boost, F9/Esc pause, F3 debug
  AI playback (--ai path)   a trained NEAT genome drives the scheduler
"""

from __future__ import annotations

import json
import math
import os
import random
import sys

import pygame

from . import paths
from .assets import Assets, ROOT
from .config import DEFAULT_STAGE, DEFAULT_TUNING
from .controller import ScheduledClickController
from .schemas import SCHEMAS, get_schema, apply_config_io
from .sim import (LockpickingSim, EV_BAR_SPAWNED, EV_PICKED, EV_MISSED,
                  EV_TIMER_BONUS, EV_GAME_OVER, EV_TARGET_REACHED)

DESIGN_W, DESIGN_H = 750, 1000
BOARD_CENTER = (369, 651)     # from CSS margins (board 600px, bottom 95px)
# Markers are radial wedges spanning from the dial edge (RING_INNER) to
# where the bars stop (RING_OUTER); the dial art is sized to seat flush
# against RING_INNER, and the background disc (BG_SIZE) extends past
# RING_OUTER as the outer rim border.
RING_OUTER = 180.0
RING_INNER = 108.0
DIAL_SIZE = 216
BG_SIZE = 470

COL_MARKER = (255, 244, 91)       # #fff45b
COL_MARKER_BLUE = (203, 241, 255)  # #cbf1ff
COL_BEAM = (150, 130, 255)         # lockpick fine-line particle tint
COL_GOLD_LIGHT = (245, 214, 149)
COL_GOLD_DARK = (199, 155, 91)
COL_BRONZE = (138, 90, 43)
COL_GREY = (184, 184, 184)
COL_RED = (255, 60, 40)

HISCORE_PATH = os.path.join(ROOT, "highscore.json")

MENU, RUNNING, PAUSED, POSTGAME = range(4)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def angle_to_xy(center, angle_deg, radius):
    """0 deg = up, clockwise."""
    rad = math.radians(angle_deg)
    return (center[0] + radius * math.sin(rad), center[1] - radius * math.cos(rad))


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "life", "max_life", "color", "size")

    def __init__(self, x, y, vx, vy, life, color, size):
        self.x, self.y, self.vx, self.vy = x, y, vx, vy
        self.life = self.max_life = life
        self.color = color
        self.size = size


class FloatText:
    def __init__(self, text, pos, color, life=0.5):
        self.text, self.pos, self.color = text, list(pos), color
        self.life = self.max_life = life


class GameApp:
    def __init__(self, ai_genome_path: str | None = None, seed: int | None = None,
                 muted: bool = False, scale: float | None = None,
                 inaccuracy: float = 0.0, reaction_time_ms: float = 0.0,
                 reaction_time_std: float = 0.05, schema: int = 0):
        pygame.init()
        try:
            pygame.mixer.init()
        except pygame.error:
            muted = True

        info = pygame.display.Info()
        if scale is None:
            scale = clamp((info.current_h - 90) / DESIGN_H, 0.4, 1.0)
        self.scale = scale
        self.window = pygame.display.set_mode((int(DESIGN_W * scale), int(DESIGN_H * scale)))
        pygame.display.set_caption("Pick the Lock")
        self.canvas = pygame.Surface((DESIGN_W, DESIGN_H)).convert_alpha()

        self.assets = Assets(muted=muted)
        self.clock = pygame.time.Clock()
        self.seed = seed
        self.inaccuracy = inaccuracy
        self.reaction_time_ms = reaction_time_ms
        self.reaction_time_std = reaction_time_std
        self.schema = schema
        self.state = MENU
        self.debug = False
        self.wall_time = 0.0
        self.shake_until = 0.0
        self.particles: list[Particle] = []
        self.float_texts: list[FloatText] = []
        self.high_score = self._load_high_score()
        self.last_final_score = 0
        # boost can be held from two sources at once (RMB and Space);
        # the sim boosts while either is down
        self._boost_mouse = False
        self._boost_key = False

        self.sim: LockpickingSim | None = None
        self.ctrl: ScheduledClickController | None = None

        # AI playback
        self.net = None
        self.ai_outputs = None
        if ai_genome_path:
            self.net = self._load_net(ai_genome_path)

        self._make_fonts()
        self._prepare_static()

    # ------------------------------------------------------------------ #
    # setup helpers

    def _make_fonts(self):
        def font(names, size, bold=False):
            return pygame.font.SysFont(names, size, bold=bold)
        self.f_title = font("georgia,times new roman", 30, bold=True)
        self.f_sign = font("georgia,times new roman", 22, bold=True)
        self.f_big = font("consolas,courier new", 52, bold=True)
        self.f_score = font("consolas,courier new", 28, bold=True)
        self.f_stat_title = font("arial", 13, bold=True)
        self.f_med = font("arial", 20)
        self.f_small = font("arial", 15)
        self.f_mono = font("consolas,courier new", 14)
        self.f_bonus = font("consolas,courier new", 30, bold=True)

    def _prepare_static(self):
        a = self.assets.images
        self.bg = None
        if a["background"]:
            self.bg = pygame.transform.smoothscale(a["background"], (DESIGN_W, DESIGN_H))
        self.lock_bg = None
        if a["lock_background"]:
            self.lock_bg = pygame.transform.smoothscale(a["lock_background"], (BG_SIZE, BG_SIZE))
        self.dial = None
        if a["lock_dial"]:
            self.dial = pygame.transform.smoothscale(a["lock_dial"], (DIAL_SIZE, DIAL_SIZE))
        self.shank = None
        if a["lock_shank"]:
            self.shank = pygame.transform.smoothscale(a["lock_shank"], (306, 204))
        self.sign = None
        if a["sign"]:
            self.sign = pygame.transform.smoothscale(a["sign"], (270, 135))
        self.arm = None
        if a["slark_arm"]:
            self.arm = pygame.transform.smoothscale(a["slark_arm"], (400, 400))
        # head frames scaled to CSS box (~244 x 290)
        self.head_scaled: list[pygame.Surface] = []
        for fr in self.assets.head_frames:
            self.head_scaled.append(pygame.transform.smoothscale(fr, (244, 290)))

    def _load_net(self, genome_path: str):
        import pickle
        import neat
        cfg_path = os.path.join(ROOT, "neat_config.txt")
        neat_cfg = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                               neat.DefaultSpeciesSet, neat.DefaultStagnation, cfg_path)
        apply_config_io(neat_cfg, get_schema(self.schema))  # match the genome's I/O sizes
        with open(genome_path, "rb") as fh:
            genome = pickle.load(fh)
        return neat.nn.FeedForwardNetwork.create(genome, neat_cfg)

    def _load_high_score(self) -> int:
        try:
            with open(HISCORE_PATH, "r", encoding="utf-8") as fh:
                return int(json.load(fh).get("high_score", 0))
        except (OSError, ValueError, json.JSONDecodeError):
            return 0

    def _save_high_score(self):
        try:
            with open(HISCORE_PATH, "w", encoding="utf-8") as fh:
                json.dump({"high_score": self.high_score}, fh)
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    # game flow

    def start_game(self):
        seed = self.seed if self.seed is not None else random.randrange(1 << 30)
        self.sim = LockpickingSim(DEFAULT_STAGE, DEFAULT_TUNING, seed=seed)
        self.ctrl = ScheduledClickController(
            self.sim, inaccuracy=self.inaccuracy,
            reaction_time_ms=self.reaction_time_ms,
            reaction_time_std=self.reaction_time_std)
        self.particles.clear()
        self.float_texts.clear()
        self.ai_outputs = None
        self._boost_mouse = self._boost_key = False
        self.state = RUNNING
        self.assets.start_music()
        self.assets.play_vo("start")
        if self.net:
            self._ai_prompt()

    def end_game(self):
        self.state = POSTGAME
        self.last_final_score = self.sim.score if self.sim else 0
        self.assets.stop_music()
        self.assets.play("lose", 0.7)
        self.assets.play_vo("gameover")
        if self.last_final_score > self.high_score:
            self.high_score = self.last_final_score
            self._save_high_score()

    # ------------------------------------------------------------------ #
    # AI

    def _ai_prompt(self):
        # standardized two-step prompt: activate the net on the schema's inputs,
        # then interpret the outputs onto the controller (a new decision replaces
        # the pending schedule). ai_outputs holds the decoded action for display.
        sch = get_schema(self.schema)
        if sch.use_input_displacement:
            # perturb the pick's position for the encoded observation, then
            # let the controller apply inaccuracy to the decoded target.
            displacement = self.ctrl.calculate_displacement()
        else:
            displacement = 0.0
        outputs = sch.activate(self.net, self.sim, displacement)
        self.ai_outputs = sch.interpret(outputs, self.ctrl)

    # ------------------------------------------------------------------ #
    # per-tick logic

    def _sim_tick(self):
        if self.net:
            events = self.ctrl.step()
        else:
            events = self.sim.tick()
        reprompt = False
        for ev in events:
            kind = ev[0]
            if kind == EV_PICKED:
                self._fx_pick(ev[1])
            elif kind == EV_MISSED:
                self._fx_miss()
            elif kind == EV_TIMER_BONUS:
                self.float_texts.append(FloatText(
                    f"+{ev[1]:.1f}", (BOARD_CENTER[0], BOARD_CENTER[1] - 84), COL_MARKER_BLUE))
            elif kind == EV_GAME_OVER:
                self.end_game()
                return
            if kind == EV_TARGET_REACHED:
                reprompt = True
            elif kind == EV_BAR_SPAWNED and self.net:
                self.ctrl.schedule_prompt(ev[1])
        if self.net and self.state == RUNNING and (
                reprompt or self.ctrl.prompt_due or not self.ctrl.active):
            self._ai_prompt()

    def _fx_pick(self, bar):
        self.assets.play("success", 0.8)
        if random.random() < 0.18:
            self.assets.play_vo("success" if random.random() < 0.5 else "unlocked", 0.8)
        color = COL_MARKER_BLUE if bar.is_blue else COL_MARKER
        x, y = angle_to_xy(BOARD_CENTER, bar.center, (RING_INNER + RING_OUTER) / 2)
        for _ in range(28):
            ang = random.uniform(0, math.tau)
            spd = random.uniform(60, 380)
            self.particles.append(Particle(
                x, y, math.cos(ang) * spd, math.sin(ang) * spd,
                random.uniform(0.25, 0.6), color, random.uniform(1.5, 4)))

    def _fx_miss(self):
        self.assets.play("break", 0.9)
        self.shake_until = self.wall_time + 0.12
        x, y = angle_to_xy(BOARD_CENTER, self.sim.pick_angle, (RING_INNER + RING_OUTER) / 2)
        for _ in range(36):
            ang = random.uniform(0, math.tau)
            spd = random.uniform(80, 460)
            col = random.choice([(255, 120, 40), (255, 60, 40), (255, 200, 80)])
            self.particles.append(Particle(
                x, y, math.cos(ang) * spd, math.sin(ang) * spd,
                random.uniform(0.3, 0.7), col, random.uniform(2, 5)))

    def _update_fx(self, dt):
        for p in self.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vy += 500 * dt
            p.life -= dt
        self.particles = [p for p in self.particles if p.life > 0]
        for t in self.float_texts:
            t.pos[1] -= 60 * dt
            t.life -= dt
        self.float_texts = [t for t in self.float_texts if t.life > 0]

    # ------------------------------------------------------------------ #
    # main loop

    def run(self):
        accumulator = 0.0
        running = True
        while running:
            dt_real = min(self.clock.tick(60) / 1000.0, 0.25)
            self.wall_time += dt_real
            running = self._handle_events()

            if self.state == RUNNING:
                accumulator += dt_real
                tick_dt = self.sim.dt
                while accumulator >= tick_dt and self.state == RUNNING:
                    accumulator -= tick_dt
                    self._sim_tick()
            else:
                accumulator = 0.0
            self._update_fx(dt_real)

            self._draw()
            scaled = pygame.transform.smoothscale(self.canvas, self.window.get_size())
            self.window.blit(scaled, (0, 0))
            pygame.display.flip()
        pygame.quit()

    def _handle_events(self) -> bool:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                return False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_F3:
                    self.debug = not self.debug
                elif ev.key in (pygame.K_F9, pygame.K_ESCAPE):
                    if self.state == RUNNING:
                        self.state = PAUSED
                        pygame.mixer.music.pause() if not self.assets.muted else None
                    elif self.state == PAUSED:
                        self.state = RUNNING
                        pygame.mixer.music.unpause() if not self.assets.muted else None
                    elif ev.key == pygame.K_ESCAPE:
                        return False
                elif ev.key == pygame.K_SPACE and self.state in (MENU, POSTGAME):
                    self.start_game()
                elif ev.key == pygame.K_SPACE and self.state == RUNNING:
                    self._set_boost(key=True)
                elif ev.key == pygame.K_UP and self.state == RUNNING and not self.net:
                    self._human_click()
            if ev.type == pygame.KEYUP and ev.key == pygame.K_SPACE:
                self._set_boost(key=False)
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if self.state in (MENU, POSTGAME):
                    if self._play_button_rect().collidepoint(self._mouse_design(ev.pos)):
                        self.start_game()
                elif self.state == RUNNING and not self.net:
                    if ev.button == 1:
                        self._human_click()
                    elif ev.button == 3:
                        self._set_boost(mouse=True)
            if ev.type == pygame.MOUSEBUTTONUP and ev.button == 3:
                self._set_boost(mouse=False)
        return True

    def _human_click(self):
        for out in self.sim.click():
            if out[0] == EV_PICKED:
                self._fx_pick(out[1])
            elif out[0] == EV_MISSED:
                self._fx_miss()
            elif out[0] == EV_TIMER_BONUS:
                self.float_texts.append(FloatText(
                    f"+{out[1]:.1f}",
                    (BOARD_CENTER[0], BOARD_CENTER[1] - 84), COL_MARKER_BLUE))
            elif out[0] == EV_GAME_OVER:
                self.end_game()

    def _set_boost(self, *, mouse: bool | None = None, key: bool | None = None):
        if mouse is not None:
            self._boost_mouse = mouse
        if key is not None:
            self._boost_key = key
        if self.sim is None or self.net or self.state != RUNNING:
            return
        held = self._boost_mouse or self._boost_key
        if held and not self.sim.rmb_held:
            self.assets.play("boost", 0.5)
        self.sim.rmb_held = held

    def _mouse_design(self, pos):
        return (pos[0] / self.scale, pos[1] / self.scale)

    def _play_button_rect(self) -> pygame.Rect:
        return pygame.Rect(DESIGN_W // 2 - 130, 830, 260, 56)

    # ------------------------------------------------------------------ #
    # drawing

    def _draw(self):
        c = self.canvas
        if self.bg:
            c.blit(self.bg, (0, 0))
        else:
            c.fill((24, 22, 26))

        # Slark behind the bars (window area of the background art)
        head = self.assets.head_frame(self.wall_time)
        if self.head_scaled:
            idx = int(self.wall_time * 1000 / 30) % len(self.head_scaled)
            bob = math.sin(self.wall_time * 1.6) * 2.0
            c.blit(self.head_scaled[idx], (DESIGN_W // 2 - 122 - 65, 80 + bob))
        if self.arm:
            c.blit(self.arm, (DESIGN_W // 2 - 200 - 50, 120))

        # title sign
        if self.sign:
            sway = math.sin(self.wall_time * 0.8) * 2.5
            rot = pygame.transform.rotozoom(self.sign, sway * 0.3, 1.0)
            c.blit(rot, rot.get_rect(center=(DESIGN_W // 2, 67 + sway)))
            label = self.f_sign.render("PICK THE LOCK", True, COL_GOLD_LIGHT)
            c.blit(label, label.get_rect(center=(DESIGN_W // 2, 88 + sway)))

        # lock shank + body
        shake = (0, 0)
        if self.wall_time < self.shake_until:
            shake = (random.randint(-8, 8), random.randint(-3, 3))
        if self.shank:
            c.blit(self.shank, (DESIGN_W // 2 - 153 + shake[0], 310 + shake[1]))
        if self.lock_bg:
            c.blit(self.lock_bg, (BOARD_CENTER[0] - BG_SIZE // 2 + shake[0], BOARD_CENTER[1] - BG_SIZE // 2 + shake[1]))
        center = (BOARD_CENTER[0] + shake[0], BOARD_CENTER[1] + shake[1])

        if self.sim:
            self._draw_board(c, center)
        self._draw_particles(c)
        for t in self.float_texts:
            alpha = int(255 * t.life / t.max_life)
            surf = self.f_bonus.render(t.text, True, t.color)
            surf.set_alpha(alpha)
            c.blit(surf, surf.get_rect(center=t.pos))

        if self.state == MENU:
            self._draw_menu(c)
        elif self.state == PAUSED:
            self._draw_pause(c)
        elif self.state == POSTGAME:
            self._draw_postgame(c)
        elif self.state == RUNNING:
            self._draw_hints(c)

        if self.debug:
            self._draw_debug(c)

    def _draw_board(self, c, center):
        sim = self.sim
        # marker wedges (dial edge -> board rim, gold gradient)
        glow = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        for bar in sim.bars:
            if bar.is_blue:
                inner, outer = (60, 120, 160), COL_MARKER_BLUE
                pulse = 0.925 + 0.075 * math.sin(self.wall_time * math.tau / 0.6)
                inner = tuple(int(ch * pulse) for ch in inner)
                outer = tuple(int(ch * pulse) for ch in outer)
            else:
                inner, outer = (170, 105, 15), (255, 205, 70)
            # draw each bar to its own layer, then add it onto the glow so that
            # where two bars overlap the summed channels read as a lighter hue
            layer = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
            self._ring_wedge(layer, inner, outer, center,
                             bar.center - bar.width / 2, bar.center + bar.width / 2)
            glow.blit(layer, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)
        c.blit(glow, (0, 0))

        # AI target ghost marker
        if self.net and self.ctrl and self.ctrl.active and self.debug:
            ta = self.ctrl.target_angle
            p1 = angle_to_xy(center, ta, RING_INNER)
            p2 = angle_to_xy(center, ta, RING_OUTER + 8)
            pygame.draw.line(c, (120, 255, 120), p1, p2, 2)

        # the lockpick beam (fine-line particle look; red while disabled)
        beam_col = COL_RED if sim.pick_disabled else COL_BEAM
        p1 = angle_to_xy(center, sim.pick_angle, RING_INNER - 6)
        p2 = angle_to_xy(center, sim.pick_angle, RING_OUTER)
        pygame.draw.line(c, (*beam_col, 110), p1, p2, 6)
        pygame.draw.line(c, (*beam_col, 200), p1, p2, 3)
        core = COL_RED if sim.pick_disabled else (255, 255, 255)
        pygame.draw.line(c, core, p1, p2, 1)

        # dial + stats
        if self.dial:
            c.blit(self.dial, (center[0] - DIAL_SIZE // 2, center[1] - DIAL_SIZE // 2))
        else:
            pygame.draw.circle(c, (20, 18, 16), center, DIAL_SIZE // 2)
        title = self.f_stat_title.render("TIME REMAINING", True, COL_GOLD_DARK)
        c.blit(title, title.get_rect(center=(center[0], center[1] - 58)))
        timer_col = COL_RED if sim.time_remaining < 5 else COL_GOLD_LIGHT
        tval = self.f_big.render(f"{sim.time_remaining:.1f}", True, timer_col)
        c.blit(tval, tval.get_rect(center=(center[0], center[1] - 16)))
        title2 = self.f_stat_title.render("SCORE", True, COL_GOLD_DARK)
        c.blit(title2, title2.get_rect(center=(center[0], center[1] + 28)))
        sval = self.f_score.render(f"{sim.score:,}", True, COL_GOLD_LIGHT)
        c.blit(sval, sval.get_rect(center=(center[0], center[1] + 56)))

    def _ring_wedge(self, surf, col_inner, col_outer, center, a0, a1, step=2.0, bands=6):
        """Radial gradient wedge between angles a0..a1 (deg, clockwise from up)."""
        n = max(2, int((a1 - a0) / step) + 1)
        angs = [a0 + (a1 - a0) * i / n for i in range(n + 1)]
        for b in range(bands):
            r0 = RING_INNER + (RING_OUTER - RING_INNER) * b / bands
            r1 = RING_INNER + (RING_OUTER - RING_INNER) * (b + 1) / bands
            f = (b + 0.5) / bands
            col = tuple(int(ci + (co - ci) * f) for ci, co in zip(col_inner, col_outer))
            ring0 = [angle_to_xy(center, a, r0) for a in angs]
            ring1 = [angle_to_xy(center, a, r1) for a in angs]
            pygame.draw.polygon(surf, (*col, 235), ring0 + ring1[::-1])
        # bright rim at the outer edge
        rim = [angle_to_xy(center, a, RING_OUTER) for a in angs]
        pygame.draw.lines(surf, (*col_outer, 255), False, rim, 3)

    def _draw_particles(self, c):
        for p in self.particles:
            alpha = int(255 * p.life / p.max_life)
            r = max(1, int(p.size * p.life / p.max_life))
            surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*p.color, alpha), (r, r), r)
            c.blit(surf, (p.x - r, p.y - r), special_flags=pygame.BLEND_RGBA_ADD)

    # -------------------------------- overlays ------------------------- #

    def _dim(self, c, alpha=185):
        veil = pygame.Surface((DESIGN_W, DESIGN_H), pygame.SRCALPHA)
        veil.fill((8, 6, 10, alpha))
        c.blit(veil, (0, 0))

    def _panel(self, c, rect):
        pygame.draw.rect(c, (22, 19, 24, 240), rect, border_radius=10)
        pygame.draw.rect(c, COL_BRONZE, rect, 2, border_radius=10)

    def _button(self, c, rect, text):
        mouse = self._mouse_design(pygame.mouse.get_pos())
        hov = rect.collidepoint(mouse)
        pygame.draw.rect(c, (90, 60, 25) if hov else (60, 40, 18), rect, border_radius=6)
        pygame.draw.rect(c, COL_GOLD_DARK, rect, 2, border_radius=6)
        label = self.f_title.render(text, True, COL_GOLD_LIGHT)
        c.blit(label, label.get_rect(center=rect.center))

    def _draw_menu(self, c):
        self._dim(c)
        panel = pygame.Rect(85, 190, 580, 560)
        self._panel(c, panel)
        y = panel.top + 36
        for text, font, col in [
            ("PICK THE LOCK", self.f_title, COL_GOLD_LIGHT),
            ("Help Slark break out of his cell!", self.f_med, COL_GREY),
        ]:
            s = font.render(text, True, col)
            c.blit(s, s.get_rect(center=(DESIGN_W // 2, y)))
            y += 42
        y += 14
        lines = [
            ("HOW TO PLAY", COL_GOLD_DARK, self.f_stat_title),
            ("Successfully pick the lock as many times as you can", COL_GREY, self.f_small),
            ("before time runs out.", COL_GREY, self.f_small),
            ("", COL_GREY, self.f_small),
            ("[LMB / UP]  PICK", COL_GOLD_LIGHT, self.f_med),
            ("Activate the lock pick when it reaches the highlighted bar.", COL_GREY, self.f_small),
            ("Missing a bar briefly disables the pick.", COL_GREY, self.f_small),
            ("Blue bars grant additional time.", COL_MARKER_BLUE, self.f_small),
            ("", COL_GREY, self.f_small),
            ("[RMB / SPACE]  BOOST", COL_GOLD_LIGHT, self.f_med),
            ("Hold for continuous speed boost.", COL_GREY, self.f_small),
            ("", COL_GREY, self.f_small),
            ("SCORING", COL_GOLD_DARK, self.f_stat_title),
            (f"Successful Pick ....... {DEFAULT_STAGE.score_per_unlock}", COL_GREY, self.f_small),
            (f"High Score ............ {self.high_score}", COL_GOLD_LIGHT, self.f_small),
        ]
        for text, col, font in lines:
            if text:
                s = font.render(text, True, col)
                c.blit(s, s.get_rect(center=(DESIGN_W // 2, y)))
            y += 26
        self._button(c, self._play_button_rect(), "PLAY")
        if self.net:
            note = self.f_small.render("AI playback mode — the NEAT genome will play", True, (120, 255, 120))
            c.blit(note, note.get_rect(center=(DESIGN_W // 2, 910)))

    def _draw_pause(self, c):
        self._dim(c, 150)
        s = self.f_title.render("PAUSED", True, COL_GOLD_LIGHT)
        c.blit(s, s.get_rect(center=(DESIGN_W // 2, 450)))
        s2 = self.f_med.render("F9 / Esc to resume", True, COL_GREY)
        c.blit(s2, s2.get_rect(center=(DESIGN_W // 2, 500)))

    def _draw_postgame(self, c):
        self._dim(c)
        panel = pygame.Rect(105, 280, 540, 420)
        self._panel(c, panel)
        sim = self.sim
        y = panel.top + 46
        s = self.f_title.render("GAME OVER", True, COL_GOLD_LIGHT)
        c.blit(s, s.get_rect(center=(DESIGN_W // 2, y)))
        y += 66
        rows = [
            (f"Successful Picks  {sim.unlock_count} x {DEFAULT_STAGE.score_per_unlock}",
             f"{sim.unlock_count * DEFAULT_STAGE.score_per_unlock}"),
            ("Total", f"{self.last_final_score}"),
        ]
        for left, right in rows:
            ls = self.f_med.render(left, True, COL_GREY)
            rs = self.f_med.render(right, True, COL_GOLD_LIGHT)
            c.blit(ls, (panel.left + 60, y))
            c.blit(rs, (panel.right - 60 - rs.get_width(), y))
            y += 40
        if self.last_final_score >= self.high_score and self.last_final_score > 0:
            s = self.f_med.render("NEW HIGH SCORE!", True, COL_MARKER)
            c.blit(s, s.get_rect(center=(DESIGN_W // 2, y + 14)))
        y += 56
        s = self.f_small.render(f"High Score: {self.high_score}", True, COL_GREY)
        c.blit(s, s.get_rect(center=(DESIGN_W // 2, y)))
        self._button(c, self._play_button_rect(), "PLAY AGAIN")

    def _draw_hints(self, c):
        left = self.f_small.render("[LMB / UP] PICK", True, COL_GREY)
        c.blit(left, (30, DESIGN_H - 120))
        right = self.f_small.render("BOOST [RMB / SPACE]", True, COL_GREY)
        c.blit(right, (DESIGN_W - 30 - right.get_width(), DESIGN_H - 80))
        pause = self.f_small.render("[F9] PAUSE", True, (110, 110, 110))
        c.blit(pause, (DESIGN_W - 30 - pause.get_width(), 24))

    def _draw_debug(self, c):
        sim = self.sim
        lines = ["-- DEBUG (F3) --"]
        if sim:
            lines += [
                f"t={sim.t:6.2f}  time_left={sim.time_remaining:6.2f}",
                f"angle={sim.pick_angle:7.2f}  dir={'cw' if sim.direction > 0 else 'ccw'}"
                f"  traveled={sim.total_traveled:8.1f}",
                f"speed={sim.current_speed:6.1f} deg/s  boost x{sim.boost_mult:.2f}",
                f"penalty phase={sim.penalty_phase} factor={sim.penalty_factor:.2f}"
                f"  {'DISABLED' if sim.pick_disabled else ''}",
                f"in_zone={'YES' if sim.hittable_bar() else 'no '}",
                f"bars={len(sim.bars)}  spawn_f={sim.spawn_frequency:.2f}/s",
                f"blue pity={sim.blue_chance:.0f}%  unlocks={sim.unlock_count}",
            ]
            for i, (fwd, bar) in enumerate(sim.bars_by_forward_distance()[:3]):
                lines.append(f"bar{i}: fwd={fwd:6.1f} w={bar.width:4.1f} "
                             f"{'BLUE' if bar.is_blue else 'org '}")
        if self.net and self.ai_outputs:
            d, s, k = self.ai_outputs
            lines.append(f"AI: dist={d:6.1f} boost_frac={s:.2f} click={k}")
            if self.ctrl and self.ctrl.active:
                lines.append(f"AI: remaining={self.ctrl.remaining:6.1f}")
        x, y = 14, 14
        for line in lines:
            s = self.f_mono.render(line, True, (140, 255, 140))
            bgr = s.get_rect(topleft=(x, y)).inflate(6, 2)
            pygame.draw.rect(self.canvas, (0, 0, 0, 160), bgr)
            c.blit(s, (x, y))
            y += 17


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description="Pick the Lock — standalone")
    parser.add_argument("--ai", metavar="GENOME_PKL", nargs="?", const="", default=None,
                        help="watch a trained NEAT genome play; with no path given, "
                             "the genome is inferred from the schema and human knobs, i.e. "
                             "models/saved/<schema>/<rt_ms>/<rt_std>/<inacc>/<index>_..._best_genome.pkl")
    parser.add_argument("--index", type=int, default=0,
                        help="when --ai infers the genome from the schema and human knobs, "
                             "pick the saved genome with this index (default 0); ignored if "
                             "--ai is given an explicit path")
    parser.add_argument("--seed", type=int, default=None, help="fixed RNG seed")
    parser.add_argument("--mute", action="store_true", help="disable audio")
    parser.add_argument("--scale", type=float, default=None, help="window scale factor")
    parser.add_argument("--inaccuracy", type=float, default=0.0,
                        help="AI aim error in [0, 1]: gaussian displacement of the "
                             "target distance, scaled by current pick speed")
    parser.add_argument("--reaction_time_ms", type=float, default=0.0,
                        help="AI reaction delay (ms, >= 0) before reprompting on a new bar spawn")
    parser.add_argument("--reaction_time_standard_deviation", type=float, default=0.05,
                        help="relative gaussian jitter of the reaction delay "
                             "(>= 0, typically 0-0.2)")
    parser.add_argument("--schema", type=int, default=0,
                        help="input/output schema id (see pickthelock.schemas); "
                             "must match the schema the genome was trained on")
    args = parser.parse_args(argv)
    if args.schema not in SCHEMAS:
        valid = ", ".join(str(k) for k in sorted(SCHEMAS))
        parser.error(f"--schema {args.schema} unknown; valid schemas: {valid}")
    if not 0.0 <= args.inaccuracy <= 1.0:
        parser.error("--inaccuracy must be between 0 and 1")
    if args.reaction_time_ms < 0.0:
        parser.error("--reaction_time_ms must be non-negative")
    if args.reaction_time_standard_deviation < 0.0:
        parser.error("--reaction_time_standard_deviation must be non-negative")
    ai_path = args.ai
    if ai_path is not None and not ai_path:
        # --ai with no path: infer the saved genome for this schema + knobs + index
        ai_path = paths.find_saved_genome(
            args.schema, args.reaction_time_ms, args.reaction_time_standard_deviation,
            args.inaccuracy, args.index)
        if ai_path is None:
            avail = paths.saved_genomes(args.schema, args.reaction_time_ms,
                                        args.reaction_time_standard_deviation,
                                        args.inaccuracy)
            have = ", ".join(str(i) for i, _ in avail) or "none"
            parser.error(
                f"no saved genome with index {args.index} for schema {args.schema} "
                f"and these human knobs (available indices: {have}).\n"
                "train one with train_neat.py using the same schema and knobs, "
                "or pass an explicit path with --ai <path>.")
    if ai_path is not None and not os.path.isfile(ai_path):
        parser.error(f"AI genome not found: {ai_path}")
    GameApp(ai_genome_path=ai_path, seed=args.seed, muted=args.mute, scale=args.scale,
            inaccuracy=args.inaccuracy, reaction_time_ms=args.reaction_time_ms,
            reaction_time_std=args.reaction_time_standard_deviation,
            schema=args.schema).run()


if __name__ == "__main__":
    main()
