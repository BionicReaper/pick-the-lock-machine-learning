"""Loading of the extracted Dota 2 assets (images / sounds).

Everything degrades gracefully: if a file is missing the game still runs
with simple placeholder shapes, so the sim/trainer never depend on assets.
"""

from __future__ import annotations

import os
import random

import pygame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTRACTED = os.path.join(ROOT, "extracted")
IMG_DIR = os.path.join(EXTRACTED, "panorama", "images", "events", "dark_carnival", "lockpicking")
SFX_DIR = os.path.join(EXTRACTED, "sounds", "misc", "dark_carnival")
VO_DIR = os.path.join(EXTRACTED, "sounds", "vo", "event_dark_carnival_slark_games")

IMAGES = {
    "background": "lockpicking_background_psd.png",
    "lock_background": "lock_background_psd.png",
    "lock_dial": "lock_dial_psd.png",
    "lock_shank": "lock_shank_psd.png",
    "sign": "sign_psd.png",
    "slark_arm": "slark_arm_psd.png",
    "slark_head": "slark_head_psd.png",
}

SFX = {
    "success": "lockpick_success.mp3",
    "break": "lockpick_break.mp3",
    "boost": "lockpick_boost.mp3",
    "lose": "minigame_mus_lose_01.mp3",
    "win": "minigame_mus_win_01.mp3",
}

MUSIC = ["minigame_mus_lp_01.mp3", "minigame_mus_lp_02.mp3", "minigame_mus_lp_03.mp3"]

VO_PREFIXES = {
    "start": "slark_lockpick_start_",
    "success": "slark_lockpick_success_",
    "unlocked": "slark_lockpick_unlocked_",
    "gameover": "slark_lockpick_gameover_",
}

# AnimatedImageStrip parameters from popup_dark_carnival_encounter_lockpicking.xml
HEAD_FRAME_W = 420
HEAD_FRAME_H = 500
HEAD_FRAME_TIME_MS = 30


class Assets:
    def __init__(self, muted: bool = False):
        self.images: dict[str, pygame.Surface | None] = {}
        self.sounds: dict[str, pygame.mixer.Sound | None] = {}
        self.vo: dict[str, list[pygame.mixer.Sound]] = {}
        self.head_frames: list[pygame.Surface] = []
        self.music_paths: list[str] = []
        self.muted = muted
        self._load_images()
        if not muted:
            self._load_sounds()

    # ------------------------------------------------------------------ #

    def _load_images(self) -> None:
        for key, fname in IMAGES.items():
            path = os.path.join(IMG_DIR, fname)
            try:
                self.images[key] = pygame.image.load(path).convert_alpha()
            except (pygame.error, FileNotFoundError, OSError):
                self.images[key] = None
        self._slice_head_strip()

    def _slice_head_strip(self) -> None:
        # slark_head_psd.png is a row-major grid of 420x500 frames
        # (retail export: 3360x4000 = 8 cols x 8 rows = 64 frames)
        strip = self.images.get("slark_head")
        if strip is None:
            return
        w, h = strip.get_size()
        cols = w // HEAD_FRAME_W if w % HEAD_FRAME_W == 0 else 1
        rows = h // HEAD_FRAME_H if h % HEAD_FRAME_H == 0 else 1
        if cols == 1 and rows == 1:
            self.head_frames = [strip]
            return
        rects = [pygame.Rect(cx * HEAD_FRAME_W, cy * HEAD_FRAME_H, HEAD_FRAME_W, HEAD_FRAME_H)
                 for cy in range(rows) for cx in range(cols)]
        self.head_frames = [strip.subsurface(r) for r in rects]

    def head_frame(self, t_seconds: float) -> pygame.Surface | None:
        if not self.head_frames:
            return None
        idx = int(t_seconds * 1000.0 / HEAD_FRAME_TIME_MS) % len(self.head_frames)
        return self.head_frames[idx]

    # ------------------------------------------------------------------ #

    def _load_sounds(self) -> None:
        for key, fname in SFX.items():
            path = os.path.join(SFX_DIR, fname)
            try:
                self.sounds[key] = pygame.mixer.Sound(path)
            except (pygame.error, FileNotFoundError, OSError):
                self.sounds[key] = None
        for key, prefix in VO_PREFIXES.items():
            clips = []
            if os.path.isdir(VO_DIR):
                for fname in sorted(os.listdir(VO_DIR)):
                    if fname.startswith(prefix) and fname.endswith(".mp3"):
                        try:
                            clips.append(pygame.mixer.Sound(os.path.join(VO_DIR, fname)))
                        except (pygame.error, OSError):
                            pass
            self.vo[key] = clips
        self.music_paths = [os.path.join(SFX_DIR, m) for m in MUSIC
                            if os.path.isfile(os.path.join(SFX_DIR, m))]

    def play(self, key: str, volume: float = 1.0) -> None:
        snd = self.sounds.get(key)
        if snd is not None and not self.muted:
            snd.set_volume(volume)
            snd.play()

    def play_vo(self, key: str, volume: float = 0.9) -> None:
        clips = self.vo.get(key) or []
        if clips and not self.muted:
            snd = random.choice(clips)
            snd.set_volume(volume)
            snd.play()

    def start_music(self, volume: float = 0.35) -> None:
        if self.music_paths and not self.muted:
            try:
                pygame.mixer.music.load(random.choice(self.music_paths))
                pygame.mixer.music.set_volume(volume)
                pygame.mixer.music.play(-1)
            except pygame.error:
                pass

    def stop_music(self) -> None:
        if not self.muted:
            try:
                pygame.mixer.music.stop()
            except pygame.error:
                pass
