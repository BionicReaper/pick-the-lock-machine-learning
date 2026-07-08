"""NEAT input/output encoding.

Inputs (NUM_INPUTS = 20), all roughly normalized to 0..1:
  For each of the N_TRACKED_BARS = 3 nearest bars, sorted by *travel needed
  until a click would hit* (0 while the pick is inside the bar's hit zone,
  so a bar just passed but still hittable stays in slot 0 instead of
  teleporting to "a full lap away"):
    0. forward distance to center / 360   (in current travel direction)
    1. reverse distance to center / 360   (360 - forward; small when the
       center was just passed)
    2. is_blue (0/1)
    3. current width / initial width
    (missing slots are padded with fwd=1, rev=1, blue=0, width=0)
  12. in_zone: 1 if a click right now would hit a bar
  13. time remaining / time limit (can exceed 1 after blue bonuses)
  14. boost multiplier, normalized: (mult-1)/(max-1)
  15. penalty factor (1 = healthy, <1 = slowed)
  16. pick_disabled: 1 while clicks are ignored (red pick)
  17. spawn frequency / (3 / base interval)
  18. current blue pity chance / 100
  19. current speed / max possible speed

Outputs (NUM_OUTPUTS = 3), expected in 0..1 (sigmoid):
  0. target distance   -> degrees = value * 360 (min clamped by tuning)
  1. average speed     -> fraction of the travel with RMB held (held first)
  2. click             -> click at the target if > 0.5, else just reprompt
"""

from __future__ import annotations

from .sim import LockpickingSim, Bar

N_TRACKED_BARS = 3
NUM_INPUTS = N_TRACKED_BARS * 4 + 8
NUM_OUTPUTS = 3


def travel_to_hit(sim: LockpickingSim, bar: Bar) -> float:
    """Degrees of travel until a click would land on `bar`.

    0 while the pick is anywhere inside the hit zone (center ahead OR just
    behind). Once the pick exits past the far edge, hitting the bar again
    genuinely requires almost a full lap, and the value reflects that."""
    raw = sim.ang_fwd(bar.center)
    half = bar.width / 2.0 + sim.tuning.hit_buffer_deg
    if raw <= half or raw >= 360.0 - half:
        return 0.0
    return raw - half


def build_inputs(sim: LockpickingSim) -> list[float]:
    s = sim.stage
    obs: list[float] = []
    ordered = sorted(((travel_to_hit(sim, b), sim.ang_fwd(b.center), b)
                      for b in sim.bars), key=lambda p: p[0])
    for i in range(N_TRACKED_BARS):
        if i < len(ordered):
            _, fwd, bar = ordered[i]
            obs.append(fwd / 360.0)
            obs.append((360.0 - fwd) / 360.0)
            obs.append(1.0 if bar.is_blue else 0.0)
            obs.append(bar.width / sim.initial_bar_width)
        else:
            obs.extend((1.0, 1.0, 0.0, 0.0))
    obs.append(1.0 if sim.hittable_bar() is not None else 0.0)
    obs.append(sim.time_remaining / s.time_limit)
    obs.append((sim.boost_mult - 1.0) / max(1e-6, s.max_speed_multiplier - 1.0))
    obs.append(sim.penalty_factor)
    obs.append(1.0 if sim.pick_disabled else 0.0)
    obs.append(sim.spawn_frequency * s.base_unlock_appear_rate / 3.0)
    obs.append(sim.blue_chance / 100.0)
    obs.append(sim.current_speed / max(1e-6, sim.max_speed))
    return obs


def decode_outputs(outputs, sim: LockpickingSim) -> tuple[float, float, bool]:
    """Map raw net outputs to (distance_deg, boost_hold_frac, do_click)."""
    dist01 = min(1.0, max(0.0, float(outputs[0])))
    speed01 = min(1.0, max(0.0, float(outputs[1])))
    click = float(outputs[2]) > 0.5
    distance = max(sim.tuning.min_target_distance_deg, dist01 * 360.0)
    return distance, speed01, click
