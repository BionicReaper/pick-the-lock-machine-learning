"""NEAT input/output encoding.

Inputs are described as *feature keys* rather than a fixed vector. FEATURE_MAP
maps every available key to the function that retrieves it, and sample_state()
runs the whole map once to produce {key: value} for the current sim state. A
schema then picks an ordered subset of those keys (its input_dictionary), and
build_inputs(sim, input_dictionary) returns exactly those values, in that order.

Key naming convention:  <property>_<format>[_<parameter>]

  <property>   what is measured, e.g. bar_forward_distance, current_speed
  <format>     how it is encoded — keep this explicit so future encodings of
               the same property don't need a rename:
                 percentage  normalized to ~0..1 by a reference/max
                 ratio       a raw factor already in a bounded range
                 boolean     0.0 / 1.0
               (future examples: radians, degrees for angles)
  <parameter>  optional index, e.g. the bar slot (1 = nearest)

Available keys (32 for the default schema):
  For each of the N_TRACKED_BARS = 6 nearest *perceived* bars, ranked by travel
  needed until a click would hit (0 while the pick is inside the hit zone), slot
  1 = nearest.  Bars still inside their reaction delay (bar.perceived False, see
  the controller) are invisible: they fill no slot and never set the hit-zone
  flag, so prompts fired before the reaction lands can't act on a bar the
  "human" hasn't noticed yet.  Missing slots pad to fwd=1, rev=1, blue=0, w=0.
    bar_forward_distance_percentage_<i>   forward travel to center / 360
    bar_reverse_distance_percentage_<i>   (360 - forward) / 360
    bar_is_blue_boolean_<i>               1 if blue
    bar_width_percentage_<i>              current width / initial width
  Globals:
    pick_in_hit_zone_boolean       1 if a click right now would hit a perceived bar
    time_remaining_percentage      time remaining / time limit (can exceed 1)
    boost_multiplier_percentage    (boost_mult - 1) / (max_mult - 1)
    penalty_factor_ratio           1 = healthy, <1 = slowed
    pick_disabled_boolean          1 while clicks are ignored (red pick)
    spawn_interval_ratio           spawn interval / base spawn interval
    blue_chance_percentage         current blue pity chance / 100
    current_speed_percentage       current speed / max possible speed

Outputs (NUM_OUTPUTS = 3), expected in 0..1 (sigmoid):
  0. target distance   -> degrees = value * 360 (min clamped by tuning)
  1. average speed     -> fraction of the travel with RMB held (held first)
  2. click             -> click at the target if > 0.5, else just reprompt
"""

from __future__ import annotations

from typing import Callable, Sequence

from .sim import LockpickingSim, Bar

N_TRACKED_BARS = 6
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


# --------------------------------------------------------------------------- #
# feature retrieval

class _Sampler:
    """Shared context the feature getters read from.

    Sorts the perceived bars once (nearest-first by travel-to-hit) so every
    per-bar getter is a cheap index instead of re-sorting.
    """
    __slots__ = ("sim", "bars")

    def __init__(self, sim: LockpickingSim):
        self.sim = sim
        # (travel_to_hit, forward_distance, bar), nearest-first
        self.bars = sorted(
            ((travel_to_hit(sim, b), sim.ang_fwd(b.center), b)
             for b in sim.bars if b.perceived),
            key=lambda p: p[0])


# per-bar getters: factories closing over the 0-based slot index

def _bar_forward_distance_normalized_360(idx: int) -> Callable[[_Sampler], float]:
    def get(s: _Sampler) -> float:
        return s.bars[idx][1] / 360.0 if idx < len(s.bars) else 1.0
    return get


def _bar_reverse_distance_normalized_360(idx: int) -> Callable[[_Sampler], float]:
    def get(s: _Sampler) -> float:
        return (360.0 - s.bars[idx][1]) / 360.0 if idx < len(s.bars) else 1.0
    return get


def _bar_is_blue_boolean(idx: int) -> Callable[[_Sampler], float]:
    def get(s: _Sampler) -> float:
        return 1.0 if idx < len(s.bars) and s.bars[idx][2].is_blue else 0.0
    return get


def _bar_width_normalized_max_width(idx: int) -> Callable[[_Sampler], float]:
    def get(s: _Sampler) -> float:
        return s.bars[idx][2].width / s.sim.initial_bar_width if idx < len(s.bars) else 0.0
    return get

def _bar_width_normalized_360(idx: int) -> Callable[[_Sampler], float]:
    def get(s: _Sampler) -> float:
        return s.bars[idx][2].width / 360.0 if idx < len(s.bars) else 0.0
    return get


# global getters

def _pick_in_hit_zone_boolean(s: _Sampler) -> float:
    hittable = s.sim.hittable_bar()
    return 1.0 if hittable is not None and hittable.perceived else 0.0


def _time_remaining_normalized_time_limit(s: _Sampler) -> float:
    return s.sim.time_remaining / s.sim.stage.time_limit


def _boost_multiplier_normalized_max_multiplier(s: _Sampler) -> float:
    return (s.sim.boost_mult - 1.0) / max(1e-6, s.sim.stage.max_speed_multiplier - 1.0)


def _penalty_factor_ratio(s: _Sampler) -> float:
    return s.sim.penalty_factor


def _pick_disabled_boolean(s: _Sampler) -> float:
    return 1.0 if s.sim.pick_disabled else 0.0


def _spawn_interval_normalized_base_unlock_appear_rate(s: _Sampler) -> float:
    return s.sim.spawn_interval / s.sim.stage.base_unlock_appear_rate


def _spawn_interval_normalized_time_limit(s: _Sampler) -> float:
    return s.sim.spawn_interval / s.sim.stage.time_limit


def _blue_chance_percentage(s: _Sampler) -> float:
    return s.sim.blue_chance / 100.0


def _current_speed_normalized_max_speed(s: _Sampler) -> float:
    return s.sim.current_speed / max(1e-6, s.sim.max_speed)


def _current_speed_normalized_360(s: _Sampler) -> float:
    return s.sim.current_speed / 360.0


def _time_to_next_spawn_normalized_time_limit(s: _Sampler) -> float:
    """Fraction of the spawn interval until the next bar appears."""
    return s.sim.spawn_timer / s.sim.stage.time_limit


# --------------------------------------------------------------------------- #
# registry: key -> getter. Extend this to add a new observation.

FEATURE_MAP: dict[str, Callable[[_Sampler], float]] = {}
for _slot in range(N_TRACKED_BARS):
    _n = _slot + 1  # keys are 1-indexed: slot 1 is the nearest bar
    FEATURE_MAP[f"bar_forward_distance_normalized_360_{_n}"] = _bar_forward_distance_normalized_360(_slot)
    FEATURE_MAP[f"bar_reverse_distance_normalized_360_{_n}"] = _bar_reverse_distance_normalized_360(_slot)
    FEATURE_MAP[f"bar_is_blue_boolean_{_n}"] = _bar_is_blue_boolean(_slot)
    FEATURE_MAP[f"bar_width_normalized_max_width_{_n}"] = _bar_width_normalized_max_width(_slot)
    FEATURE_MAP[f"bar_width_normalized_360_{_n}"] = _bar_width_normalized_360(_slot)
FEATURE_MAP["pick_in_hit_zone_boolean"] = _pick_in_hit_zone_boolean
FEATURE_MAP["time_remaining_normalized_time_limit"] = _time_remaining_normalized_time_limit
FEATURE_MAP["boost_multiplier_normalized_max_multiplier"] = _boost_multiplier_normalized_max_multiplier
FEATURE_MAP["penalty_factor_ratio"] = _penalty_factor_ratio
FEATURE_MAP["pick_disabled_boolean"] = _pick_disabled_boolean
FEATURE_MAP["spawn_interval_normalized_base_unlock_appear_rate"] = _spawn_interval_normalized_base_unlock_appear_rate
FEATURE_MAP["spawn_interval_normalized_time_limit"] = _spawn_interval_normalized_time_limit
FEATURE_MAP["blue_chance_percentage"] = _blue_chance_percentage
FEATURE_MAP["current_speed_normalized_max_speed"] = _current_speed_normalized_max_speed
FEATURE_MAP["current_speed_normalized_360"] = _current_speed_normalized_360
FEATURE_MAP["time_to_next_spawn_normalized_time_limit"] = _time_to_next_spawn_normalized_time_limit


# The default schema's ordered inputs: every per-bar feature interleaved per
# slot, then the globals — reproducing the original build_inputs(sim) vector.
DEFAULT_INPUT_KEYS: tuple[str, ...] = tuple(
    key
    for n in range(1, N_TRACKED_BARS + 1)
    for key in (f"bar_forward_distance_normalized_360_{n}",
                f"bar_reverse_distance_normalized_360_{n}",
                f"bar_is_blue_boolean_{n}",
                f"bar_width_normalized_max_width_{n}")
) + (
    "pick_in_hit_zone_boolean",
    "time_remaining_normalized_time_limit",
    "boost_multiplier_normalized_max_multiplier",
    "penalty_factor_ratio",
    "pick_disabled_boolean",
    "spawn_interval_normalized_base_unlock_appear_rate",
    "blue_chance_percentage",
    "current_speed_normalized_max_speed",
)

NUM_INPUTS = len(DEFAULT_INPUT_KEYS)


def sample_state(sim: LockpickingSim) -> dict[str, float]:
    """Retrieve every available feature for the current sim state."""
    s = _Sampler(sim)
    return {key: get(s) for key, get in FEATURE_MAP.items()}


def build_inputs(sim: LockpickingSim, input_dictionary: Sequence[str]) -> list[float]:
    """Activation inputs for one schema: the selected feature keys, in order."""
    state = sample_state(sim)
    try:
        return [state[key] for key in input_dictionary]
    except KeyError as e:
        raise KeyError(f"input key {e.args[0]!r} is not in FEATURE_MAP; "
                       f"available keys: {sorted(FEATURE_MAP)}") from None


def decode_outputs(outputs, sim: LockpickingSim) -> tuple[float, float, bool]:
    """Map raw net outputs to (distance_deg, boost_hold_frac, do_click)."""
    dist01 = min(1.0, max(0.0, float(outputs[0])))
    speed01 = min(1.0, max(0.0, float(outputs[1])))
    click = float(outputs[2]) > 0.5
    distance = max(sim.tuning.min_target_distance_deg, dist01 * 360.0)
    return distance, speed01, click
