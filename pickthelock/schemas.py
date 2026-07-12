"""Swappable input/output schemas, selected by a `--schema <int>` flag.

A *schema* is the standardized interface between a trained genome and the game.
Prompting is always the same two steps, driven by the caller:

    outputs = schema.activate(net, sim)   # 1. encode state, run the net
    schema.interpret(outputs, ctrl)        # 2. decode outputs, drive the controller

Step 1 (`activate`) is generic for every schema: it runs the schema's ordered
`input_dictionary` of feature keys through observations.build_inputs and then
through the net, returning the raw outputs. Swapping the input encoding is thus
just choosing a different tuple of keys (see observations.FEATURE_MAP).
Step 2 (`interpret`) is the schema's decision-to-action mapping: it reads the
raw outputs and calls whatever it wants on the controller's interface (schedule
a click, reprompt, a time-based regime, ...). It returns the decoded action so
callers can display it, but its effect is on `ctrl`.

A schema therefore supplies two swappable pieces — `input_dictionary` (which
feature keys, in what order) and `interpret` (the output regime) — plus its
output size. num_inputs is derived from the input_dictionary length.

Schema 0 reproduces the original hard-wired behavior, so existing genomes keep
working with no flag (default 0).

All game rules stay in the single neat_config.txt; only the I/O sizes differ per
schema and are applied onto the loaded config by apply_config_io() before any
genome or network is created.

Adding a schema: register any new feature keys in observations.FEATURE_MAP,
write an interpret_vN here, then add an entry to SCHEMAS with its ordered
input_dictionary. Nothing in the entry points or the config file changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from pickthelock.sim import LockpickingSim

from .observations import (
    build_inputs,
    FEATURE_MAP,
    DEFAULT_INPUT_KEYS,
)


@dataclass(frozen=True)
class Schema:
    input_dictionary: Sequence[str]      # ordered feature keys into FEATURE_MAP
    interpret: Callable[..., tuple]       # (outputs, ctrl) -> decoded action
    num_outputs: int
    use_input_displacement: bool  # degrees, travel-direction perturbation for observations.build_inputs
    output_keys: Sequence[str]    # ordered human names for outputs 0..num_outputs-1; used by graph_genome.py

    @property
    def num_inputs(self) -> int:
        return len(self.input_dictionary)

    def activate(self, net, sim, displacement: float = 0.0):
        """Encode the sim state with this schema and run it through the net.

        `displacement` (degrees, travel-direction) positionally perturbs the
        pick for the encoded observation — see observations.build_inputs and
        LockpickingSim.perturbed_angle. Defaults to 0.0 (the live pick)."""
        return net.activate(build_inputs(sim, self.input_dictionary, displacement))

# --------------------------------------------------------------------------- #
# output_decoders: interpret the raw net outputs into calls on the controller interface

def decode_outputs_default(outputs, sim: LockpickingSim) -> tuple[float, float, bool]:
    """Map raw net outputs to (distance_deg, boost_hold_frac, do_click)."""
    dist01 = min(1.0, max(0.0, float(outputs[0])))
    speed01 = min(1.0, max(0.0, float(outputs[1])))
    click = float(outputs[2]) > 0.5
    distance = max(sim.tuning.min_target_distance_deg, dist01 * 360.0)
    return distance, speed01, click

def decode_outputs_schema_1(outputs) -> tuple[bool, bool]:
    """Map raw net outputs to (distance_deg, boost_hold_frac, do_click).

    Schema 1 has no click output; the controller always clicks at the end of
    the boost hold. The third return value is always False."""
    click = outputs[0] > 0.5
    boost = outputs[1] > 0.5
    return click, boost


# --------------------------------------------------------------------------- #
# interpreters: raw net outputs -> calls on the controller interface

def interpret_scheduled_default(outputs, ctrl) -> tuple:
    """Original regime: decode a (distance, boost, click) target and schedule it.

    Aim inaccuracy is applied by the controller (ctrl.schedule), never here, so
    training and play agree — see the controller's `inaccuracy` knob. Returns
    the decoded action for display/telemetry; the effect is on ctrl.
    """
    dist, boost_frac, do_click = decode_outputs_default(outputs, ctrl.sim)
    ctrl.cancel()                              # a new decision replaces the pending schedule
    ctrl.schedule(dist, boost_frac, do_click)  # controller applies inaccuracy
    return dist, boost_frac, do_click

def interpret_instant(outputs, ctrl) -> tuple:
    """Instant regime: decode a (distance, boost) target and apply it immediately.

    Aim inaccuracy is applied by the controller (ctrl.schedule), never here, so
    training and play agree — see the controller's `inaccuracy` knob. Returns
    the decoded action for display/telemetry; the effect is on ctrl.
    """
    click, boost = decode_outputs_schema_1(outputs)
    boost_frac = 1.0 if boost else 0.0
    ctrl.cancel()                              # a new decision replaces the pending schedule
    ctrl.schedule(ctrl.sim.tuning.min_target_distance_deg, boost_frac, do_click=click)  # controller applies inaccuracy
    return click, boost



# --------------------------------------------------------------------------- #
# Schema 1: improving normalization of the inputs and adding missing time
# to next spawn

SCHEMA_1_INPUT_KEYS: tuple[str, ...] = tuple(
    key
    for n in range(1, 6 + 1)
    for key in (f"bar_forward_distance_normalized_360_{n}",
                f"bar_reverse_distance_normalized_360_{n}",
                f"bar_is_blue_boolean_{n}",
                f"bar_width_normalized_360_{n}")
) + (
    "pick_in_hit_zone_boolean",
    "time_remaining_normalized_time_limit",
    "penalty_factor_ratio",
    "pick_disabled_boolean",
    "spawn_interval_normalized_time_limit",
    "blue_chance_percentage",
    "current_speed_normalized_360",
    "time_to_next_spawn_normalized_time_limit"
)

# --------------------------------------------------------------------------- #
# registry

SCHEMAS: dict[int, Schema] = {
    0: Schema(input_dictionary=DEFAULT_INPUT_KEYS, interpret=interpret_scheduled_default,
              num_outputs=3, use_input_displacement=False,
              output_keys=("target distance", "hold speed", "click")),
    1: Schema(input_dictionary=SCHEMA_1_INPUT_KEYS, interpret=interpret_instant,
              num_outputs=2, use_input_displacement=False,
              output_keys=("click", "boost")),
    2: Schema(input_dictionary=SCHEMA_1_INPUT_KEYS, interpret=interpret_instant,
              num_outputs=2, use_input_displacement=True,
              output_keys=("click", "boost")),
}

# fail fast on a typo'd or unregistered key in any schema's input_dictionary
for _sid, _schema in SCHEMAS.items():
    _unknown = [k for k in _schema.input_dictionary if k not in FEATURE_MAP]
    if _unknown:
        raise KeyError(f"schema {_sid} references unknown feature keys {_unknown}; "
                       f"register them in observations.FEATURE_MAP")
    if len(_schema.output_keys) != _schema.num_outputs:
        raise ValueError(f"schema {_sid} has {len(_schema.output_keys)} output_keys "
                         f"but num_outputs={_schema.num_outputs}")


def get_schema(n: int) -> Schema:
    try:
        return SCHEMAS[n]
    except KeyError:
        valid = ", ".join(str(k) for k in sorted(SCHEMAS))
        raise KeyError(f"unknown schema {n!r}; valid schemas: {valid}") from None


def apply_config_io(config, schema: Schema) -> None:
    """Override just the I/O sizes on a loaded NEAT config for this schema.

    Sets both the counts and the derived input/output key lists, so genome
    creation (training) and FeedForwardNetwork.create (playback) both build the
    right number of nodes. Must run before either happens.
    """
    gc = config.genome_config
    gc.num_inputs = schema.num_inputs
    gc.input_keys = [-i - 1 for i in range(schema.num_inputs)]
    gc.num_outputs = schema.num_outputs
    gc.output_keys = [i for i in range(schema.num_outputs)]
