"""Swappable input/output schemas, selected by a `--schema <int>` flag.

A *schema* is the standardized interface between a trained genome and the game.
Prompting is always the same two steps, driven by the caller:

    outputs = schema.activate(net, sim)   # 1. encode state, run the net
    schema.interpret(outputs, ctrl)        # 2. decode outputs, drive the controller

Step 1 (`activate`) is generic for every schema: it runs the schema's own
`build_inputs(sim)` encoding through the net and returns the raw outputs.
Step 2 (`interpret`) is the schema's decision-to-action mapping: it reads the
raw outputs and calls whatever it wants on the controller's interface (schedule
a click, reprompt, a time-based regime, ...). It returns the decoded action so
callers can display it, but its effect is on `ctrl`.

A schema therefore only has to supply two swappable pieces — `build_inputs`
(the input encoding) and `interpret` (the output regime) — plus its I/O sizes:

  num_inputs / num_outputs   override the NEAT config for this schema

Schema 0 reproduces the original hard-wired behavior, so existing genomes keep
working with no flag (default 0).

All game rules stay in the single neat_config.txt; only the I/O sizes differ per
schema and are applied onto the loaded config by apply_config_io() before any
genome or network is created.

Adding a schema: write build_inputs_vN / decode_outputs_vN in observations.py,
write an interpret_vN here, then add an entry to SCHEMAS. Nothing in the entry
points or the config file needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .observations import (
    build_inputs,
    decode_outputs,
    NUM_INPUTS,
    NUM_OUTPUTS,
)


@dataclass(frozen=True)
class Schema:
    build_inputs: Callable[..., list[float]]      # sim -> input vector
    interpret: Callable[..., tuple]               # (outputs, ctrl) -> decoded action
    num_inputs: int
    num_outputs: int

    def activate(self, net, sim):
        """Encode the sim state with this schema and run it through the net."""
        return net.activate(self.build_inputs(sim))


# --------------------------------------------------------------------------- #
# interpreters: raw net outputs -> calls on the controller interface

def interpret_scheduled(outputs, ctrl) -> tuple:
    """Original regime: decode a (distance, boost, click) target and schedule it.

    Aim inaccuracy is applied by the controller (ctrl.schedule), never here, so
    training and play agree — see the controller's `inaccuracy` knob. Returns
    the decoded action for display/telemetry; the effect is on ctrl.
    """
    dist, boost_frac, do_click = decode_outputs(outputs, ctrl.sim)
    ctrl.cancel()                              # a new decision replaces the pending schedule
    ctrl.schedule(dist, boost_frac, do_click)  # controller applies inaccuracy
    return dist, boost_frac, do_click


# --------------------------------------------------------------------------- #
# registry

SCHEMAS: dict[int, Schema] = {
    0: Schema(build_inputs=build_inputs, interpret=interpret_scheduled,
              num_inputs=NUM_INPUTS, num_outputs=NUM_OUTPUTS),
}


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
