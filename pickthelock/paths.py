"""Filesystem layout for trained genomes, checkpoints and graphs.

Everything lives under ``models/``, split into three parallel trees keyed by
the human-imperfection knobs a genome was trained with
(``reaction_time_ms`` / ``reaction_time_standard_deviation`` / ``inaccuracy``)::

    models/saved/<rt_ms>/<rt_std>/<inacc>/<index>_<timestamp>_<score>_best_genome.pkl
    models/graphs/<rt_ms>/<rt_std>/<inacc>/<index>_<timestamp>_<score>_best_genome.html
    models/temp/<rt_ms>/<rt_std>/<inacc>/<run_id>/
        best_genome.pkl
        winner_genome.pkl
        fitness_history.csv
        checkpoints/neat-checkpoint-N

So a run with the default human knobs (reaction_time_ms=0.0,
reaction_time_standard_deviation=0.05, inaccuracy=0.0) lands under
``.../0.0/0.05/0.0/``. The extra ``<run_id>`` layer under ``temp/`` keeps
parallel training processes from clobbering each other's in-progress files;
on termination the best genome is promoted from there into ``saved/`` under a
unique ``<index>_<timestamp>_<score>_best_genome.pkl`` name (the index
auto-increments per parameter set, so successive runs never overwrite). The
consumers (play.py, graph_genome.py) select one by its index, defaulting to 0.
"""

from __future__ import annotations

import os
import re
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(ROOT, "models")
SAVED_DIR = os.path.join(MODELS_DIR, "saved")
TEMP_DIR = os.path.join(MODELS_DIR, "temp")
GRAPHS_DIR = os.path.join(MODELS_DIR, "graphs")

BEST_GENOME_NAME = "best_genome.pkl"
WINNER_GENOME_NAME = "winner_genome.pkl"
HISTORY_NAME = "fitness_history.csv"
CHECKPOINTS_NAME = "checkpoints"
CHECKPOINT_PREFIX = "neat-checkpoint-"


def param_subpath(reaction_time_ms: float, reaction_time_std: float,
                  inaccuracy: float) -> str:
    """'<rt_ms>/<rt_std>/<inacc>' folder triple for these human knobs."""
    return os.path.join(str(reaction_time_ms), str(reaction_time_std), str(inaccuracy))


# --------------------------------------------------------------------- #
# saved/ — the promoted best genome for a parameter set

def saved_dir(reaction_time_ms: float, reaction_time_std: float,
              inaccuracy: float) -> str:
    return os.path.join(SAVED_DIR,
                        param_subpath(reaction_time_ms, reaction_time_std, inaccuracy))


# saved genomes are named "<index>_<timestamp>_<score>_best_genome.pkl"; the
# leading integer is the auto-incrementing index the consumers select by.
_SAVED_RE = re.compile(r"^(\d+)_.+_" + re.escape(BEST_GENOME_NAME) + r"$")


def saved_genome_filename(index: int, score, when: float | None = None) -> str:
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(when))
    return f"{index}_{ts}_{score}_{BEST_GENOME_NAME}"


def saved_genomes(reaction_time_ms: float, reaction_time_std: float,
                  inaccuracy: float) -> list[tuple[int, str]]:
    """(index, path) for every saved genome at these knobs, sorted by index."""
    d = saved_dir(reaction_time_ms, reaction_time_std, inaccuracy)
    found = []
    if os.path.isdir(d):
        for name in os.listdir(d):
            m = _SAVED_RE.match(name)
            if m:
                found.append((int(m.group(1)), os.path.join(d, name)))
    found.sort(key=lambda t: t[0])
    return found


def next_saved_index(reaction_time_ms: float, reaction_time_std: float,
                     inaccuracy: float) -> int:
    """The index a newly promoted genome should take (max existing + 1, else 0)."""
    existing = saved_genomes(reaction_time_ms, reaction_time_std, inaccuracy)
    return existing[-1][0] + 1 if existing else 0


def new_saved_genome_path(reaction_time_ms: float, reaction_time_std: float,
                          inaccuracy: float, score, when: float | None = None) -> str:
    """Full destination path for a freshly promoted genome (auto-assigned index)."""
    index = next_saved_index(reaction_time_ms, reaction_time_std, inaccuracy)
    name = saved_genome_filename(index, score, when)
    return os.path.join(saved_dir(reaction_time_ms, reaction_time_std, inaccuracy), name)


def find_saved_genome(reaction_time_ms: float, reaction_time_std: float,
                      inaccuracy: float, index: int = 0) -> str | None:
    """Path of the saved genome with this index, or None if absent."""
    for idx, path in saved_genomes(reaction_time_ms, reaction_time_std, inaccuracy):
        if idx == index:
            return path
    return None


# --------------------------------------------------------------------- #
# graphs/ — mirrors the saved tree, one .html per model

def graphs_dir(reaction_time_ms: float, reaction_time_std: float,
               inaccuracy: float) -> str:
    return os.path.join(GRAPHS_DIR,
                        param_subpath(reaction_time_ms, reaction_time_std, inaccuracy))


def graph_path_for_model(model_path: str) -> str:
    """Default graph output for ``model_path``.

    A model living under ``saved/`` mirrors to the same path under ``graphs/``
    with an ``.html`` extension; anything else falls back to a sibling ``.html``.
    """
    model_abs = os.path.abspath(model_path)
    try:
        rel = os.path.relpath(model_abs, SAVED_DIR)
    except ValueError:
        rel = os.pardir  # different drive on Windows -> not under saved/
    if not rel.startswith(os.pardir):
        return os.path.splitext(os.path.join(GRAPHS_DIR, rel))[0] + ".html"
    return os.path.splitext(model_abs)[0] + ".html"


# --------------------------------------------------------------------- #
# temp/ — per-run scratch space (history, in-progress genomes, checkpoints)

def temp_run_dir(reaction_time_ms: float, reaction_time_std: float,
                 inaccuracy: float, run_id) -> str:
    return os.path.join(TEMP_DIR,
                        param_subpath(reaction_time_ms, reaction_time_std, inaccuracy),
                        str(run_id))


def checkpoints_dir(run_dir: str) -> str:
    return os.path.join(run_dir, CHECKPOINTS_NAME)


def checkpoint_prefix(run_dir: str) -> str:
    """filename_prefix neat.Checkpointer writes 'neat-checkpoint-N' under."""
    return os.path.join(checkpoints_dir(run_dir), CHECKPOINT_PREFIX)
