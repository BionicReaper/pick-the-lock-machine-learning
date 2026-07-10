"""Headlessly run a saved NEAT genome over many simulations and report score stats.

No pygame — this drives the same headless episode loop training uses
(train_neat.run_episode), so scores match what a genome sees during evaluation.
The genome, human-imperfection knobs and schema are selected exactly like play.py:

    .venv\\Scripts\\python.exe evaluate.py --ai PATH/genome.pkl --runs 100
    .venv\\Scripts\\python.exe evaluate.py --runs 100          (infer genome from schema + knobs + --index)
    .venv\\Scripts\\python.exe evaluate.py --runs 100 --index 2 --inaccuracy 0.05

--ai with no path infers the saved genome from the schema and human knobs,
picking the one with --index (default 0), i.e.
    models/saved/<schema>/<rt_ms>/<rt_std>/<inacc>/<index>_..._best_genome.pkl

Each of --runs simulations uses a distinct seed (seed_base + i). By default
seed_base is randomized per invocation (and printed, so a run can be repeated
with --seed-base); pin it for a reproducible/comparable evaluation. Raw per-run
scores are aggregated (mean / min / max / median / stdev) and the training
fitness blend (W_AVG*avg + W_WORST*worst + W_BEST*best) is shown for reference.
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import pickle
import random
import statistics

import neat

from pickthelock import paths
from pickthelock.config import DEFAULT_STAGE
from pickthelock.schemas import SCHEMAS, get_schema, apply_config_io
from train_neat import run_episode, W_AVG, W_WORST, W_BEST, MAX_EPISODE_SECONDS

ROOT = paths.ROOT
CONFIG_PATH = os.path.join(ROOT, "neat_config.txt")

# a run "wins" by reaching num_unlocks unlocks; since score is exactly
# unlock_count * score_per_unlock and that's the only win path, a run won iff
# its score reached this threshold
WIN_SCORE = DEFAULT_STAGE.num_unlocks * DEFAULT_STAGE.score_per_unlock


def load_net(genome_path: str, schema: int):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG_PATH)
    apply_config_io(config, get_schema(schema))  # match the genome's I/O sizes
    with open(genome_path, "rb") as fh:
        genome = pickle.load(fh)
    return neat.nn.FeedForwardNetwork.create(genome, config)


# --------------------------------------------------------------------- #
# parallel workers: build the net once per process (pool initializer) so it
# isn't re-pickled for every run, then each task just runs one seeded episode.

_WORKER: dict = {}


def _init_worker(genome_path: str, schema: int) -> None:
    _WORKER["net"] = load_net(genome_path, schema)
    _WORKER["schema"] = schema


def _run_task(args):
    seed, inaccuracy, reaction_ms, reaction_std, max_ep_s = args
    score = run_episode(_WORKER["net"], seed, inaccuracy, reaction_ms,
                        reaction_std, max_ep_s, _WORKER["schema"])
    return seed, score


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Headlessly evaluate a saved NEAT genome over many runs")
    parser.add_argument("--ai", metavar="GENOME_PKL", nargs="?", const="", default=None,
                        help="genome to evaluate; with no path given, the genome is "
                             "inferred from the schema and human knobs, i.e. "
                             "models/saved/<schema>/<rt_ms>/<rt_std>/<inacc>/<index>_..._best_genome.pkl")
    parser.add_argument("--index", type=int, default=0,
                        help="when --ai infers the genome from the schema and human knobs, "
                             "pick the saved genome with this index (default 0); ignored if "
                             "--ai is given an explicit path")
    parser.add_argument("--runs", type=int, default=100,
                        help="number of headless simulations to run (default 100)")
    parser.add_argument("--seed-base", type=int, default=None,
                        help="first seed; run i uses seed_base + i. Default: a random "
                             "base each invocation (printed below so a run can be repeated)")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1),
                        help="parallel worker processes (default: CPUs - 1)")
    parser.add_argument("--inaccuracy", type=float, default=0.0,
                        help="AI aim error in [0, 1]: gaussian displacement of the "
                             "target distance, scaled by current pick speed")
    parser.add_argument("--reaction_time_ms", type=float, default=0.0,
                        help="AI reaction delay (ms, >= 0) before reprompting on a new bar spawn")
    parser.add_argument("--reaction_time_standard_deviation", type=float, default=0.05,
                        help="relative gaussian jitter of the reaction delay "
                             "(>= 0, typically 0-0.2)")
    parser.add_argument("--max_episode_seconds", type=float, default=MAX_EPISODE_SECONDS,
                        help="hard cap on episode length in sim seconds")
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
    if args.max_episode_seconds <= 0.0:
        parser.error("--max_episode_seconds must be positive")
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    # random base each run unless the user pins one for reproducibility; keep it
    # low enough that seed_base + runs can't overflow a 30-bit seed
    if args.seed_base is None:
        args.seed_base = random.randrange(1 << 30)

    ai_path = args.ai
    if not ai_path:
        # no --ai (or --ai with no path): infer the saved genome for this
        # schema + knobs + index
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
    if not os.path.isfile(ai_path):
        parser.error(f"AI genome not found: {ai_path}")

    net = load_net(ai_path, args.schema)  # validate genome/schema up front

    workers = min(args.workers, args.runs)
    print(f"Evaluating {os.path.relpath(ai_path, ROOT)}")
    print(f"  schema={args.schema}  inaccuracy={args.inaccuracy}  "
          f"reaction_time_ms={args.reaction_time_ms}  "
          f"reaction_time_std={args.reaction_time_standard_deviation}")
    print(f"  {args.runs} runs, seeds {args.seed_base}..{args.seed_base + args.runs - 1}, "
          f"{workers} worker(s)\n")

    tasks = [(args.seed_base + i, args.inaccuracy, args.reaction_time_ms,
              args.reaction_time_standard_deviation, args.max_episode_seconds)
             for i in range(args.runs)]

    scores = []
    if workers > 1:
        # imap keeps results in task (seed) order as they stream back
        pool = multiprocessing.Pool(workers, initializer=_init_worker,
                                    initargs=(ai_path, args.schema))
        try:
            for i, (seed, score) in enumerate(pool.imap(_run_task, tasks)):
                scores.append(score)
                print(f"  run {i + 1:>4}  seed={seed}  score={score}")
        finally:
            pool.close()
            pool.join()
    else:
        for i, (seed, inacc, rt_ms, rt_std, max_ep_s) in enumerate(tasks):
            score = run_episode(net, seed, inacc, rt_ms, rt_std, max_ep_s, args.schema)
            scores.append(score)
            print(f"  run {i + 1:>4}  seed={seed}  score={score}")

    avg = sum(scores) / len(scores)
    fitness = W_AVG * avg + W_WORST * min(scores) + W_BEST * max(scores)
    stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0

    wins = sum(1 for s in scores if s >= WIN_SCORE)

    print()
    print(f"runs      {len(scores)}")
    print(f"wins      {wins}/{len(scores)}  ({100 * wins / len(scores):.1f}%, "
          f"score >= {WIN_SCORE})")
    print(f"mean      {avg:.1f}")
    print(f"median    {statistics.median(scores):.1f}")
    print(f"stdev     {stdev:.1f}")
    print(f"min       {min(scores)}")
    print(f"max       {max(scores)}")
    print(f"fitness   {fitness:.1f}   "
          f"({W_AVG}*avg + {W_WORST}*worst + {W_BEST}*best, training blend)")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
