"""NEAT training for Pick the Lock.

Each genome is evaluated on EVAL_RUNS headless simulations (same seed set
for every genome within a generation, rotating across generations) and its
fitness combines the runs:

    fitness = 0.5 * average + 0.25 * worst + 0.25 * best

The model is (re)prompted at episode start, whenever a new bar spawns, and
whenever the scheduled target distance is reached. Each prompt produces
(target distance, boost-hold fraction, click?) which is handed to the
cancellable ScheduledClickController.

Human-imperfection knobs: --inaccuracy adds gaussian aim error to each
prompted target distance (scaled by current pick speed); --reaction_time_ms
(jittered by --reaction_time_standard_deviation) delays the reprompt that
follows a new bar spawn by that many milliseconds worth of ticks, and until
that reaction lands the new bar is hidden from build_inputs entirely — so
prompts fired in between (a click's target-reached reprompt, idle
reprompts) cannot plan around a bar the "human" hasn't noticed yet. When a
knob is 0 its random draws are skipped entirely. --max_episode_seconds caps
episode length; raising it makes long-surviving genomes cost more to evaluate.

Usage:
    .venv\\Scripts\\python.exe train_neat.py --generations 100
    .venv\\Scripts\\python.exe train_neat.py --resume models/neat-checkpoint-42
    .venv\\Scripts\\python.exe train_neat.py --smoke        (tiny sanity run)

Outputs (models/):
    best_genome.pkl        best genome seen so far (updated on improvement)
    winner_genome.pkl      best genome of the final generation
    fitness_history.csv    per-generation best/mean fitness
    neat-checkpoint-N      resumable population checkpoints
"""

from __future__ import annotations

import argparse
import csv
import itertools
import multiprocessing
import os
import pickle
import random
import time

import neat

from pickthelock.config import DEFAULT_STAGE, DEFAULT_TUNING
from pickthelock.controller import ScheduledClickController
from pickthelock.observations import build_inputs, decode_outputs
from pickthelock.sim import LockpickingSim, EV_BAR_SPAWNED, EV_TARGET_REACHED

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
CONFIG_PATH = os.path.join(ROOT, "neat_config.txt")

EVAL_RUNS = 15               # headless simulations per genome (5-10)
W_AVG, W_WORST, W_BEST = 0.60, 0.05, 0.35
MAX_EPISODE_SECONDS = 600.0  # default --max_episode_seconds (timer bonuses extend games)


# --------------------------------------------------------------------- #
# episode

def run_episode(net, seed: int, inaccuracy: float = 0.0,
                reaction_time_ms: float = 0.0, reaction_time_std: float = 0.05,
                max_episode_seconds: float = MAX_EPISODE_SECONDS) -> int:
    sim = LockpickingSim(DEFAULT_STAGE, DEFAULT_TUNING, seed=seed)
    ctrl = ScheduledClickController(sim)
    gauss = random.gauss

    def prompt():
        dist, boost_frac, do_click = decode_outputs(net.activate(build_inputs(sim)), sim)
        if inaccuracy > 0.0:
            # human aim error: the faster the pick moves, the sloppier the aim
            dist += sim.current_speed * inaccuracy * gauss(0.0, 1.0)
        ctrl.schedule(dist, boost_frac, do_click)

    prompt()
    tick_rate = DEFAULT_TUNING.tick_rate
    max_ticks = int(max_episode_seconds * tick_rate)
    has_reaction = reaction_time_ms > 0.0
    reaction_base_ticks = (reaction_time_ms / 1000.0) * tick_rate
    scheduled_reactions: set[int] = set()
    unreacted_bars: list[tuple] = []   # (reaction tick, Bar) hidden until then
    for tick in range(max_ticks):
        should_prompt = False
        events = ctrl.step()
        if sim.game_over:
            break
        for ev in events:
            if ev[0] == EV_BAR_SPAWNED:
                # unforeseeable stimulus: the reprompt lands only after a
                # human-like (jittered) reaction delay
                if not has_reaction:
                    should_prompt = True   # 0 delay would fire this tick anyway
                else:
                    if reaction_time_std == 0.0:
                        due = tick + round(reaction_base_ticks)
                    else:
                        delay = reaction_base_ticks * (1.0 + gauss(0.0, 1.0) * reaction_time_std)
                        due = tick + max(0, round(delay))
                    scheduled_reactions.add(due)
                    if due > tick:
                        # not noticed yet: build_inputs skips the bar, so
                        # prompts fired meanwhile (e.g. right after a click)
                        # can't plan around it
                        ev[1].perceived = False
                        unreacted_bars.append((due, ev[1]))
            elif ev[0] == EV_TARGET_REACHED:
                should_prompt = True
        if unreacted_bars:
            # reaction lands: bar becomes visible on the same tick its
            # scheduled reprompt fires (harmless if it already despawned)
            for due, bar in unreacted_bars:
                if due <= tick:
                    bar.perceived = True
            unreacted_bars = [p for p in unreacted_bars if p[0] > tick]
        if scheduled_reactions and tick in scheduled_reactions:
            scheduled_reactions.discard(tick)
            should_prompt = True
        if not ctrl.active:
            should_prompt = True
        if should_prompt:
            prompt()
    return sim.score


def eval_genome(genome, config, seeds, inaccuracy: float = 0.0,
                reaction_time_ms: float = 0.0, reaction_time_std: float = 0.05,
                max_episode_seconds: float = MAX_EPISODE_SECONDS) -> float:
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    scores = [run_episode(net, s, inaccuracy, reaction_time_ms, reaction_time_std,
                          max_episode_seconds)
              for s in seeds]
    return (W_AVG * (sum(scores) / len(scores))
            + W_WORST * min(scores)
            + W_BEST * max(scores))


# top-level so Windows 'spawn' processes can pickle it
def _eval_task(args):
    genome_id, genome, config, seeds, inaccuracy, reaction_ms, reaction_std, max_ep_s = args
    return genome_id, eval_genome(genome, config, seeds, inaccuracy, reaction_ms,
                                  reaction_std, max_ep_s)


def _atomic_pickle(path: str, obj) -> None:
    """Write-then-rename so a Ctrl+C mid-dump can't truncate the file."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)


# --------------------------------------------------------------------- #
# training driver

class Trainer:
    def __init__(self, config, workers: int, runs: int, seed_base: int,
                 inaccuracy: float = 0.0, reaction_time_ms: float = 0.0,
                 reaction_time_std: float = 0.05,
                 max_episode_seconds: float = MAX_EPISODE_SECONDS):
        self.config = config
        self.workers = workers
        self.runs = runs
        self.seed_base = seed_base
        self.inaccuracy = inaccuracy
        self.reaction_time_ms = reaction_time_ms
        self.reaction_time_std = reaction_time_std
        self.max_episode_seconds = max_episode_seconds
        self.generation = 0
        self.best_fitness = float("-inf")
        self.history: list[tuple[int, float, float]] = []
        self.pool = None
        if workers > 1:
            self.pool = multiprocessing.Pool(workers)

    def eval_genomes(self, genomes, config):
        # same seeds for every genome within a generation, new set each gen
        seeds = [self.seed_base + self.generation * 7919 + i for i in range(self.runs)]
        tasks = [(gid, g, config, seeds, self.inaccuracy, self.reaction_time_ms,
                  self.reaction_time_std, self.max_episode_seconds) for gid, g in genomes]
        if self.pool is not None:
            results = dict(self.pool.map(_eval_task, tasks))
        else:
            results = dict(_eval_task(t) for t in tasks)
        total = 0.0
        best = float("-inf")
        best_genome = None
        for gid, genome in genomes:
            genome.fitness = results[gid]
            total += genome.fitness
            if genome.fitness > best:
                best, best_genome = genome.fitness, genome
        mean = total / max(1, len(genomes))
        self.history.append((self.generation, best, mean))
        if best > self.best_fitness and best_genome is not None:
            self.best_fitness = best
            _atomic_pickle(os.path.join(MODELS_DIR, "best_genome.pkl"), best_genome)
            print(f"  ** new best fitness {best:.0f} (gen {self.generation}) "
                  f"-> models/best_genome.pkl")
        self.generation += 1

    def save_history(self):
        path = os.path.join(MODELS_DIR, "fitness_history.csv")
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["generation", "best_fitness", "mean_fitness"])
            w.writerows(self.history)

    def close(self, terminate: bool = False):
        if self.pool is not None:
            if terminate:
                self.pool.terminate()   # workers may be mid-task after Ctrl+C
            else:
                self.pool.close()
            self.pool.join()


def main():
    parser = argparse.ArgumentParser(description="Train a NEAT net to play Pick the Lock")
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--runs", type=int, default=EVAL_RUNS,
                        help="simulations per genome evaluation (5-10)")
    parser.add_argument("--workers", type=int,
                        default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--pop", type=int, default=None, help="override population size")
    parser.add_argument("--seed-base", type=int, default=1234)
    parser.add_argument("--inaccuracy", type=float, default=0.0,
                        help="aim error in [0, 1]: gaussian displacement of the target "
                             "distance, scaled by current pick speed")
    parser.add_argument("--reaction_time_ms", type=float, default=0.0,
                        help="reaction delay (ms, >= 0) before reprompting on a new bar spawn")
    parser.add_argument("--reaction_time_standard_deviation", type=float, default=0.05,
                        help="relative gaussian jitter of the reaction delay "
                             "(>= 0, typically 0-0.2)")
    parser.add_argument("--max_episode_seconds", type=float, default=MAX_EPISODE_SECONDS,
                        help="hard cap on episode length in sim seconds (timer bonuses can "
                             "extend games this far; long-surviving genomes cost "
                             "proportionally more to evaluate)")
    parser.add_argument("--resume", default=None, help="path to a neat-checkpoint-N file")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny run (pop 16, 2 gens, 3 sims, 1 worker) to verify the pipeline")
    args = parser.parse_args()
    if not 0.0 <= args.inaccuracy <= 1.0:
        parser.error("--inaccuracy must be between 0 and 1")
    if args.reaction_time_ms < 0.0:
        parser.error("--reaction_time_ms must be non-negative")
    if args.reaction_time_standard_deviation < 0.0:
        parser.error("--reaction_time_standard_deviation must be non-negative")
    if args.max_episode_seconds <= 0.0:
        parser.error("--max_episode_seconds must be positive")

    os.makedirs(MODELS_DIR, exist_ok=True)

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation, CONFIG_PATH)
    if args.smoke:
        args.generations, args.runs, args.workers, args.pop = 2, 3, 1, 16
    if args.pop:
        config.pop_size = args.pop

    if args.resume:
        pop = neat.Checkpointer.restore_checkpoint(args.resume)
        pop.config = config
        # neat-python bug: the new-node-id counter lives on the config, not
        # the checkpoint, so after restore it's None and gets seeded from a
        # single genome's max node id -- lower than ids already present in
        # other restored genomes, which later collides (assert new_id not in
        # node_dict). Reseed it above the max node id across the whole pop.
        max_node_id = max((k for g in pop.population.values() for k in g.nodes),
                          default=config.genome_config.num_outputs)
        config.genome_config.node_indexer = itertools.count(max_node_id + 1)
    else:
        pop = neat.Population(config)

    trainer = Trainer(config, args.workers, args.runs, args.seed_base,
                      args.inaccuracy, args.reaction_time_ms,
                      args.reaction_time_standard_deviation,
                      args.max_episode_seconds)
    if args.resume:
        # keep the per-generation seed rotation moving forward after a resume
        trainer.generation = pop.generation

    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(
        generation_interval=10, time_interval_seconds=None,
        filename_prefix=os.path.join(MODELS_DIR, "neat-checkpoint-")))

    t0 = time.time()
    winner = None
    try:
        winner = pop.run(trainer.eval_genomes, args.generations)
    except KeyboardInterrupt:
        gen = trainer.generation
        ckpt = neat.Checkpointer(
            filename_prefix=os.path.join(MODELS_DIR, "neat-checkpoint-"))
        ckpt.save_checkpoint(config, pop.population, pop.species, gen)
        print(f"\nInterrupted at generation {gen}. Best so far is safe in "
              f"models/best_genome.pkl (saved on every improvement).")
        print(f"Resume with:  train_neat.py --resume models/neat-checkpoint-{gen}")
    finally:
        trainer.save_history()
        trainer.close(terminate=winner is None)

    if winner is not None:
        _atomic_pickle(os.path.join(MODELS_DIR, "winner_genome.pkl"), winner)
    print(f"\nDone in {time.time() - t0:.0f}s. Best fitness {trainer.best_fitness:.0f}.")
    print("Watch it play:  .venv\\Scripts\\python.exe play.py --ai models/best_genome.pkl")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
