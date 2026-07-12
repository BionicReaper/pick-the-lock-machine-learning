"""Entry point: python play.py [--ai [GENOME_PKL]] [--index N] [--mute] [--seed N]
[--inaccuracy F] [--reaction_time_ms MS] [--reaction_time_standard_deviation F]
[--schema N]

--ai with no path infers the genome from the schema and human knobs, picking the
saved genome with --index (default 0) under
models/saved/<schema>/<rt_ms>/<rt_std>/<inacc>/<index>_best_genome_<score>.pkl."""

from pickthelock.game import main

if __name__ == "__main__":
    main()
