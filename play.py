"""Entry point: python play.py [--ai models/best_genome.pkl] [--mute] [--seed N]
[--inaccuracy F] [--reaction_time_ms MS] [--reaction_time_standard_deviation F]"""

from pickthelock.game import main

if __name__ == "__main__":
    main()
