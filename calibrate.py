"""Entry point: python calibrate.py [--mute] [--scale F] [--seed N]
[--reaction-samples N] [--inaccuracy-samples N]

Runs the two calibration tests (reaction, aim accuracy), fits the human
imperfection knobs by gaussian MLE and prints the matching train_neat.py
command. See pickthelock/calibration.py for the test design."""

from pickthelock.calibration import main

if __name__ == "__main__":
    main()
