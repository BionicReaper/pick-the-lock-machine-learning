"""Pick the Lock — standalone recreation of the Dota 2 Dark Carnival minigame.

Game logic lives in sim.py and is shared verbatim between the playable
pygame client (game.py) and the headless NEAT trainer (train_neat.py).
"""

from .config import StageConfig, SimTuning, DEFAULT_STAGE, DEFAULT_TUNING
from .sim import LockpickingSim, EV_BAR_SPAWNED, EV_BAR_EXPIRED, EV_PICKED, EV_MISSED, EV_GAME_OVER, EV_TARGET_REACHED, EV_TIMER_BONUS
from .controller import ScheduledClickController
from .observations import build_inputs, NUM_INPUTS, NUM_OUTPUTS, decode_outputs
