"""Sanity checks for the sim + controller, no pygame/neat needed.

1. Determinism: same seed => identical outcome.
2. A simple heuristic bot (aim at the nearest bar, boost when far) must
   score well above zero — proves bars spawn, clicks land, timer bonuses work.
3. A never-clicking bot ends with score 0 at ~30s.

Run:  .venv\\Scripts\\python.exe smoke_test.py
"""

from __future__ import annotations

from pickthelock.config import DEFAULT_STAGE, DEFAULT_TUNING
from pickthelock.controller import ScheduledClickController
from pickthelock.observations import build_inputs, DEFAULT_INPUT_KEYS, NUM_INPUTS
from pickthelock.schemas import SCHEMAS
from pickthelock.sim import LockpickingSim, EV_BAR_SPAWNED, EV_TARGET_REACHED

MAX_TICKS = 600 * DEFAULT_TUNING.tick_rate


def heuristic_episode(seed: int, verbose: bool = False):
    sim = LockpickingSim(DEFAULT_STAGE, DEFAULT_TUNING, seed=seed)
    ctrl = ScheduledClickController(sim)

    def prompt():
        ordered = sim.bars_by_forward_distance()
        if ordered:
            fwd, bar = ordered[0]
            # prefer blue bars if one is close behind the nearest
            for f, b in ordered[:3]:
                if b.is_blue:
                    fwd, bar = f, b
                    break
            boost = 0.8 if fwd > 90 else 0.0
            ctrl.schedule(max(1.0, fwd), boost, True)
        else:
            ctrl.schedule(30.0, 0.0, False)  # idle sweep, reprompt soon

    prompt()
    for _ in range(MAX_TICKS):
        events = ctrl.step()
        if sim.game_over:
            break
        if any(e[0] in (EV_BAR_SPAWNED, EV_TARGET_REACHED) for e in events) or not ctrl.active:
            prompt()
    if verbose:
        print(f"  seed={seed}: score={sim.score} picks={sim.unlock_count} "
              f"game_len={sim.t:.1f}s")
    return sim.score, sim.unlock_count, round(sim.t, 3)


def passive_episode(seed: int):
    sim = LockpickingSim(DEFAULT_STAGE, DEFAULT_TUNING, seed=seed)
    ticks = 0
    while not sim.game_over and ticks < MAX_TICKS:
        sim.tick()
        ticks += 1
    return sim.score, round(sim.t, 2)


def main():
    print("== observation vector ==")
    sim = LockpickingSim(seed=42)
    for _ in range(120):
        sim.tick()
    obs = build_inputs(sim, DEFAULT_INPUT_KEYS)
    assert len(obs) == NUM_INPUTS, f"expected {NUM_INPUTS} inputs, got {len(obs)}"
    print(f"  {len(obs)} inputs OK: {[round(v, 3) for v in obs]}")

    print("== schema registry ==")
    for sid, schema in sorted(SCHEMAS.items()):
        n = len(build_inputs(sim, schema.input_dictionary))
        assert n == schema.num_inputs, (
            f"schema {sid}: build_inputs gave {n} values, num_inputs={schema.num_inputs}")
        print(f"  schema {sid}: {n} inputs / {schema.num_outputs} outputs OK")

    print("== determinism ==")
    a = heuristic_episode(7)
    b = heuristic_episode(7)
    assert a == b, f"non-deterministic: {a} vs {b}"
    print(f"  identical runs OK {a}")

    print("== passive bot (never clicks) ==")
    score, t = passive_episode(3)
    assert score == 0 and abs(t - 30.0) < 0.5, (score, t)
    print(f"  score={score}, ended at t={t}s OK")

    print("== heuristic bot, 5 seeds ==")
    scores = []
    for seed in range(5):
        s, picks, t = heuristic_episode(seed, verbose=True)
        scores.append(s)
    assert max(scores) > 0, "heuristic bot never scored — sim is broken"
    print(f"  avg={sum(scores)/len(scores):.0f} best={max(scores)} worst={min(scores)}")
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
