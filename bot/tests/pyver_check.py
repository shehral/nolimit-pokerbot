"""Import the single-file bundle under THIS interpreter and push synthetic decisions through every version.

    <python> bot/tests/pyver_check.py bot/dist/nolimit_bot.py
Prints the interpreter version, import + warm-up time, and counts of illegal results and caught exceptions
(STRICT mode is on, so an exception would propagate and abort the run). Standard library only; no project imports.
"""
import importlib.util
import os
import random
import sys
import time


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "dist",
                                                                "nolimit_bot.py")
    os.environ["NOLIMIT_BOT_STRICT"] = "1"
    spec = importlib.util.spec_from_file_location("nolimit_bundle_check", path)
    m = importlib.util.module_from_spec(spec)
    t = time.perf_counter()
    spec.loader.exec_module(m)
    imp_ms = (time.perf_counter() - t) * 1e3
    ad = sys.modules["nolimit_bot_adapter"]
    states = m._synthetic_states(random.Random(7), 3000)
    bad = errors = 0
    for ver in (0, 1, 2):
        bot = getattr(m, "BotV%d" % ver)(0, {"n_seats": 8, "payouts": [50, 30, 20, 0, 0, 0, 0, 0], "seed": 3})
        for st in states:
            bot.seat = st["seat"]
            bot._brain.seat = st["seat"]
            a = bot.act(st)
            if not ad.is_legal(a, st):
                bad += 1
        errors += bot.errors
    print("python %s: import+warm-up %.0f ms, warmup_info %s, 9,000 decisions, illegal %d, errors %d"
          % (sys.version.split()[0], imp_ms, m.WARMUP_INFO, bad, errors))
    return 0 if bad == 0 and errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
