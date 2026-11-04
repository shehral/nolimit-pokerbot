"""Import the single-file bundle under THIS interpreter and push synthetic decisions through every version.

    <python> bot/tests/pyver_check.py [bot/dist/nolimit_bot.py] [--json out.json]
Prints the interpreter version, import + warm-up time (wall and process CPU; the import builds nothing heavy: the
tables are literals, the warm-up plays 300 synthetic decisions per version), construction time of Bot(seat, config)
for 8 and 10 seats under TOP3 and TOP4 payouts (first construction of a payout builds and times its ICM pack;
repeats hit the cache), and counts of illegal results and caught exceptions over 9,000 synthetic decisions
(STRICT mode is on, so an exception would propagate and abort the run). Standard library only; no project imports.
Run it in a FRESH process per interpreter (the import time is only meaningful the first time).
"""
import importlib.util
import json
import os
import random
import sys
import time


def _construct_ms(m, n_seats, payouts, reps=20):
    cfg = {"n_seats": n_seats, "payouts": payouts, "seed": 3, "start_stack": 2000}
    t = time.perf_counter()
    m.Bot(0, dict(cfg))
    first = (time.perf_counter() - t) * 1e3
    ts = []
    for s in range(reps):
        t = time.perf_counter()
        m.Bot(s % n_seats, dict(cfg))
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return {"first_ms": round(first, 3), "median_ms": round(ts[len(ts) // 2], 3), "max_ms": round(ts[-1], 3)}


def main():
    args = [a for a in sys.argv[1:]]
    out_json = ""
    if "--json" in args:
        i = args.index("--json")
        out_json = args[i + 1]
        del args[i:i + 2]
    path = args[0] if args else os.path.join(os.path.dirname(__file__), "..", "dist", "nolimit_bot.py")
    os.environ["NOLIMIT_BOT_STRICT"] = "1"
    spec = importlib.util.spec_from_file_location("nolimit_bundle_check", path)
    m = importlib.util.module_from_spec(spec)
    t = time.perf_counter()
    c = time.process_time()
    spec.loader.exec_module(m)
    imp_ms = (time.perf_counter() - t) * 1e3
    imp_cpu_ms = (time.process_time() - c) * 1e3
    ad = sys.modules["nolimit_bot_adapter"]
    construct = {}
    for n in (8, 10):
        for name, pays in (("TOP3", [50, 30, 20]), ("TOP4", [40, 30, 20, 10])):
            construct[f"{n}seats_{name}"] = _construct_ms(m, n, pays + [0] * (n - len(pays)))
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
    worst = max(v["max_ms"] for v in construct.values())
    firsts = max(v["first_ms"] for v in construct.values())
    print("%s python %s: import+warm-up %.0f ms (cpu %.0f ms), warmup_info %s, construct 8/10 seats: first <= %.2f ms, "
          "repeat max %.2f ms, 9,000 decisions, illegal %d, errors %d"
          % (time.strftime("%Y-%m-%d %H:%M:%S %Z"), sys.version.split()[0], imp_ms, imp_cpu_ms, m.WARMUP_INFO,
             firsts, worst, bad, errors))
    if out_json:
        with open(out_json, "w") as f:
            json.dump({"python": sys.version.split()[0], "import_ms": imp_ms, "import_cpu_ms": imp_cpu_ms,
                       "warmup": dict(m.WARMUP_INFO), "construct": construct, "illegal": bad, "errors": errors,
                       "loadavg": os.getloadavg(), "stamp": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, f, indent=1)
    return 0 if bad == 0 and errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
