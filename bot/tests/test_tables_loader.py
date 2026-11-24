"""Table swapping (bot/tables_loader.py): every bad candidate falls back to the built-in NLHE tables with a reason,
good rebuilt tables are swapped in (brain globals, class space, fits rule), and the one-line log names the tables.
The swap itself is tested in subprocesses because brain.py picks its tables once, at import."""
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bot import tables_loader as TL      # noqa: E402

VOUT = os.path.join(ROOT, "bot", "tests", "data", "tables")   # fixtures built by variant/verify/run_builds.py
SHORT = os.path.join(VOUT, "verify_shortdeck.py")
TOSS90 = os.path.join(VOUT, "verify_toss.py")        # the verifier's 90 s toss build: stderr 0.024, gate must fail


def _copy_edit(tmp_path, src, name, old, new):
    with open(src) as f:
        s = f.read()
    assert old in s
    p = tmp_path / name
    p.write_text(s.replace(old, new, 1))
    return str(p)


def test_missing_path_falls_back(tmp_path):
    a = TL.load(str(tmp_path / "nope.py"), quiet=True)
    assert a.kind == "builtin" and "file not found" in a.reasons
    assert "REJECTED" in TL.INFO["line"] and "built-in NLHE" in TL.INFO["line"]


def test_broken_module_falls_back(tmp_path):
    p = tmp_path / "broken.py"
    p.write_text("EQ_LABELS = 'AA KK'\ndef (:\n")
    a = TL.load(str(p), quiet=True)
    assert a.kind == "builtin" and any("SyntaxError" in r for r in a.reasons)


def test_sanity_failure_falls_back(tmp_path):
    with open(SHORT) as f:
        s = f.read()
    p = tmp_path / "bad_sanity.py"
    p.write_text(s + "\nSANITY = {'ok': False, 'failed': ['spread_ok']}\n")
    a = TL.load(str(p), quiet=True)
    assert a.kind == "builtin" and any("SANITY" in r for r in a.reasons)


def test_noisy_toss_build_is_gated_on_stderr():
    a = TL.load(TOSS90, quiet=True)
    assert a.kind == "builtin"
    assert any(r.startswith("stderr a 0.02") for r in a.reasons), a.reasons


def test_checkpoint_refused_unless_allowed(tmp_path, monkeypatch):
    p = _copy_edit(tmp_path, SHORT, "ckpt.py", "'final': True", "'final': False")
    a = TL.load(p, quiet=True)
    assert a.kind == "builtin" and any("checkpoint" in r for r in a.reasons), a.reasons
    monkeypatch.setenv(TL.ENV_ALLOW_CKPT, "1")
    assert TL.load(p, quiet=True).kind == "variant"


def test_labels_of_another_game_are_rejected(tmp_path):
    # shortdeck tables that claim (via SPEC_PARAMS) to be 52-card hold'em: label sets differ -> rejected
    with open(SHORT) as f:
        s = f.read()
    p = tmp_path / "liar.py"
    p.write_text(s + "\nSPEC_PARAMS = {'name': 'nlhe', 'hole_dealt': 2}\n")
    a = TL.load(str(p), quiet=True)
    assert a.kind == "builtin" and any("label sets differ" in r for r in a.reasons), a.reasons


def test_good_tables_are_accepted():
    a = TL.load(SHORT, quiet=True)
    assert a.kind == "variant" and a.hs.n_classes == 81 and a.use_fits
    assert "81 classes" in TL.INFO["line"]


def test_resolve_path_env_and_default(monkeypatch, tmp_path):
    monkeypatch.setenv(TL.ENV_TABLES, "builtin")
    assert TL.resolve_path()[0] is None
    monkeypatch.setenv(TL.ENV_TABLES, SHORT)
    assert TL.resolve_path() == (SHORT, TL.ENV_TABLES)
    monkeypatch.delenv(TL.ENV_TABLES)
    p, how = TL.resolve_path()
    assert (p is None) == (not os.path.exists(os.path.join(ROOT, "bot", TL.DEFAULT_FILE)))


def _brain_in_subprocess(env_tables, extra=""):
    code = ("import json, sys; sys.path.insert(0, %r)\n" % ROOT +
            "from bot import brain, shell\n"
            "print(json.dumps({'seam': brain._SEAM is not None, 'info': brain.TABLES_INFO,"
            " 'n': len(brain.LABELS), 'jam': list(brain.PARAMS['JAM_FIT']), 'warm': shell.WARMUP_INFO}))\n" + extra)
    env = dict(os.environ)
    env.pop(TL.ENV_QUIET, None)
    if env_tables is None:
        env.pop("NOLIMIT_TABLES", None)
    else:
        env["NOLIMIT_TABLES"] = env_tables
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=180)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[0]), r.stderr


def test_brain_swaps_in_good_tables_and_logs_one_line():
    out, err = _brain_in_subprocess(SHORT)
    assert out["seam"] and out["n"] == 81 and "warmup_ms" in out["warm"]
    assert out["jam"][0] != 3.29                      # refit fits from the shortdeck solve
    lines = [ln for ln in err.splitlines() if ln.startswith("[nolimit_bot] tables:")]
    assert len(lines) == 1 and "verify_shortdeck.py" in lines[0]


def test_brain_falls_back_to_builtin_on_rejected_tables():
    out, err = _brain_in_subprocess(TOSS90)
    assert not out["seam"] and out["n"] == 169 and out["jam"][0] == 3.29
    assert "REJECTED" in err


def test_brain_builtin_when_forced():
    out, err = _brain_in_subprocess("builtin")
    assert not out["seam"] and out["n"] == 169 and "forces the built-in" in err


def test_ante_tables_keep_nlhe_fits(tmp_path):
    # tables re-solved WITH antes must not replace the fits: the brain prices antes as dead money already (P11)
    src = os.path.join(VOUT, "verify_antes.py")
    a = TL.load(src, quiet=True)
    assert a.kind == "variant" and a.spec.ante_bb > 0 and not a.use_fits


def test_bundle_picks_up_tables_next_to_it(tmp_path):
    dist = os.path.join(ROOT, "bot", "dist", "nolimit_bot.py")
    shutil.copy(dist, tmp_path / "nolimit_bot.py")
    shutil.copy(SHORT, tmp_path / TL.DEFAULT_FILE)
    code = ("import importlib.util as u\n"
            "s = u.spec_from_file_location('b', 'nolimit_bot.py'); m = u.module_from_spec(s); s.loader.exec_module(m)\n"
            "print(m.brain._SEAM is not None, len(m.brain.LABELS))\n")
    env = dict(os.environ)
    env.pop("NOLIMIT_TABLES", None)
    r = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True, env=env,
                       timeout=180)
    assert r.returncode == 0, r.stderr
    assert r.stdout.split() == ["True", "81"], (r.stdout, r.stderr)
