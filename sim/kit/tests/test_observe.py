"""observe(): public events only (plus your own private deal), untimed, default no-op skipped."""
from sim.kit.engine import Table
from sim.kit.fixtures import CheckCallAgent, RecorderAgent


def recorder_table(n=6, max_hands=30, seed="obs"):
    sinks = {}

    def make(seat, cfg):
        sinks[seat] = []
        return RecorderAgent(seat, cfg, sinks[seat])

    res = Table([make] * n, seed=seed, max_hands=max_hands).play()
    return res, sinks


def test_private_deal_only_to_its_owner_and_no_hole_cards_before_showdown():
    res, sinks = recorder_table()
    for seat, evs in sinks.items():
        deals = [e for e in evs if e["type"] == "deal"]
        assert deals and all(e["seat"] == seat for e in deals)
        for e in evs:
            if e["type"] not in ("deal", "showdown"):
                assert "hole" not in e and "shown" not in e


def test_event_sequence_and_contents():
    res, sinks = recorder_table(n=3, max_hands=5)
    evs = sinks[0]
    types = [e["type"] for e in evs]
    assert types[0] == "hand_start" and types[1] == "deal" and types[-1] == "table_end"
    hs = evs[0]
    assert {"hand_no", "level", "sb", "bb", "ante", "button", "sb_seat", "bb_seat", "stacks", "alive",
            "posts"} <= set(hs)
    acts = [e for e in evs if e["type"] == "action"]
    assert acts and {"seat", "action", "amount", "street", "pot", "to_call", "all_in"} <= set(acts[0])
    ends = [e for e in evs if e["type"] == "hand_end"]
    assert len(ends) == res.hands_played
    shows = [e for e in evs if e["type"] == "showdown"]
    assert shows and all(len(e["board"]) == 5 for e in shows)
    for e in evs:                          # values are immutable so one bot cannot corrupt another's events
        for v in e.values():
            assert not isinstance(v, (list, dict, set))


def test_events_are_copies_per_agent():
    res, sinks = recorder_table(n=3, max_hands=3)
    a = [e for e in sinks[0] if e["type"] == "action"][0]
    b = [e for e in sinks[1] if e["type"] == "action"][0]
    assert a == b and a is not b


def test_default_observe_is_skipped_and_untimed_observe_is_measured():
    res = Table([CheckCallAgent] * 4, seed="skip", max_hands=10).play()
    assert all(st.observe_calls == 0 for st in res.stats)
    res, sinks = recorder_table(n=4, max_hands=10)
    assert all(st.observe_calls == len(sinks[s]) for s, st in enumerate(res.stats))
    assert all(st.observe_ns > 0 for st in res.stats)
