"""Safe-default bot for the variant rehearsal (the variant checklist).

Betting: the kit's example agent (sim/kit/example_agent.py), unchanged. Any non-betting request (here the toss
"discard" request) goes through variant.defaults.respond(view), which keeps the best two cards by static score and
answers with the tossed index. Every path is wrapped: an exception in the variant layer returns the layer's own
default; an exception in betting returns check, else fold (never raises).

Spec strings for the runner: "variant.verify.safe_bot:SafeDefaultBot".
"""
from sim.kit.cards import CARD_INT
from sim.kit.example_agent import ExampleAgent

from variant.defaults import respond

# kit request name -> (layer kind, response format, k, kit answer builder)
REQUESTS = {
    "discard": ("toss", "index", 1, lambda r: {"action": "discard", "index": r}),
}


def request_view(state, kind, fmt, k):
    """Kit state -> the layer's request view (card ints in kit numbering)."""
    return {"kind": kind, "response": fmt, "k": k,
            "hole": [CARD_INT[c] for c in state["hole"]],
            "board": [CARD_INT[c] for c in state.get("board") or ()],
            "options": list(state["discard_options"]) if state.get("discard_options") is not None else None,
            "pot": state.get("pot"), "stack": (state.get("stacks") or [None] * 9)[state.get("seat", 0)]}


def _first_option(state):
    opts = state.get("discard_options") or [0]
    return {"action": "discard", "index": opts[-1]}


def _check_or_fold(state):
    return {"action": "check"} if "check" in (state.get("legal_actions") or ()) else {"action": "fold"}


class SafeDefaultBot(ExampleAgent):
    def act(self, state):
        req = state.get("request", "act")
        if req != "act":
            try:
                kind, fmt, k, build = REQUESTS.get(req, ("other", "option", None, None))
                if build is None:          # unknown request: first legal option the state offers
                    return _first_option(state)
                r = respond(request_view(state, kind, fmt, k))
                if not isinstance(r, int) or r not in (state.get("discard_options") or (0, 1, 2)):
                    return _first_option(state)
                return build(r)
            except Exception:  # noqa: BLE001
                return _first_option(state)
        try:
            return ExampleAgent.act(self, state)
        except Exception:  # noqa: BLE001
            return _check_or_fold(state)


Agent = SafeDefaultBot
