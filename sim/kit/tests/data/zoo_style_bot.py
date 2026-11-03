"""A bot written the way zoo files will be: imports the kit base class, defines one subclass."""
from sim.kit.agent import Agent


class NitStyleAgent(Agent):
    def act(self, state):
        return {"action": "check"} if "check" in state["legal_actions"] else {"action": "fold"}
