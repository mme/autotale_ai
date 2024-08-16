"""
This is the state definition for the autotale AI.
It defines the state of the agent and the state of the conversation.
"""

from typing import List, TypedDict
from langgraph.graph import MessagesState


class Character(TypedDict):
    """
    Represents a character in the tale.
    """
    name: str
    appearance: str
    traits: str

class AgentState(MessagesState):
    """
    This is the state of the agent.
    It is a subclass of the MessagesState class from langgraph.
    """
    rough_idea: str
    characters: List[Character]
    story: str

    changed_node: str
