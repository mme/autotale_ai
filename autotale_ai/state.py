"""
This is the state definition for the autotale AI.
It defines the state of the agent and the state of the conversation.
"""

from typing import List, TypedDict
from langgraph.graph import MessagesState
from langchain_core.pydantic_v1 import BaseModel, Field

class Character(TypedDict):
    """
    Represents a character in the tale.
    """
    name: str
    appearance: str
    traits: str


class Page(BaseModel):
    """
    Represents a page in the children's story. Keep it simple, 3-4 sentences per page.
    """
    content: str = Field(..., description="A single page in the story")
    image_url: str


class AgentState(MessagesState):
    """
    This is the state of the agent.
    It is a subclass of the MessagesState class from langgraph.
    """
    outline: str
    characters: List[Character]
    story: List[Page]
