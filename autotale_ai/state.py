"""
This is the state definition for the autotale AI.
It defines the state of the agent and the state of the conversation.
"""

import operator
from typing import List, TypedDict
from typing_extensions import Annotated
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

class CharacterImage(TypedDict):
    """
    Represents an image of a character in the tale.
    """
    description: str
    image_url: str

class AgentState(MessagesState):
    """
    This is the state of the agent.
    It is a subclass of the MessagesState class from langgraph.
    """
    outline: str
    characters: List[Character]
    story: List[Page]
    page_images: Annotated[list, operator.add]
    # story: dict
    # character_images: Annotated[list, operator.add]
    # page_images: Annotated[list, operator.add]
    # changed_node: str


class PageImageGenerationState(TypedDict):
    """
    This is the state of the page image generation.
    """
    page: Page
    idx: int
    messages: list
    characters: List[Character]

class CharacterImageGenerationState(TypedDict):
    """
    This is the state of the character image generation.
    """
    character: Character
    image_url: str
    messages: list
    should_generate: bool
    image_description: str
