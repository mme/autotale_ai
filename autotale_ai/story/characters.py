"""
Characters node.
"""

from typing import List
import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import RunnableConfig

from autotale_ai.state import AgentState
from autotale_ai.story.instructions import INSTRUCTIONS

# pylint: disable=line-too-long

class Character(BaseModel):
    """
    Represents a character in the tale.
    """
    name: str = Field(description="The name of the character")
    appearance: str = Field(description="The appearance of the character")
    traits: str = Field(description="The traits of the character")

class Characters(BaseModel):
    """
    Represents the characters in the story.
    """
    characters: List[Character] = Field(description="The characters in the story")

def make_system_message(state: AgentState):
    """
    Make a system message for the characters node.
    """
    content = (
        INSTRUCTIONS +
        "The user and the AI are having a conversation about writing a children's story. " +
        "It's your job to extract the book's main characters from the conversation." +
        "Ideally, it should be around 3 of them, but it can be more if the user wants it." +
        "Make the appearance and traits of the characters as detailed as possible." +
        "The user has provided the following rough idea for the story: " +
        state["rough_idea"] +
        "\n"
    )
    if state["characters"] is not None:
        if state["changed_node"] is None:
            content += (
                "We already have the following characters, but the user wants to refine or change them: " +
                json.dumps(state["characters"])
            )
        else:
            content += (
                f"The user changed {state['changed_node']} and now we have to see if the characters are still appropriate." +
                "The current characters are: " +
                json.dumps(state["characters"]) +
                "\n" +
                "Make sure to keep all fitting characters intact."
            )

    print(content)

    return SystemMessage(
        content=content
    )

def characters_node(state: AgentState, config: RunnableConfig):
    """
    The characters node is responsible for generating the characters in the story.
    """
    model = ChatOpenAI(model="gpt-4o").with_structured_output(Characters)
    response = model.invoke([
        *state["messages"],
        make_system_message(state)
    ], config)
    response_dict = response.dict()
    return {
        "messages": AIMessage(content=json.dumps(response_dict)),
        "characters": response_dict["characters"],
        "changed_node": state["changed_node"] or "characters",
    }
