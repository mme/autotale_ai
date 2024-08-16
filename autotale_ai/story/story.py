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

class Page(BaseModel):
    """
    Represents a page in the children's story. Keep it simple, 3-4 sentences per page.
    """
    content: str = Field(description="The content of the page")

class Story(BaseModel):
    """
    This is a list of pages of the book of the children's story.
    """
    pages: List[Page] = Field(description="The pages of the book of the children's story")

def make_system_message(state: AgentState):
    """
    Make a system message for the story node.
    """
    content=(
        INSTRUCTIONS +
        "The user and the AI are having a conversation about writing a children's story. " +
        "It's your job to write a really nice children's story. Make it exciting and original. " +
        "The first page should only contain the name of the story. " +
        "This is the user's idea of the story: " +
        state["rough_idea"] +
        "This is the list of characters of the story: " +
        json.dumps(state["characters"]) +
        "\n\n"
    )
    if state["story"] is not None:
        if state["changed_node"] is None:
            content += (
                "We already have the following story, but the user wants to refine or change it: " +
                json.dumps(state["story"])
            )
        else:
            content += (
                f"The user changed {state['changed_node']} and now we have to see if the story is still appropriate." +
                "The current story is: " +
                json.dumps(state["story"])
            )

    return SystemMessage(
        content=content
    )

def story_node(state: AgentState, config: RunnableConfig):
    """
    The story node is responsible for generating the pages of the children's story.
    """
    model = ChatOpenAI(model="gpt-4o").with_structured_output(Story)
    response = model.invoke([
        *state["messages"],
        make_system_message(state)
    ], config)
    response_dict = response.dict()
    return {
        "messages": AIMessage(content=json.dumps(response_dict)),
        "story": response_dict["pages"],
        "changed_node": state["changed_node"] or "story",
    }
