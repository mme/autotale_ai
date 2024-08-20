"""
Story node.
"""

from typing import List
import json
from langchain_core.tools import tool

from autotale_ai.state import AgentState, Page

@tool
def set_story(pages: List[Page]):
    """
    Considering the outline and characters, write a story.
    Keep it simple, 3-4 sentences per page.
    """
    return pages

def story_node(state: AgentState):
    """
    The story node is responsible for extracting the story from the conversation.
    """
    last_message = state["messages"][-1]
    return {
        "story": json.loads(last_message.content)["pages"]
    }
