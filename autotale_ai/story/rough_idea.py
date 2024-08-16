"""
Rough idea node.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from autotale_ai.state import AgentState

@tool
def set_rough_idea(rough_idea: str):
    """
    From the context of the conversation, set the rough idea for the story
    that the user has in mind.
    """
    return rough_idea

def make_system_message(state: AgentState):
    """
    Make a system message for the rough idea node.
    """
    content=(
        "The user and the AI are having a conversation about writing a children's story. " +
        "It's your job to extract the rough idea from the conversation."
    )
    if state["rough_idea"] is not None:
        content += (
            "The user has already provided a rough idea for the story, but now they want to "
            "refine or change it: " +
            state["rough_idea"]
        )
        
    return SystemMessage(
        content=content
    )

async def rough_idea_node(state: AgentState, config: RunnableConfig):
    """
    The rough idea node is responsible for generating a rough idea for the story.
    """
    model = ChatOpenAI(model="gpt-4o").bind_tools([set_rough_idea], tool_choice="set_rough_idea")
    response = await model.ainvoke([
        *state["messages"],
        make_system_message(state)
    ], config)
    rough_idea = response.tool_calls[0]["args"]["rough_idea"]
    messages = [
        response,
        ToolMessage(
            tool_call_id=response.tool_calls[0]["id"], 
            content=rough_idea,            
        )
    ]
    return {
        "messages": messages,
        "rough_idea": rough_idea,
        "changed_node": state["changed_node"] or "rough_idea",
    }
