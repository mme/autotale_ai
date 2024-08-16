"""
Main chatbot node.
"""

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import END

from autotale_ai.state import AgentState

# pylint: disable=line-too-long

async def chatbot_node(state: AgentState, config: RunnableConfig):
    """
    The chatbot is responsible for answering the user's questions and selecting
    the next route.
    """
    tools = [_make_chatbot_routes(state)]
    response = await ChatOpenAI(model="gpt-4o").bind_tools(tools).ainvoke([
        *state["messages"],
        SystemMessage(
            content=(
                "You help the user write a children's story. Please assist the user by " +
                "either having a conversation or by routing them to the appropriate step in the " +
                "story writing process. " +
                "Do not make proactive suggestions, only if the user explicitly asks for them. " +
                "Do not repeat the whole story again."
            )
        )
    ], config)

    return {
        "messages": response,

        # reset changed_node and prev_state
        "changed_node": None,
        "prev_state": None
    }


@tool()
def routing_tool(route: str): # pylint: disable=unused-argument
    """Dummy tool for the tool node."""
    return route

chatbot_tools_node = ToolNode([routing_tool])

def _make_chatbot_routes(state: AgentState):
    """
    Create an OpenAI tool that will return the next route.
    (Using the native format so we can have enums)
    """
    rough_idea = [
        "rough_idea", 
        "Determine the rough idea for a children's story.",
        "Make sure to collect at least a simple idea about the story before moving to this step.",
        "requires no previous information"
    ]
    characters = [
        "characters", 
        "Create a list of characters for the story.",
        "Make sure to have an idea about the characters before moving to this step.",
        "requires a rough idea"
    ]
    story = [
        "story", 
        "Write the story.",
        "You may collect additional information about the story if needed, then write the story.",
        "requires characters"
    ]

    available_routes = [rough_idea]
    unavailable_routes = []

    (available_routes
     if state["rough_idea"] is not None
     else unavailable_routes).append(characters)

    (available_routes
     if state["rough_idea"] is not None and state["characters"] is not None
     else unavailable_routes).append(story)

    description = "Select a route indicating the next action the user wants to take.\n"
    description += "Don't mention unavailable routes, unless the user explicitly asks for them.\n"
    description += "VERY IMPORTANT: If the user wants to change any aspect of the story, you MUST\n"
    description += "route them to the appropriate node. If the user wants to change the rough idea,\n"
    description += "for example, you MUST route them to the rough_idea node if available.\n"
    description += "If the user wants to change the characters (for example a name), "
    description += "you MUST route them to the characters node.\n"
    description += "If the user wants to change the story, you MUST route them to the story node.\n"
    description += "You get the idea\n"

    parameter_description = "\n\nThe following routes are available:\n"
    parameter_description += "\n".join(f"- {route[0]}: {route[1]} ({route[2]})" for route in available_routes)
    if len(unavailable_routes) > 0:
        parameter_description += "\n\nThe following routes are unavailable:\n"
        parameter_description += "\n".join(f"- {route[0]}: {route[1]} ({route[3]})" for route in unavailable_routes) 

    return {
        "name": "routing_tool",
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "route": {
                    "type": "string",
                    "description": parameter_description,
                    "enum": [route[0] for route in available_routes]
                }
            }
        }
    }


def route_chatbot(state):
    """Route the chatbot to tools or END."""
    if not state["messages"][-1].tool_calls:
        return END
    return "chatbot_tools_node"
