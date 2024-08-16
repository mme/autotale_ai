"""
This is the main entry point for the autotale AI.
It defines the workflow graph and the entry point for the agent.
"""

from langgraph.graph import StateGraph, END

from autotale_ai.state import AgentState
from autotale_ai.chatbot import chatbot_node, chatbot_tools_node, route_chatbot
from autotale_ai.story.rough_idea import rough_idea_node
from autotale_ai.story.characters import characters_node
from autotale_ai.story.story import story_node

def route_story_writing(state):
    """Route to story writing nodes."""
    last_message = state["messages"][-1]
    if last_message.name != "routing_tool":
        return "chatbot_node"
    # route to the node determined by the LLM
    return last_message.content

# Define a new graph
workflow = StateGraph(AgentState)
workflow.add_node("chatbot_node", chatbot_node)
workflow.add_node("chatbot_tools_node", chatbot_tools_node)
workflow.add_node("rough_idea_node", rough_idea_node)
workflow.add_node("characters_node", characters_node)
workflow.add_node("story_node", story_node)

# Chatbot
workflow.set_entry_point("chatbot_node")
workflow.add_edge("chatbot_node", END)
workflow.add_conditional_edges(
    "chatbot_node",
    route_chatbot,
    {
        END: END,
        "chatbot_tools_node": "chatbot_tools_node",
    }
)
workflow.add_conditional_edges(
    "chatbot_tools_node", 
    route_story_writing,
    {
        "chatbot_node": "chatbot_node",
        "rough_idea": "rough_idea_node",
        "characters": "characters_node",
        "story": "story_node",
    }
)
workflow.add_conditional_edges(
    "rough_idea_node",
    lambda state: "chatbot_node" if not state.get("characters") else "characters_node",
    {
        "chatbot_node": "chatbot_node",
        "characters_node": "characters_node",
    }
)

workflow.add_conditional_edges(
    "characters_node",
    lambda state: "chatbot_node" if not state.get("story") else "story_node",
    {
        "chatbot_node": "chatbot_node",
        "story_node": "story_node",
    }
)

workflow.add_edge("story_node", "chatbot_node")


graph = workflow.compile()
