"""
This is the main entry point for the autotale AI.
It defines the workflow graph and the entry point for the agent.
"""
# pylint: disable=line-too-long, unused-import

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import ToolMessage

from autotale_ai.state import AgentState
from autotale_ai.chatbot import chatbot_node
from autotale_ai.story.outline import outline_node
from autotale_ai.story.characters import characters_node
from autotale_ai.story.story import story_node
# from autotale_ai.story.character_images import (
  
#   generate_character_image_node,
#   continue_to_character_image_generation,
#   generate_character_images_node
# )
from autotale_ai.story.page_images import (
  page_image_generation_parallel,
  generate_page_image_node,
#   continue_to_page_image_generation,
#   generate_page_images_node
)

def route_story_writing(state):
    """Route to story writing nodes."""
    print(state["messages"][-1])
    last_message = state["messages"][-1]

    if isinstance(last_message, ToolMessage):
        return last_message.name
    return END

# Define a new graph
workflow = StateGraph(AgentState)
workflow.add_node("chatbot_node", chatbot_node)
workflow.add_node("outline_node", outline_node)
workflow.add_node("characters_node", characters_node)
workflow.add_node("story_node", story_node)
# workflow.add_node("generate_page_image_node", generate_page_image_node)

# workflow.add_node("characters_node", characters_node)
# workflow.add_node("story_node", story_node)
# workflow.add_node("generate_character_image_node", generate_character_image_node)
# workflow.add_node("generate_page_image_node", generate_page_image_node)
# workflow.add_node("generate_character_images_node", generate_character_images_node)
# workflow.add_node("generate_page_images_node", generate_page_images_node)

# Chatbot
workflow.set_entry_point("chatbot_node")
# workflow.add_edge("chatbot_node", END)

workflow.add_conditional_edges(
    "chatbot_node", 
    route_story_writing,
    {
        "set_outline": "outline_node",
        "set_characters": "characters_node",
        "set_story": "story_node",
        END: END,
        # "characters": "characters_node",
        # "story": "story_node",
        # "generate_character_images": "generate_character_images_node",
        # "generate_page_images": "generate_page_images_node",
    }
)
workflow.add_edge(
    "outline_node",
    "chatbot_node"
)

workflow.add_edge(
    "characters_node",
    "chatbot_node"
)

workflow.add_edge(
    "story_node",
    "chatbot_node"
)

# workflow.add_conditional_edges(
#     "story_node",
#     page_image_generation_parallel,
#     ["generate_page_image_node"]
# )


# workflow.add_conditional_edges(
#     "generate_page_images_node",
#     continue_to_page_image_generation,
#     ["generate_page_image_node"]
# )

# workflow.add_conditional_edges(
#     "characters_node",
#     continue_to_character_image_generation,
#     ["generate_character_image_node"]
# )

# workflow.add_conditional_edges(
#     "generate_character_images_node",
#     continue_to_character_image_generation,
#     ["generate_character_image_node"]
# )


# workflow.add_conditional_edges(
#     "generate_character_image_node",
#     lambda state: "chatbot_node" if not state.get("story") else "story_node",
#     {
#         "chatbot_node": "chatbot_node",
#         "story_node": "story_node",
#     }
# )
# workflow.add_edge("generate_page_image_node", "chatbot_node")

graph = workflow.compile()
