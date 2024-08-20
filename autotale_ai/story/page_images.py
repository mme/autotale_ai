"""
Characters node.
"""

import json

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableConfig
from langgraph.constants import Send
from openai import OpenAI

from autotale_ai.state import AgentState, PageImageGenerationState
#
def generate_page_image_node(state: PageImageGenerationState, config: RunnableConfig): # pylint: disable=unused-argument
    """
    Generate an image for a page.
    """
    client = OpenAI()

    image_description = generate_page_image_description(state, config)
    response = client.images.generate(
        model="dall-e-3",
        prompt=image_description,
        size="1024x1024",
        quality="standard",
        n=1,
    )
    image_url = response.data[0].url
    return {
        "page_images": [
            {
                "image_url": image_url,
                "image_description": image_description,
                "idx": state["idx"]
            }
        ]
    }


def page_image_generation_parallel(state: AgentState):
    """
    We will return a list of `Send` objects
    Each `Send` object consists of the name of a node in the graph
    as well as the state to send to that node
    """
    result = []
    for idx, page in enumerate(state["story"]):
        messages = state.get("messages")

        result.append(
            Send(
                "generate_page_image_node",
                {
                    "page": page,
                    "messages": messages,
                    "characters": state["characters"],
                    "idx": idx
                }
            )
        )
    return result

class ImageDescription(BaseModel):
    """
    Represents the description of an image of a page in the story.
    """
    description: str

def generate_page_image_description(state: PageImageGenerationState, config: RunnableConfig):
    """
    Generate a description of the image of a page.
    """

    system_message = SystemMessage(
        content=(
f"""
The user and the AI are having a conversation about writing a children's story. 
It's your job to generate a vivid description of a page of that story.
Make the description as detailed as possible.
This is the page: \n\n{json.dumps(state["page"])}\n\n
These are all characters in the story: \n\n{json.dumps(state["characters"])}\n\n
(Not all of the characters are necessarily in the page, so only consider characters that are in the page.)
Imagine an image of the page. Describe the looks of the scene in great detail.
Describe the characters in the scene, but also the setting and the scene itself in great detail.
If the user mentioned a specific style for the images in the conversation, YOU MUST 
include that style in your description. Describe the style in detail, it's important.
"""
        )
    )
    model = ChatOpenAI(model="gpt-4o").with_structured_output(ImageDescription)
    response = model.invoke([
        *state["messages"],
        system_message
    ], config)

    return response.description
