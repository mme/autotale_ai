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

from autotale_ai.state import AgentState, CharacterImageGenerationState
# pylint: disable=line-too-long


class ShouldRegenerateImage(BaseModel):
    """
    Represents the decision to regenerate the image of a character.
    """
    should_regenerate: bool

def should_generate_character_image(state: CharacterImageGenerationState, config: RunnableConfig):
    """
    Decide if the image for a character should be generated.
    """
    if not state["image_url"]:
        return True

    system_message = SystemMessage(
        content=(
            "The user and the AI are having a conversation about writing a children's story. " +
            "Decide if the image for this character should be regenerated. " +
            "Regenerate if the description of the character has changed or if the user asked for a new image. " +
            "This is the character: \n\n" + json.dumps(state["character"]) + "\n\n"
        )
    )

    model = ChatOpenAI(model="gpt-4o").with_structured_output(ShouldRegenerateImage)
    response = model.invoke([
        *state["messages"],
        system_message
    ], config)

    return response.should_regenerate

class ImageDescription(BaseModel):
    """
    Represents the description of an image of a character in the story.
    """
    description: str

def generate_character_image_description(state: CharacterImageGenerationState, config: RunnableConfig):
    """
    Generate a description of the image of a character.
    """

    system_message = SystemMessage(
        content=(
            "The user and the AI are having a conversation about writing a children's story. " +
            "It's your job to generate a vivid description of a character in the story." +
            "Make the description as detailed as possible." +
            "This is the character: \n\n" + json.dumps(state["character"]) + "\n\n" +
            "Imagine an image of the character. Describe the looks of the character in great detail." +
            "Also describe the setting in which the image is taken." +
            "Make sure to include the name of the character and full description of the character in your output. "+
            "If the user mentioned a specific style for the images in the conversation, YOU MUST " +
            "include that style in your description. Describe the style in detail, it's important." +
            "\n"
        )
    )
    model = ChatOpenAI(model="gpt-4o").with_structured_output(ImageDescription)
    response = model.invoke([
        *state["messages"],
        system_message
    ], config)

    return response.description

def generate_character_image_node(state: CharacterImageGenerationState, config: RunnableConfig): # pylint: disable=unused-argument
    """
    Generate an image for a character.
    """
    client = OpenAI()

    if should_generate_character_image(state, config):
        image_description = generate_character_image_description(state, config)
        response = client.images.generate(
            model="dall-e-3",
            prompt=image_description,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        return {
            "character_images": [
                {
                    "image_url": image_url,
                    "image_description": image_description
                }
            ]
        }
    return None

def generate_character_images_node(state: CharacterImageGenerationState, config: RunnableConfig): # pylint: disable=unused-argument
    """
    Generate images for all characters in the story. (Dummy node)
    """
    return None

def continue_to_character_image_generation(state: AgentState):
    """
    We will return a list of `Send` objects
    Each `Send` object consists of the name of a node in the graph
    as well as the state to send to that node
    """
    result = []
    for idx, c in enumerate(state["characters"]):
        messages = state.get("messages")
        image_url = None
        if state.get("character_images") and len(state["character_images"]) > idx:
            image_url = state["character_images"][idx]["image_url"]

        result.append(
            Send(
                "generate_character_image_node",
                {
                    "character": c,
                    "messages": messages,
                    "image_url": image_url
                }
            )
        )
    return result

