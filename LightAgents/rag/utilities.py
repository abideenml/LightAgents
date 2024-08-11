import base64
import os
import uuid

from pydantic import BaseModel


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def encode_images_in_folder(folder_path):
    encoded_images = {}
    for image in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image)
        encoded_images[image] = encode_image(image_path)
    return encoded_images

class ImageDescriptionResponse(BaseModel):
    message: str

def get_image_description_ollama_or_openai(encoded_image,client,response_model = ImageDescriptionResponse,model= "gpt-4o-mini"):
    messages = [
        {
            "role": "system",
            "content": "Your job is to describe the image in detail. If there is any text in the image, please transcribe it."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What’s in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ]
        }
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model
    )
    return resp.message

def append_descriptions_to_markdown(md_text, image_descriptions, image_folder):
    for image, description in image_descriptions.items():
        image_path = image_folder + "/" + image
        md_image_tag = f"![]({image_path})"
        md_description = f"\n\n*Image_Description: {description}*\n"
        md_text = md_text.replace(md_image_tag, md_description)
    return md_text

def create_unique_id(chunk_name):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_name))


# load md text from file
def read_markdown_file(file_path):
    """
    Reads the content of a Markdown file and returns it as a string.

    :param file_path: Path to the Markdown file.
    :return: Content of the file as a string.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"The file at {file_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None
    

async def get_image_description_ollama_or_openai_async(encoded_image, client, response_model=ImageDescriptionResponse, model="gpt-4o-mini"):
    messages = [
        {
            "role": "system",
            "content": "Your job is to describe the image in detail. If there is any text in the image, please transcribe it."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What’s in this image?"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encoded_image}"
                    }
                }
            ]
        }
    ]
    resp = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model
    )
    return resp.message