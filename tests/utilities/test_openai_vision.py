import os
import instructor
import openai
import dotenv
import base64
from pydantic import BaseModel

dotenv.load_dotenv()

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


client = instructor.from_openai(openai.OpenAI(
    api_key=OPEN_API_KEY
), mode=instructor.Mode.JSON)

images_folder = "images"
images_list = []
for image in os.listdir(images_folder):
    images_list.append(encode_image(f"{images_folder}/{image}"))



def get_response(client ,encoded_url, response_model):
   messages =  [
      {
        "role": "system",
        "content": "Your job is to describe the image in detail.If there is any text in the image, please transcribe it."
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
            "url": f"data:image/jpeg;base64,{encoded_url}"
          }
        }
      ]
    }
  ]
   resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    response_model=response_model
  )
   return resp

   

class Response(BaseModel):
    message: str

resp = get_response(client, images_list[0], Response)
print(resp.message)
