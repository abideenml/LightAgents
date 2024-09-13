import os
import sys

# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import instructor
from openai import OpenAI
from LightAgents.agents.mem0_agent import MemoAgent
from LightAgents.systemprompt.systemprompt import SimpleSystemPromptGenerator
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(".env" , override=True)

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
        }
    },
}

OPEN_API_KEY = os.getenv("OPEN_API_KEY")
## Initilize client
client = instructor.from_openai(OpenAI(
    api_key=OPEN_API_KEY
), mode=instructor.Mode.JSON)


## Create simple system prompt generator
system_prompt_generator = SimpleSystemPromptGenerator(goal="Assist the user in academic research by providing information or performing actions related to research papers.",
                                                      background="You are a personal AI tutor, explain the concepts, help in understanding the research papers, and provide the summary of the research papers.")

system_prompt = system_prompt_generator.generate()

## response model
class Response(BaseModel):
    message: str


# Create mem0 powered agent
agent = MemoAgent(client=client, model_name="gpt-3.5-turbo", system_prompt=system_prompt, memo_config=config)


while True:
    user_message = input("User: ")
    response = agent.run(response_model=Response, role="user", content=user_message , user_id="researcher_12")
    print(f"System: {response.message}")