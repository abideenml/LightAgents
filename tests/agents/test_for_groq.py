import os
import sys

# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from typing import List
from pydantic import BaseModel
from LightAgents.agents.agent import Agent, ToolExecution
from LightAgents.systemprompt.systemprompt import SimpleSystemPromptGenerator
from tests.agents.defined_tools import check_availability_tool, add_item_tool
import instructor
import dotenv
from groq import Groq



dotenv.load_dotenv('.env')



# Define the response models
class ResponseModel(BaseModel):
    isToolNeeded: bool
    message: str
    functionName: str
    parameters: List[dict]

# System prompt setup for a hotel assistant
goal = "Assist the user by providing information or performing actions related to hotel inventory."
background = '''
-You are a helpful hotel assistant with access to inventory management tools. You can check the availability of items and add new items to the inventory.
-If the user asks for an item and it's not available, you can offer to add it to the inventory.
-The user will let you know about the response of the tool you executed.
'''
system_prompt_generator = SimpleSystemPromptGenerator(goal=goal, background=background)
system_prompt = system_prompt_generator.generate()

# Tools setup
tools = [check_availability_tool, add_item_tool]

Groq_API_KEY = os.getenv("GROQ_API_KEY")

# Agent setup
client = instructor.from_groq(Groq(api_key=Groq_API_KEY) , mode=instructor.Mode.JSON) 

hotel_assistant = Agent(client=client, model_name="llama-3.1-70b-versatile", system_prompt=system_prompt, tools=tools)

while True:
    user_message = input("You: ")
    if user_message == "exit":
        break
    resp = hotel_assistant.run(response_model=ResponseModel, role="user", content=user_message)
    if resp.isToolNeeded:
        parameters = {}
        for parameter in resp.parameters:
            for key, value in parameter.items():
                parameters[key] = value
        tool_execution = ToolExecution(tool_name=resp.functionName, parameters=parameters)
        tool_response = hotel_assistant.run_tool_sync(tool_execution)
        print(f"System (Internal): {tool_response.message}")
        print(f"System (Internal): {tool_response.tool_output}")
        resp = hotel_assistant.run(response_model=ResponseModel,role="user" ,content=f"Response of the tool:{tool_response.tool_output}")
        print(f"System: {resp.message}")
    else:
        print(f"System: {resp.message}")