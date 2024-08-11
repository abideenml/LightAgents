import instructor
from openai import OpenAI
from pydantic import BaseModel
from LightAgents.memory.MemoryBuffer import MemoryBuffer
from LightAgents.memory.MemoryStorage import MemoryStorage
from LightAgents.agents.agent import Agent
from LightAgents.systemprompt.systemprompt import SimpleSystemPromptGenerator


# Initialize Memory Storage
memory_storage = MemoryStorage()
print(memory_storage.load_messages())

# Initialize Memory Buffer
memory_buff = MemoryBuffer(storage_db=memory_storage)
memory_buff.load_messages()



# Initialize sample prompt generator
system_prompt_generator = SimpleSystemPromptGenerator(
    goal="Provide answer by solving the steps provided to you.",
    background="Your are a math tutor for pre-schoolers, you need to explain the problem in easiest way possible and derive the solution in steps.")

# Agent Client 
client = instructor.from_openai(OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='your_api_key',
), mode=instructor.Mode.JSON)


# Initialize Response Model
class ResponseModel(BaseModel):
    answer: str

# Initialize Agent
agent = Agent(client=client, model_name="llama3", system_prompt_generator=system_prompt_generator, memory_buffer=memory_buff)

# agent.store_message("My name is Adeel", "user")
# agent.store_message("Hi", "assistant")
# agent.store_message("How are you?", "user")
resp = agent.run(ResponseModel, user_message="What is my name?", last_k_messages=5)

print(resp)