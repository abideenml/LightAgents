from typing import List
import instructor
from pydantic import BaseModel, ValidationError
from LightAgents.systemprompt.systemprompt import SystemPromptGenerator
from LightAgents.tools.tools import Tool
from LightAgents.agents.helper_classes import ToolExecution, ToolResponse
from mem0 import Memory

class MemoAgent:
    def __init__(self, client: instructor.client.Instructor, model_name: str, system_prompt_generator: SystemPromptGenerator, tools: List[Tool] = None, memo_config: dict = None):
        if tools is None:
            tools = []
        self.client = client
        self.model_name = model_name
        self.system_prompt_generator = system_prompt_generator
        self.tools = tools
        self.memory = Memory.from_config(memo_config) if memo_config else Memory()
        self.messages = []

    def run_tool_sync(self, tool_execution: ToolExecution) -> ToolResponse:
        tool_to_execute = None
        for tool in self.tools:
            if tool.function.__name__ == tool_execution.tool_name:
                tool_to_execute = tool
                break
        if tool_to_execute is None:
            return ToolResponse(success=False, message=f"No tool function found with the name: {tool_execution.tool_name}")

        try:
            validated_parameters = tool_to_execute.validate_input(**tool_execution.parameters)
            print(validated_parameters)
            tool_output = tool_to_execute.run(**validated_parameters.dict())
            return ToolResponse(success=True, message="Tool executed successfully", tool_output=tool_output.dict())
        except ValidationError as e:
            return ToolResponse(success=False, message=f"Parameter validation error: {str(e)}")

    async def run_tool_async(self, tool_execution: ToolExecution) -> ToolResponse:
        tool_to_execute = None
        for tool in self.tools:
            if tool.function.__name__ == tool_execution.tool_name:
                tool_to_execute = tool
                break
        if tool_to_execute is None:
            return ToolResponse(success=False, message=f"No tool function found with the name: {tool_execution.tool_name}")

        try:
            validated_parameters = tool_to_execute.validate_input(**tool_execution.parameters)
            print(validated_parameters)
            tool_output = await tool_to_execute.run(**validated_parameters.dict())
            return ToolResponse(success=True, message="Tool executed successfully", tool_output=tool_output.dict())
        except ValidationError as e:
            return ToolResponse(success=False, message=f"Parameter validation error: {str(e)}")

    def run_tool(self, tool_execution: ToolExecution) -> ToolResponse:
        tool_to_execute = None
        for tool in self.tools:
            if tool.function.__name__ == tool_execution.tool_name:
                tool_to_execute = tool
                break
        if tool_to_execute is None:
            return ToolResponse(success=False, message=f"No tool function found with the name: {tool_execution.tool_name}")

        if tool_to_execute.type == "async":
            return self.run_tool_async(tool_execution)
        else:
            return self.run_tool_sync(tool_execution)

    def _generate_tools_prompt(self):
        if not self.tools:
            return "No tools information available."
        tools_info = "Tools Information:\n" + "\n".join([f"- {tool.generate_schemas()}" for tool in self.tools])
        return tools_info

    def run(self, response_model: BaseModel, user_message: str, user_id: str, **kwargs):
        system_prompt = self.system_prompt_generator.generate()
        if self.tools:
            tools_prompt = self._generate_tools_prompt()
            system_prompt = f"{system_prompt}\n\n{tools_prompt}"
        if kwargs:
            additional_details = "; ".join([f"{key.capitalize().replace('_', ' ')}: {value}" for key, value in kwargs.items()])
            user_message = f"{user_message}\n\nAdditional details:\n{additional_details}."
        return self.get_response(system_prompt, response_model, user_message, user_id)
    

    async def run_async(self, response_model: BaseModel, user_message: str, user_id: str, **kwargs):
        system_prompt = self.system_prompt_generator.generate()
        if self.tools:
            tools_prompt = self._generate_tools_prompt()
            system_prompt = f"{system_prompt}\n\n{tools_prompt}"
        if kwargs:
            additional_details = "; ".join([f"{key.capitalize().replace('_', ' ')}: {value}" for key, value in kwargs.items()])
            user_message = f"{user_message}\n\nAdditional details:\n{additional_details}."
        return await self.get_response_async(system_prompt, response_model, user_message, user_id)

    def get_response(self, system_prompt: str, response_model: BaseModel, user_message: str, user_id: str):
        self.messages = []
        self.messages.append({"role": "system", "content": system_prompt})
        

        user_message_to_add = f"{user_message}"
        
        # Retrieve relevant memories
        relevant_memories = self.search_memories(user_message, user_id=user_id)
        if relevant_memories:
            user_message_to_add = f"{user_message}\n\nMemories:\n" + "\n".join(relevant_memories)
        self.messages.append({"role": "user", "content": user_message_to_add})
            
        assistant_resp, completion = self.client.chat.completions.create_with_completion(
            model=self.model_name,
            messages=self.messages,
            response_model=response_model,
        )
        # self.messages.append({"role": "assistant", "content": str(assistant_resp)})
        print(completion.usage)
        # Store the question in memory
        self.memory.add(data=str(assistant_resp), user_id=user_id , prompt=user_message)
        return assistant_resp

    async def get_response_async(self, system_prompt: str, response_model: BaseModel, user_message: str, user_id: str):
        self.messages = []
        self.messages.append({"role": "system", "content": system_prompt})
        
        user_message_to_send = f"{user_message}"
        
        # Retrieve relevant memories
        relevant_memories = self.search_memories(user_message, user_id=user_id)
        if relevant_memories:
            user_message_to_send = f"{user_message}\n\nMemories:\n" + "\n".join(relevant_memories)
        self.messages.append({"role": "user", "content": user_message_to_send})
            
        assistant_resp, completion = await self.client.chat.completions.create_with_completion(
            model=self.model_name,
            messages=self.messages,
            response_model=response_model,
        )
        print("Usage Details:")
        print(completion.usage)
        # self.messages.append({"role": "assistant", "content": str(assistant_resp)})
        # Store the question in memory
        self.memory.add(data=str(assistant_resp),prompt=user_message ,user_id=user_id)
        return assistant_resp

    def get_memories(self, user_id: str):
        memories = self.memory.get_all(user_id=user_id)
        return [m['text'] for m in memories]

    def search_memories(self, query: str, user_id: str):
        memories = self.memory.search(query, user_id=user_id)
        return [m['text'] for m in memories]