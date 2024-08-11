from typing import List, Optional

from pydantic import BaseModel, ValidationError
from LightAgents.systemprompt.systemprompt import SystemPromptGenerator
from LightAgents.tools.tools import Tool
import instructor
from LightAgents.memory.MemoryBuffer import MemoryBuffer
from LightAgents.agents.helper_classes import ToolExecution, ToolResponse


class Agent:
    def __init__(self, client: instructor.client.Instructor, model_name: str, system_prompt_generator: SystemPromptGenerator, tools: List[Tool] = None , memory_buffer: MemoryBuffer = None):
        if tools is None:
            tools = []
        self.client = client
        self.model_name = model_name
        self.system_prompt_generator = system_prompt_generator
        self.tools = tools
        if memory_buffer is None:
            self.memory_buffer = MemoryBuffer()
        else:
            self.memory_buffer = memory_buffer


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
        # Create a list of all the tools schema
        tools_info = "Tools Information:\n" + "\n".join([f"- {tool.generate_schemas()}" for tool in self.tools])
        return tools_info

    def run(self,  response_model: BaseModel , user_message: str , last_k_messages: Optional[int] = 5 ,**kwargs ):
        # Generate the system prompt
        system_prompt = self.system_prompt_generator.generate()
        if self.tools:
            tools_prompt = self._generate_tools_prompt()
            system_prompt = f"{system_prompt}\n\n{tools_prompt}"
        # Get the response from the model
        if kwargs:
            additional_details = "; ".join([f"{key.capitalize().replace('_', ' ')}: {value}" for key, value in kwargs.items()])
            user_message = f"{user_message}\n\nAdditional details:\n{additional_details}."
        return self.get_response(system_prompt,response_model,user_message, last_k_messages )
    
    def store_message(self, message: str, role: str):
        self.memory_buffer.store_message(message, role)

    def get_previous_messages(self):
        return self.memory_buffer.get_previous_messages()
    
    def get_latest_messages(self, k: int):
        return self.memory_buffer.get_latest_messages(k)


    def get_response(self, system_prompt: str , response_model: BaseModel , user_message: str , last_k_messages: Optional[int] = 5):
        messages = []
        messages.append({"role": "system", "content": system_prompt})
        messages.extend(self.get_latest_messages(last_k_messages))
        messages.append({"role": "user", "content": user_message})
        self.store_message(user_message, "user")
        assistant_resp =  self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            response_model=response_model,
            max_retries=5,
            strict=False
        )
        self.store_message(str(assistant_resp), "assistant")
        return assistant_resp
