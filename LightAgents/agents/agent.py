from typing import List, Optional

from pydantic import BaseModel, ValidationError
from LightAgents.tools.tools import Tool
import instructor
from LightAgents.memory.MemoryBuffer import MemoryBuffer
from LightAgents.agents.helper_classes import ToolExecution, ToolResponse


class Agent:
    """
    The Agent class is the main interface for interacting with the AI assistant. It provides methods to run tools, get responses from the model, store messages in memory, and retrieve previous messages. The Agent class is responsible for managing the conversation flow, executing tools, and providing responses to the user. It uses the Instructor client  and the MemoryBuffer class to store and retrieve messages from memory.
    """

    def __init__(
        self,
        client: instructor.client.Instructor,  # The Instructor client used to interact with the underlying model.
        model_name: str,  # The name of the model to use for generating responses.
        system_prompt: str,  # The system prompt to be used when generating responses.
        tools: List[Tool] = None,  # A list of tools available to the assistant.
        memory_buffer: MemoryBuffer = None,  # An optional MemoryBuffer object to store and retrieve messages.
    ):
        if tools is None:
            tools = []
        self.client = client
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.tools = tools
        self.current_input_tokens = 0
        self.current_completion_tokens = 0
        self.total_input_tokens = 0
        self.total_completion_tokens = 0
        if memory_buffer is None:
            self.memory_buffer = MemoryBuffer()
        else:
            self.memory_buffer = memory_buffer

    # The run_tool_sync method executes a tool synchronously and returns the output as a ToolResponse object.
    def run_tool_sync(self, tool_execution: ToolExecution) -> ToolResponse:
        tool_to_execute = None
        # Find the tool to execute based on the tool name
        for tool in self.tools:
            if tool.function.__name__ == tool_execution.tool_name:
                tool_to_execute = tool
                break
        # If the tool is not found, return the tool response without any tool_output field.
        if tool_to_execute is None:
            return ToolResponse(
                success=False,
                message=f"No tool function found with the name: {tool_execution.tool_name}",
            )
        # Validate the input parameters for the tool
        try:
            validated_parameters = tool_to_execute.validate_input(
                **tool_execution.parameters
            )
            # Run the tool with the validated parameters
            tool_output = tool_to_execute.run(**validated_parameters.dict())
            # Return the tool response with the tool_output field
            return ToolResponse(
                success=True,
                message="Tool executed successfully",
                tool_output=tool_output.dict(),
            )
        # If there is a validation error, return the error message in the tool response
        except ValidationError as e:
            return ToolResponse(
                success=False, message=f"Parameter validation error: {str(e)}"
            )

    # The run_tool_async method executes a tool asynchronously and returns the output as a ToolResponse object.
    async def run_tool_async(self, tool_execution: ToolExecution) -> ToolResponse:
        tool_to_execute = None
        # Find the tool to execute based on the tool name
        for tool in self.tools:
            if tool.function.__name__ == tool_execution.tool_name:
                tool_to_execute = tool
                break
        # If the tool is not found, return the tool response without any tool_output field.
        if tool_to_execute is None:
            return ToolResponse(
                success=False,
                message=f"No tool function found with the name: {tool_execution.tool_name}",
            )

        try:
            # Validate the input parameters for the tool
            validated_parameters = tool_to_execute.validate_input(
                **tool_execution.parameters
            )
            # Run the tool with the validated parameters
            tool_output = await tool_to_execute.run(**validated_parameters.dict())
            # Return the tool response with the tool_output field
            return ToolResponse(
                success=True,
                message="Tool executed successfully",
                tool_output=tool_output.dict(),
            )
        # If there is a validation error, return the error message in the tool response
        except ValidationError as e:
            return ToolResponse(
                success=False, message=f"Parameter validation error: {str(e)}"
            )

    # The run_tool method executes a tool based on the type of tool (sync or async) and returns the output as a ToolResponse object.
    def run_tool(self, tool_execution: ToolExecution) -> ToolResponse:
        # Find the tool to execute based on the tool name
        tool_to_execute = None
        for tool in self.tools:
            # Compare the tool name with the tool_execution tool_name
            if tool.function.__name__ == tool_execution.tool_name:
                tool_to_execute = tool
                break
        # If the tool is not found, return the tool response without any tool_output field.
        if tool_to_execute is None:
            return ToolResponse(
                success=False,
                message=f"No tool function found with the name: {tool_execution.tool_name}",
            )

        # Execute the tool based on the type of tool (sync or async)
        if tool_to_execute.type == "async":
            return self.run_tool_async(tool_execution)
        else:
            return self.run_tool_sync(tool_execution)

    # The _generate_tools_prompt method generates a list of all the tools available to the assistant.
    def _generate_tools_prompt(self):
        if not self.tools:
            return "No tools information available."
        # Create a list of all the tools schema
        tools_info = "*Tools Available:*:\n" + "\n".join(
            [f"- {tool.generate_schemas()}" for tool in self.tools]
        )
        return tools_info
    
    def modify_system_prompt(self):
        if self.tools:
            tools_prompt = self._generate_tools_prompt()
            modified_system_prompt = f"{self.system_prompt}\n\n{tools_prompt}"
            return modified_system_prompt
        return self.system_prompt
    
    def process_additional_details(self, content, **kwargs):
        if kwargs:
            additional_details = "; ".join(
                [
                    f"{key.capitalize().replace('_', ' ')}: {value}"
                    for key, value in kwargs.items()
                ]
            )
            # Append additional details to the content
            if isinstance(content, dict):
                content["additional_details"] = additional_details
            # If the content is a list, append the additional details as a new message
            else:
                content = f"{content}\n\n{additional_details}"
        return content

    # The run method is the main method used to interact with the AI assistant. It uses the system prompt, gets the response from the model, and stores the messages in memory.
    def run(
        self,
        response_model: BaseModel,  # The response model to use for generating responses. see instructor documentation for more details.
        role: str,  # Role (user, assistant, system) of the message.
        content: str | list,  # The content of the message.
        run_with_completion: bool = False,
        append_all_prev_messages=False,  # Whether to append all previous messages to the current conversation.
        last_k_messages: Optional[
            int
        ] = 5,  # The number of previous messages to include in the conversation
        **kwargs,  # Additional keyword arguments
    ):
        system_prompt = self.modify_system_prompt()
        content = self.process_additional_details(content, **kwargs)
        # Get the response from the model
        return self.get_response(
            system_prompt,
            response_model,
            role,
            content,
            run_with_completion,
            append_all_prev_messages,
            last_k_messages,
        )
    
    # The run_async method is the main method used to interact with the AI assistant asynchronously. It uses the system prompt, gets the response from the model, and stores the messages in memory.
    async def run_async(
        self,
        response_model: BaseModel,  # The response model to use for generating responses. see instructor documentation for more details.
        role: str,  # Role (user, assistant, system) of the message.
        content: str | list,  # The content of the message.
        run_with_completion: bool = False,
        append_all_prev_messages=False,  # Whether to append all previous messages to the current conversation.
        last_k_messages: Optional[
            int
        ] = 5,  # The number of previous messages to include in the conversation
        **kwargs,  # Additional keyword arguments
    ):
        system_prompt = self.modify_system_prompt()
        content = self.process_additional_details(content, **kwargs)
        # Get the response from the model
        return await self.get_response_async(
            system_prompt,
            response_model,
            role,
            content,
            run_with_completion,
            append_all_prev_messages,
            last_k_messages,
        )

    # The store_message method stores a message in the memory buffer.
    def store_message(self, message: str | list, role: str):
        self.memory_buffer.store_message(message, role)

    # The get_previous_messages method retrieves all the previous messages from the memory buffer.
    def get_previous_messages(self):
        return self.memory_buffer.get_previous_messages()

    # The get_latest_messages method retrieves the latest k messages from the memory buffer.
    def get_latest_messages(self, k: int):
        return self.memory_buffer.get_latest_messages(k)

    # The reset_memory method resets the memory buffer.
    def reset_memory(self):
        self.memory_buffer.reset_memory()

    # The get_response method generates a response from the model based on the system prompt, role, and content.
    def get_response(
        self,
        system_prompt: str,  # The system prompt to be used when generating responses.
        response_model: BaseModel,  # The response model to use for generating responses.
        role: str,  # Role (user, assistant, system) of the message.
        content: str | list,  # The content of the message.
        run_with_completion: bool = False, # Whether to run the model with completion.
        append_all_prev_messages=False,  # Whether to append all previous messages to the current conversation.
        last_k_messages: Optional[
            int
        ] = 5,  # The number of previous messages to include in the conversation
    ):
        # initialize the messages list
        messages = []
        # Append the system prompt to the messages list
        messages.append({"role": "system", "content": system_prompt})
        # Append all previous messages to the messages list if append_all_prev_messages is True
        if append_all_prev_messages:
            messages.extend(self.get_previous_messages())
        else:
            messages.extend(self.get_latest_messages(last_k_messages))
        # Append the current message to the messages list
        messages.append({"role": role, "content": content})

        if run_with_completion:
            # Get the response from the model
            assistant_resp, completion = (
                self.client.chat.completions.create_with_completion(
                    model=self.model_name,
                    messages=messages,
                    response_model=response_model,
                    max_retries=5,
                    strict=True,
                )
            )
            # Update the token counts
            self.current_completion_tokens = completion.usage.completion_tokens
            self.current_input_tokens = completion.usage.prompt_tokens
            self.total_completion_tokens += self.current_completion_tokens
            self.total_input_tokens += self.current_input_tokens
        
        else:
            # Get the response from the model
            assistant_resp = (
                self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_model=response_model,
                    max_retries=5,
                    strict=False,
                )
            )

        # Store the messages in memory
        self.store_message(content, role)
        # Store the assistant response in memory
        self.store_message(str(assistant_resp), "assistant")
        return assistant_resp
    
    async def get_response_async(
        self,
        system_prompt: str,  # The system prompt to be used when generating responses.
        response_model: BaseModel,  # The response model to use for generating responses.
        role: str,  # Role (user, assistant, system) of the message.
        content: str | list,  # The content of the message.
        run_with_completion: bool = False, # Whether to run the model with completion.
        append_all_prev_messages=False,  # Whether to append all previous messages to the current conversation.
        last_k_messages: Optional[
            int
        ] = 5,  # The number of previous messages to include in the conversation
    ):
        # initialize the messages list
        messages = []
        # Append the system prompt to the messages list
        messages.append({"role": "system", "content": system_prompt})
        # Append all previous messages to the messages list if append_all_prev_messages is True
        if append_all_prev_messages:
            messages.extend(self.get_previous_messages())
        else:
            messages.extend(self.get_latest_messages(last_k_messages))
        # Append the current message to the messages list
        messages.append({"role": role, "content": content})

        if run_with_completion:
            # Get the response from the model
            assistant_resp, completion = (
                await self.client.chat.completions.create_with_completion(
                    model=self.model_name,
                    messages=messages,
                    response_model=response_model,
                    max_retries=5,
                    strict=True,
                )
            )
            # Update the token counts
            self.current_completion_tokens = completion.usage.completion_tokens
            self.current_input_tokens = completion.usage.prompt_tokens
            self.total_completion_tokens += self.current_completion_tokens
            self.total_input_tokens += self.current_input_tokens
        
        else:
            # Get the response from the model
            assistant_resp = (
                await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_model=response_model,
                    max_retries=5,
                    strict=False,
                )
            )

        # Store the messages in memory
        self.store_message(content, role)
        # Store the assistant response in memory
        self.store_message(str(assistant_resp), "assistant")
        return assistant_resp
