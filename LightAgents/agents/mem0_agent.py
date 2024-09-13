from typing import List
import instructor
from pydantic import BaseModel, ValidationError
from LightAgents.tools.tools import Tool
from LightAgents.agents.helper_classes import ToolExecution, ToolResponse
from mem0 import Memory


class MemoAgent:
    """
    The Mem0Agent uses mem0 as a memory store to store and retrieve information.
    """

    def __init__(
        self,
        client: instructor.client.Instructor,  # Instructor client
        model_name: str,  # Model name
        system_prompt: str,  # System prompt
        tools: List[Tool] = None,  # List of tools
        memo_config: dict = None,  # Configuration for mem0
    ):
        if tools is None:
            tools = []
        self.client = client
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.tools = tools
        self.memory = Memory.from_config(memo_config) if memo_config else Memory()
        self.messages = []
        self.current_input_tokens = 0
        self.current_completion_tokens = 0
        self.total_input_tokens = 0
        self.total_completion_tokens = 0

    # Function to run a tool synchronously
    def run_tool_sync(self, tool_execution: ToolExecution) -> ToolResponse:
        tool_to_execute = None
        # Find the tool to execute
        for tool in self.tools:
            # Check if the tool name matches the tool name in the tool execution
            if tool.function.__name__ == tool_execution.tool_name:
                tool_to_execute = tool
                break
        # If no tool is found, return the tool response without any tool output
        if tool_to_execute is None:
            return ToolResponse(
                success=False,
                message=f"No tool function found with the name: {tool_execution.tool_name}",
            )
        # Validate the input parameters
        try:
            # Validate the input parameters
            validated_parameters = tool_to_execute.validate_input(
                **tool_execution.parameters
            )
            # Run the tool with the validated parameters
            tool_output = tool_to_execute.run(**validated_parameters.dict())
            # Return the tool response with the tool output
            return ToolResponse(
                success=True,
                message="Tool executed successfully",
                tool_output=tool_output.dict(),
            )
        # If there is a validation error, return the tool response with the error message
        except ValidationError as e:
            return ToolResponse(
                success=False, message=f"Parameter validation error: {str(e)}"
            )

    # Function to run a tool asynchronously
    async def run_tool_async(self, tool_execution: ToolExecution) -> ToolResponse:
        # Find the tool to execute
        tool_to_execute = None
        # Find the tool to execute
        for tool in self.tools:
            # Check if the tool name matches the tool name in the tool execution
            if tool.function.__name__ == tool_execution.tool_name:
                tool_to_execute = tool
                break
        # If no tool is found, return the tool response without any tool output
        if tool_to_execute is None:
            # Return the tool response without any tool output
            return ToolResponse(
                success=False,
                message=f"No tool function found with the name: {tool_execution.tool_name}",
            )
        # Validate the input parameters
        try:
            # Validate the input parameters
            validated_parameters = tool_to_execute.validate_input(
                **tool_execution.parameters
            )
            # Run the tool with the validated parameters
            tool_output = await tool_to_execute.run(**validated_parameters.dict())
            # Return the tool response with the tool output
            return ToolResponse(
                success=True,
                message="Tool executed successfully",
                tool_output=tool_output.dict(),
            )
        # If there is a validation error, return the tool response with the error message
        except ValidationError as e:
            return ToolResponse(
                success=False, message=f"Parameter validation error: {str(e)}"
            )

    # Function to run a tool
    def run_tool(self, tool_execution: ToolExecution) -> ToolResponse:
        # Find the tool to execute
        tool_to_execute = None
        # Find the tool to execute
        for tool in self.tools:
            # Check if the tool name matches the tool name in the tool execution
            if tool.function.__name__ == tool_execution.tool_name:
                tool_to_execute = tool
                break
        # If no tool is found, return the tool response without any tool output
        if tool_to_execute is None:
            # Return the tool response without any tool output
            return ToolResponse(
                success=False,
                message=f"No tool function found with the name: {tool_execution.tool_name}",
            )

        # Check if the tool is async or sync
        if tool_to_execute.type == "async":
            return self.run_tool_async(tool_execution)
        # If the tool is sync, run the tool synchronously
        else:
            return self.run_tool_sync(tool_execution)

    # Function to generate the system prompt
    def _generate_tools_prompt(self):
        if not self.tools:
            return "No tools information available."
        tools_info = "*Tools Available to you:*" + "\n".join(
            [f"- {tool.generate_schemas()}" for tool in self.tools]
        )
        return tools_info

    # Function to run the agent
    def run(
        self,
        response_model: BaseModel,
        role: str,
        content: str | dict,
        user_id: str,
        run_with_completion: bool = False,
        **kwargs,
    ):
        system_prompt = self.system_prompt
        # Generate the tools prompt and add it to the system prompt
        if self.tools:
            tools_prompt = self._generate_tools_prompt()
            system_prompt = f"{system_prompt}\n\n{tools_prompt}"
        # Add additional details to the user message
        if kwargs:
            additional_details = "; ".join(
                [
                    f"{key.capitalize().replace('_', ' ')}: {value}"
                    for key, value in kwargs.items()
                ]
            )
            # Add additional details to the user message
            if isinstance(content, dict):
                content["additional_details"] = additional_details
            # If the content is a list, append the additional details as a new message
            else:
                content = f"{content}\n\n{additional_details}"
        # Run the agent
        return self.get_response(
            system_prompt, response_model, role, content, user_id, run_with_completion
        )

    # Function to run the agent asynchronously
    async def run_async(
        self,
        response_model: BaseModel,
        role: str,
        content: str | dict,
        user_id: str,
        run_with_completion: bool = False,
        **kwargs,
    ):
        system_prompt = self.system_prompt
        # Generate the tools prompt and add it to the system prompt
        if self.tools:
            tools_prompt = self._generate_tools_prompt()
            system_prompt = f"{system_prompt}\n\n{tools_prompt}"
        if kwargs:
            additional_details = "; ".join(
                [
                    f"{key.capitalize().replace('_', ' ')}: {value}"
                    for key, value in kwargs.items()
                ]
            )
            # Add additional details to the user message
            if isinstance(content, dict):
                content["additional_details"] = additional_details
            # If the content is a list, append the additional details as a new message
            else:
                content = f"{content}\n\n{str(additional_details)}"
        # Run the agent asynchronously
        return await self.get_response_async(
            system_prompt, response_model, role, content, user_id, run_with_completion
        )

    # Function to get the response from the assistant synchronously
    def get_response(
        self,
        system_prompt: str,  # System prompt
        response_model: BaseModel,  # Response model
        role: str,  # Role
        content: str | dict,  # Content
        user_id: str | int,  # User ID
        run_with_completion: bool = False,  # Run with completion
    ):
        # Initialize the messages list
        self.messages = []
        # Add the system prompt to the messages
        self.messages.append({"role": "system", "content": system_prompt})
        # search for relevant memories
        relevant_memories = self.search_memories(str(content), user_id=user_id)
        # Add the user message to the messages
        if relevant_memories:
            # Add the relevant memories to the content
            if isinstance(content, dict):
                content["memories"] = relevant_memories
            else:
                content = f"{content}\n\nMemories:\n" + "\n".join(relevant_memories)

        # Add the user message to the messages
        self.messages.append({"role": role, "content": content})

        if run_with_completion:
            assistant_resp, completion = (
                self.client.chat.completions.create_with_completion(
                    model=self.model_name,
                    messages=self.messages,
                    response_model=response_model,
                )
            )

            # Update the token counts
            self.current_completion_tokens = completion.usage.completion_tokens
            self.current_input_tokens = completion.usage.prompt_tokens
            self.total_completion_tokens += self.current_completion_tokens
            self.total_input_tokens += self.current_input_tokens

        else:
            assistant_resp = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                response_model=response_model,
            )

        # Add the assistant response to the memory
        self.memory.add(data=str(assistant_resp), prompt=content, user_id=user_id)

        return assistant_resp

    async def get_response_async(
        self,
        system_prompt: str,
        response_model: BaseModel,
        role: str,
        content: str | dict,
        user_id: str,
        run_with_completion: bool = False,
    ):
        self.messages = []
        self.messages.append({"role": "system", "content": system_prompt})

        # Retrieve relevant memories
        relevant_memories = self.search_memories(str(content), user_id=user_id)

        if relevant_memories:
            if isinstance(content, dict):
                content["memories"] = relevant_memories
            else:
                content = f"{content}\n\nMemories:\n" + "\n".join(relevant_memories)

        self.messages.append({"role": role, "content": content})

        if run_with_completion:
            (
                assistant_resp,
                completion,
            ) = await self.client.chat.completions.create_with_completion(
                model=self.model_name,
                messages=self.messages,
                response_model=response_model,
            )

            self.current_completion_tokens = completion.usage.completion_tokens
            self.current_input_tokens = completion.usage.prompt_tokens
            self.total_completion_tokens += self.current_completion_tokens
            self.total_input_tokens += self.current_input_tokens

        else:
            assistant_resp = await self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                response_model=response_model,
            )

        self.memory.add(data=str(assistant_resp), prompt=content, user_id=user_id)

        return assistant_resp

    # get memories
    def get_memories(self, user_id: str):
        memories = self.memory.get_all(user_id=user_id)
        return [m["text"] for m in memories]

    # search memories
    def search_memories(self, query: str, user_id: str):
        memories = self.memory.search(query, user_id=user_id)
        return [m["text"] for m in memories]
