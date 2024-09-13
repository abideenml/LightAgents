import sys
import os
import openai
from rich.console import Console

# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import dotenv
import instructor
from LightAgents.agents.agent import Agent
from LightAgents.agents.helper_classes import ToolExecution
from pydantic import BaseModel, Field
from examples.mini_engineer.available_tools import tools_list
from examples.mini_engineer.system_prompt import system_prompt
from rich.panel import Panel
from rich.text import Text

class ToolCall(BaseModel):
    tool_name: str = Field(..., title="Tool Name", description="The name of the tool to be executed.")
    parameters: dict = Field({}, title="Parameters", description="The parameters required for the tool.")

class MiniEngineerResponse(BaseModel):
    summary: str = Field(..., title="Summary", description="The main crux of the response. This field will be used to inform the user about the status of their request and the outcome of the tool calls.")
    thoughts: list[str] = Field([], title="Thoughts", description="Analysis and thoughts on the user's request and how to best respond step by step. It can include the tools to be used, parameters to be passed, and any other relevant information.")
    tool_needed: bool = Field(False, title="Tool Needed", description="Whether a tool is needed to fulfill the user's request.")
    tool_call: list[ToolCall] = Field([], title="Tool Call", description="The tool call to be executed. It includes the tool name and parameters. It can include multiple tool calls if needed to fulfill the user's request. The tools will be executed in the order they are listed.")
    task_complete: bool = Field(False, title="Task Complete", description="Whether the entire task has been completed or if further actions are needed.")

def process_tool_call_information(tool_call_params: dict):
    parameters = {}
    for key, value in tool_call_params.items():
        if key == "path":
            parameters[key] = os.path.join(base_dir, value)
        else:
            parameters[key] = value
    return parameters

def process_tool_execution_information(tool_output: dict, tool_name: str): 
    tool_output_array = []
    if "type" not in tool_output:
        tool_output["type"] = "text"
    if "tool_name" not in tool_output:
        tool_output["name"] = tool_name
    if "purpose" not in tool_output:
        tool_output["purpose"] = "The purpose of this msg is to provide the output of the tool execution you requested."
    tool_output_array.append(tool_output)
    return tool_output_array


dotenv.load_dotenv()
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

# Agent setup for OpenAI GPT-4o-mini
client = instructor.from_openai(openai.OpenAI(api_key=OPEN_API_KEY), mode=instructor.Mode.JSON)
mini_engineer = Agent(client=client, model_name="gpt-4o-mini", system_prompt=system_prompt, tools=tools_list)

# Agent setup for llam 3.1 70b versatile
# TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
# client = openai.OpenAI(
#     base_url="https://api.together.xyz/v1",
#     api_key=TOGETHER_API_KEY,
# )
# client = instructor.from_openai(client , mode=instructor.Mode.JSON)
# mini_engineer = Agent(client=client, model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", system_prompt=system_prompt, tools=tools_list)

base_dir = "C:\\Users\\PMLS\\Desktop\\mini_engineer"

console = Console()

def execute_tool_calls(tool_calls):
    for tool_call in tool_calls:
        parameters = process_tool_call_information(tool_call.parameters)
        tool_execution = ToolExecution(tool_name=tool_call.tool_name, parameters=parameters)
        tool_response = mini_engineer.run_tool_sync(tool_execution)
        mini_engineer.store_message(f"The requested tool : {tool_call.tool_name} returned the following content: {str(tool_response.tool_output)}", role="user")
        console.print(Panel(f"🛠️ Tool Response: {tool_response.tool_output}", 
                            title=f"[bold cyan]{tool_call.tool_name}[/bold cyan]", 
                            border_style="red", 
                            expand=False))

def process_user_request(user_input):
    try:
        task_complete = False
        while not task_complete:
            mini_engineer_response = mini_engineer.run(response_model=MiniEngineerResponse, role="user", content=user_input)
            console.print(Panel(mini_engineer_response.summary, title="🤖 Mini Engineer", style="bold yellow", expand=False))
            
            for i, thought in enumerate(mini_engineer_response.thoughts, 1):
                console.print(Text(f"💭 Thought {i}: {thought}", style="italic magenta"))
            
            if mini_engineer_response.tool_needed:
                for i, tool_call in enumerate(mini_engineer_response.tool_call, 1):
                    console.print(Panel(f"🔧 Tool: {tool_call.tool_name}\n📊 Parameters: {tool_call.parameters}", 
                                        title=f"[bold cyan]Tool Call {i}[/bold cyan]", 
                                        border_style="cyan", 
                                        expand=False))
                execute_tool_calls(mini_engineer_response.tool_call)
                
                follow_up_response = mini_engineer.run(response_model=MiniEngineerResponse, role="user", content="Do you need to perform any more actions to complete the task? If so, what are they?")
                
                if follow_up_response.tool_needed:
                    user_input = "Please continue with the next steps to complete the task."
                    console.print("🔄 [bold blue]Continuing with next steps...[/bold blue]")
                else:
                    task_complete = True
            else:
                task_complete = True
        final_response = mini_engineer.run(response_model=MiniEngineerResponse, role="user", content="Summarize the actions taken and confirm if the task is now complete.")
        console.print(Panel(final_response.summary, title="🎉 Final Summary", style="bold green", expand=False))
    except Exception as e:
        console.print(Panel(f"An error occurred: {str(e)}", title="⚠️ Error", style="bold red", expand=False))

console.print(Panel.fit("Welcome to Mini Engineer! 🚀", style="bold green"))

while True:
    console.print("\n[bold green]Mini Engineer is ready to assist you. Type 'exit' to end the conversation.[/bold green]")
    user_input = console.input("[bold blue]User: [/bold blue]")
    if user_input.lower() == "exit":
        console.print(Panel.fit("Thank you for using Mini Engineer! Goodbye! 👋", style="bold green"))
        break
    process_user_request(user_input)