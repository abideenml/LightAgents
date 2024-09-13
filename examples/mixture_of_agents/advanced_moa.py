import asyncio
import os
import sys

import openai
from rich.console import Console

# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from LightAgents.agents.agent import Agent
import instructor
import dotenv

# Load the API key from the .env file
dotenv.load_dotenv('.env')
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Initialize the OpenAI client (together API makes use of the OpenAI SDK under the hood)
client = openai.AsyncOpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY,
)
# Convert the OpenAI client to an instructor client
client = instructor.from_openai(client)

layers = 3

# Reference models for the mixture of agents
reference_models = [
    "Qwen/Qwen2-72B-Instruct",
    "Qwen/Qwen1.5-72B-Chat",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    "databricks/dbrx-instruct",
]

# Aggregator model for the mixture of agents
aggregator_model = "mistralai/Mixtral-8x22B-Instruct-v0.1"
aggreagator_system_prompt = """You have been provided with a set of responses from various open-source models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability."""
proposer_system_prompt = """You are tasked with generating a response to the user query. Your response should be accurate, informative, and well-structured. Your response should offer a unique perspective or additional insights to enhance the overall quality of the answer."""

def get_final_aggregator_system_prompt(responses):
    return aggreagator_system_prompt + "\n" + "\n".join([f"{i+1}. {str(element)}" for i, element in enumerate(responses)])

user_query = "What are the benefits of eating healthy?"

console = Console()

async def get_response_from_proposer_model(model_name):
    console.print(f"[bold green]Querying model:[/bold green] {model_name}")
    agent = Agent(client=client, model_name=model_name, system_prompt=proposer_system_prompt)
    response = await agent.run_async(role="user", content=user_query , response_model=None)
    console.print(f"[bold blue]Response from {model_name}:[/bold blue] {response.choices[0].message.content}")
    return response.choices[0].message.content

async def get_response_from_aggregator_model(responses):
    modified_aggregator_system_prompt = get_final_aggregator_system_prompt(responses)
    console.print("[bold green]Aggregating responses...[/bold green]")
    agg_agent = Agent(client=client, model_name=aggregator_model, system_prompt=modified_aggregator_system_prompt)
    agg_response = await agg_agent.run_async(role="user", content=user_query , response_model=None)
    console.print(f"[bold blue]Aggregator Response:[/bold blue] {agg_response.choices[0].message.content}")
    return agg_response.choices[0].message.content

async def run_llm(reference_models=reference_models):
    console.print("[bold yellow]Running proposer models...[/bold yellow]")
    responses = await asyncio.gather(*[get_response_from_proposer_model(model_name) for model_name in reference_models])
    console.print("[bold yellow]Proposer models completed.[/bold yellow]")
    for _ in range(1, layers-1):
        console.print(f"[bold yellow]Running Inner layer {_} of the MOA...[/bold yellow]")
        responses = await asyncio.gather(*[get_response_from_aggregator_model(responses) for _ in range(layers)])
    return responses

async def main():
    console.print("[bold magenta]Starting the MOA procedure...[/bold magenta]")
    responses = await run_llm()
    modified_aggregator_system_prompt = get_final_aggregator_system_prompt(responses)
    final_agg_agent = Agent(client=client, model_name=aggregator_model, system_prompt=modified_aggregator_system_prompt)
    agg_response = await final_agg_agent.run_async(role="user", content=user_query , response_model=None)
    console.print(f"[bold red]Final Aggregator Response:[/bold red] {agg_response.choices[0].message.content}")

console.print("[bold cyan]Running main function...[/bold cyan]")
asyncio.run(main())
console.print("[bold cyan]Main function completed.[/bold cyan]")