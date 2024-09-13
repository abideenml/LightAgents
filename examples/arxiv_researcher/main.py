import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import asyncio
import openai
import dotenv
from pydantic import BaseModel, Field
from typing import List, Optional
from rich.console import Console
from examples.arxiv_researcher.helper_functions import ChunkResponseList, query_arxiv
from examples.arxiv_researcher.tools import Tools, add_arxiv_ppr, get_processed_titles
import instructor
from LightAgents.agents.agent import ToolExecution
from LightAgents.agents.mem0_agent import MemoAgent
from LightAgents.systemprompt.systemprompt import SimpleSystemPromptGenerator


dotenv.load_dotenv()

console = Console()

class ArxivResearcherResponse(BaseModel):
    summary: str = Field(..., description="Summary of the answer to the user query")
    details: List[str] = Field(
        ..., description="Details of the answer to the user query"
    )

class OverseerResponse(BaseModel):
    isToolRequired: bool
    message: str
    refinedQuery: str
    toolName: Optional[str] = None
    parameters: Optional[dict] = None

def create_overseer(client, tool_list) -> MemoAgent:
    system_prompt_generator = SimpleSystemPromptGenerator(
        goal="Your an expert overseer , you will be given processed titles of the research papers and the user query. You can also add new research papers to the database by  making use  the of the *add_arxiv_ppr* tool provided to you. Semantically check if the present research papers will be enough to answer the user query or add more research papers to the database by making use of the tools provided.",
        background='''- Try to semantically check if the processed titles of the research papers will be enough to answer the user query.
- If your are absolutely certain that the present research papers are enough to answer the user query , you do not need call the **add_arxiv_ppr** tool.
- If you think that the present research papers are not enough to answer the user query , you must call the **add_arxiv_ppr** tool to add more research papers to the vector store.''',
    )
    system_prompt = system_prompt_generator.generate()
    overseer_agent = MemoAgent(
        client=client,
        model_name="gpt-3.5-turbo",
        system_prompt=system_prompt,
        tools=tool_list,
    )
    return overseer_agent

def query_overseer_agent(agent: MemoAgent, response_model, content: str, role: str, user_id: str = "overseer_10") -> str:
    response = agent.run(
        response_model=response_model, content=content, role=role, user_id=user_id
    )
    return response

def create_rag_researcher(client) -> MemoAgent:
    system_prompt_generator = SimpleSystemPromptGenerator(
        goal="You are an arxiv researcher , powered by OpenAI's GPT Model. With every user query you will also get the most similar chunks of research papers regarding the query ( it will include the score , chunk , title and abstract) see the scores to determine if you should use that chunk or not. Use  these chunks while responding to the user query.",
        background="""- While responding to the user query , make use of the chunks provided while crafting your answer.
- Your answer will be primarily based on the user query and the chunks provided to you.
- If you are not satisfied with the chunks provided to you , do not use them and respond to the user query based on your own knowledge.
- The chunks provided to you will contain scores , abstracts , titles and the chunks themselves. Use the scores to determine if you should use that chunk or not.""",
    )
    system_prompt = system_prompt_generator.generate()
    dotenv.load_dotenv(".env", override=True)

    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": "localhost",
                "port": 6333,
            },
        },
    }

    agent = MemoAgent(
        client=client,
        model_name="gpt-3.5-turbo",
        system_prompt=system_prompt,
        memo_config=config,
    )
    return agent

async def query_rag_researcher(agent: MemoAgent, response_model, role: str, content: str, user_id: str, chunks_list: ChunkResponseList = None) -> str:
    if chunks_list is not None:
        chunks_resp_list = []
        for chunk in chunks_list.chunks:
            chunk_dict = {
                chunk.title + "(" + str(chunk.score) + ")": {
                    "chunk": chunk.chunk,
                    "abstract": chunk.abstract,
                    "score": chunk.score,
                }
            }
            chunks_resp_list.append(chunk_dict)
        response = await agent.run_async(
            response_model=response_model,
            role=role,
            content=content,
            user_id=user_id,
            chunks_list=chunks_resp_list,
        )
    else:
        response = await agent.run_async(
            response_model=response_model, role=role, content=content, user_id=user_id
        )
    return response

async def execute_semantic_search_and_query_researcher(user_response, my_agent):
    chunk_responses = query_arxiv(user_response)
    chunk_response_list = ChunkResponseList(chunks=chunk_responses)
    response = await query_rag_researcher(
        agent=my_agent, response_model=ArxivResearcherResponse, role="user", content=user_response, user_id="researcher_10", chunks_list=chunk_response_list
    )
    console.print("Arxiv Researcher:", style="bold yellow")
    console.print(f"Summary of the answer to your query: {response.summary}", style="bold blue")
    console.print(f"Details of the answer to your query: {response.details}", style="italic magenta")

async def main():
    OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
    researcher_client = instructor.from_openai(
        openai.AsyncOpenAI(api_key=OPEN_API_KEY), mode=instructor.Mode.JSON
    )
    overseer_client = instructor.from_openai(
        openai.OpenAI(api_key=OPEN_API_KEY), mode=instructor.Mode.JSON
    )
    my_agent = create_rag_researcher(researcher_client)
    overseer_agent = create_overseer(overseer_client, Tools)

    console.print("🚀 Welcome to the Arxiv Researcher! 📚", style="bold green")

    while True:
        console.print("\n🔍 Choose your mode:", style="bold cyan")
        console.print("  [O] Overseer Mode")
        console.print("  [M] Manual Mode")
        console.print("  [E] Exit")
        mode = input("Enter your choice: ").lower()

        if mode == 'e':
            console.print("👋 Thank you for using Arxiv Researcher. Goodbye!", style="bold magenta")
            break

        if mode not in ['o', 'm']:
            console.print("❌ Invalid mode. Please try again.", style="bold red")
            continue

        if mode == 'o':
            console.print("\n🤖 Overseer Mode Activated!", style="bold yellow")
            user_response = input("🔎 What would you like to research? ")
            processed_titles = str(get_processed_titles())
            query_to_overseer = user_response + "*Processed Titles of Papers present in Vector Store:*" + processed_titles
            
            overseer_response = query_overseer_agent(
                overseer_agent, OverseerResponse, content=query_to_overseer, role="user"
            )
            console.print(f"🧠 Overseer's thoughts: {overseer_response.message}", style="italic cyan")

            if overseer_response.isToolRequired and overseer_response.toolName == "add_arxiv_ppr":
                console.print("📥 Adding new research papers...", style="bold blue")
                tool_execution = ToolExecution(
                    tool_name=overseer_response.toolName, parameters=overseer_response.parameters
                )
                add_arxiv_response = await overseer_agent.run_tool(tool_execution)
                console.print(f"✅ {add_arxiv_response.message}", style="green")
                
                overseer_response = query_overseer_agent(
                    overseer_agent,
                    OverseerResponse,
                    "Result of 'add_arxiv_ppr' tool: " + str(add_arxiv_response.success) + " details: " + str(add_arxiv_response.message),
                    "user"
                )

            console.print("🔬 Analyzing research papers...", style="bold blue")
            await execute_semantic_search_and_query_researcher(overseer_response.refinedQuery, my_agent)

        else:  # Manual Mode
            console.print("\n🛠 Manual Mode Activated!", style="bold yellow")
            if input("📚 Do you want to add research papers to the database? (y/n) ").lower() == 'y':
                research_paper_query = input("🔍 Enter your query for new papers: ")
                await add_arxiv_ppr(research_paper_query)
                console.print("✅ New papers added successfully!", style="green")

            user_response = input("🔎 What would you like to research? ")
            console.print("🔬 Analyzing research papers...", style="bold blue")
            await execute_semantic_search_and_query_researcher(user_response, my_agent)

        console.print("\n✨ Research complete! Need anything else?", style="bold green")

if __name__ == "__main__":
    asyncio.run(main())
