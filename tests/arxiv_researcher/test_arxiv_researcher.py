import os
import shutil
import sys
# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import re
from typing import List, Optional
import arxiv
import instructor
import openai
import pymupdf4llm
import aiohttp
import asyncio
import os
import json
import dotenv
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from LightAgents.agents.agent import ToolExecution
from LightAgents.agents.mem0_agent import MemoAgent
from LightAgents.systemprompt.systemprompt import SimpleSystemPromptGenerator
from LightAgents.tools.tools import Tool, ToolInputSchema, ToolOutputSchema
from LightAgents.rag.emdedders import create_dense_vector_via_sentence_transformer
from LightAgents.rag.splitters import generate_chunks_using_markdown_splitter_hf
from LightAgents.rag.utilities import append_descriptions_to_markdown, encode_images_in_folder, get_image_description_ollama_or_openai_async
from LightAgents.rag.vector_stores import  query_qdrant_dense_vectors, store_dense_vectors_in_qdrant

dotenv.load_dotenv()

# Pydantic models

class ChunkResponse(BaseModel):
    chunk : str
    title : str
    score : float
    abstract : str

class ChunkResponseList(BaseModel):
    chunks: List[ChunkResponse]

class Response(BaseModel):
    message: str

class ImageDescriptionResponse(BaseModel):
    message: str


class OverseerResponse(BaseModel):
    isToolRequired: bool
    message: str
    refinedQuery: str
    toolName: Optional[str] = None
    parameters: Optional[List[dict]] = None

# Tools

# Function to get processed titles
def get_processed_titles(filename = 'processed_titles.json') -> List[str]:
    file_path = os.path.join(os.getcwd(), "tests/arxiv_researcher/" + filename)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            processed_titles = json.load(f)
    else:
        processed_titles = []
    return  {"titles": processed_titles}

# Function to add reearch paper into qdrant
async def add_arxiv_ppr(query, req_limit=3) -> bool:
    # Load existing processed titles from JSON file
    if os.path.exists("processed_titles.json"):
        with open("processed_titles.json", "r") as f:
            processed_titles = json.load(f)
    else:
        processed_titles = []

    arxiv_client = arxiv.Client()
    arxiv_search = arxiv.Search(
        query=query,
        max_results=req_limit,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = arxiv_client.results(arxiv_search)
    tasks = [process_paper(result, processed_titles) for result in results]
    await asyncio.gather(*tasks)

    # Save processed titles to a JSON file
    with open("processed_titles.json", "w") as f:
        json.dump(processed_titles, f, indent=4)

    return True

class AddArxivPprInputSchema(ToolInputSchema):
    query: str = Field(..., description="The query to search for research papers")
    req_limit: int = Field(3, description="The number of results to process")

class AddArxivPprOutputSchema(ToolOutputSchema):
    result: bool = Field(..., description="Whether the operation was successful")

add_arxiv_ppr_tool = Tool(
    function=add_arxiv_ppr,
    input_schema=AddArxivPprInputSchema,
    output_schema=AddArxivPprOutputSchema,
    description="Add research papers from arXiv to the Qdrant database , based on the specified query , the function is async and can take some time to complete",
    type="async"
)

Tools = [add_arxiv_ppr_tool]

# Functions
async def fetch_pdf(session, url):
    async with session.get(url) as response:
        return await response.read()


async def process_paper(result, processed_titles):
    sanitized_title = re.sub(r'[<>:"/\\|?*]', '_', result.title)
    if result.title in processed_titles:
        print(f"Skipping already processed paper: {sanitized_title}")
        return

    async with aiohttp.ClientSession() as session:
        pdf_content = await fetch_pdf(session, result.pdf_url)
        temp_pdf_path = sanitized_title + ".pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_content)
        print(f"Processing paper: {sanitized_title}")    
        images_dir = sanitized_title + "_images"
        content = pymupdf4llm.to_markdown(temp_pdf_path, page_chunks=True, image_format="jpeg", write_images=True, image_path=images_dir)
        
        # Encode images
        encoded_images = encode_images_in_folder(images_dir)
        OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
        client = instructor.from_openai(openai.AsyncOpenAI(api_key=OPEN_API_KEY), mode=instructor.Mode.JSON)
        
        image_descriptions = {image: await get_image_description_ollama_or_openai_async(encoded_image, client) for image, encoded_image in encoded_images.items()}

        modified_content = []

        for page in content:
            modified_content.append(append_descriptions_to_markdown(page['text'], image_descriptions, images_dir))
        # Generate chunks
        max_tokens = 250
        list_of_chunks = []
        for page in modified_content:
            chunks = list(generate_chunks_using_markdown_splitter_hf(page, max_tokens, sanitized_title))
            list_of_chunks.extend(chunks)

        print(f"Number of chunks: {len(list_of_chunks)}")

        dense_vectors = [create_dense_vector_via_sentence_transformer(chunk) for _, chunk in list_of_chunks]

        # Assuming result.authors is a list of arxiv.Result.Author objects
        author_names = [author.name for author in result.authors]


        metadata_regarding_paper = {
            "title": sanitized_title,
            "abstract": result.summary,
            "doi": result.doi,
            "pdf_url": result.pdf_url,
            "authors": author_names
        }

        dense_vector_storage = "all-MiniLM-L6-v2"

        store_dense_vectors_in_qdrant(list_of_chunks, "arxiv-chunks", dense_vectors,  dense_vector_storage,  metadata_regarding_paper)

        # Save the title to the processed_titles list
        processed_titles.append(result.title)

        # remove the temp pdf file
        os.remove(temp_pdf_path)
        # remove the images directory
        shutil.rmtree(images_dir)

        print(f"Done processing paper- {result.title}")

async def init_arxiv():
    # Load existing processed titles from JSON file
    if os.path.exists("processed_titles.json"):
        with open("processed_titles.json", "r") as f:
            processed_titles = json.load(f)
    else:
        processed_titles = []

    arxiv_client = arxiv.Client()
    arxiv_search = arxiv.Search(
        query="Electric vehicle",
        max_results=2,
        sort_by=arxiv.SortCriterion.Relevance
    )

    results = arxiv_client.results(arxiv_search)
    tasks = [process_paper(result, processed_titles) for result in results]
    await asyncio.gather(*tasks)

    # Save processed titles to a JSON file
    with open("processed_titles.json", "w") as f:
        json.dump(processed_titles, f, indent=4)


def query_arxiv(query , req_limit=3) -> List[ChunkResponse]:
    dense_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    collection_name = "arxiv-chunks"
    results =  query_qdrant_dense_vectors(query, dense_encoder,"all-MiniLM-L6-v2" ,req_limit, collection_name, None)
    chunk_responses = []
    for result in results:
       for scored_point in result[1]:
            chunk_responses.append(ChunkResponse(chunk=scored_point.payload['chunk'], title=scored_point.payload['title'], score=scored_point.score, abstract=scored_point.payload['abstract']))
    
    return chunk_responses


def create_overseer(client , tool_list) -> MemoAgent:
    ## Create simple system prompt generator
    system_prompt_generator = SimpleSystemPromptGenerator(goal="Verify by the titles of the research paper if they would be adequate to answer the query or not.If you are  sure  they wont be enough , make use of Tool given to you to add more research papers.Also refine the query given to you and remove unnecessary commands from it , so the query can be used to search for relevant chunks of research papers.",
                                                      background="You are an overseer of the research papers database. You will be given titles of the research papers currently stored in the database along with the user query.You can give the also add new research papers by giving appropriate function name and paramaters of the tools provided to you.Semantically check if the present research papers will be enough to answer the user query or add more research papers to the database by making use of the tools provided. Give the go ahead if you are absolutely sure that the present research papers are enough to answer the user query.")
    
    agent = MemoAgent(client=client, model_name="gpt-3.5-turbo", system_prompt_generator=system_prompt_generator, tools=tool_list)

    return agent

def query_overseer(agent: MemoAgent, response_model, user_query: str , user_id: str = "overseer_10") -> str:
    response = agent.run(response_model=response_model, user_message=user_query ,user_id= user_id)
    return response

def create_rag_researcher(client) -> MemoAgent:
    ## Create simple system prompt generator
    system_prompt_generator = SimpleSystemPromptGenerator(goal="Assist the assistant in academic research by providing information or performing actions related to research papers.",
                                                      background="You are a arxiv research assistant, along with user query you will also get most similar chunks of research papers regarding the query ( it will include the score , chunk , title and abstract) see the scores to determine if you should use that chunk or not. Use  these chunks while responding to the user query.")
    

    dotenv.load_dotenv(".env" , override=True)

    config = {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": "localhost",
                "port": 6333,
            }
        },
    }

    agent = MemoAgent(client=client, model_name="gpt-3.5-turbo", system_prompt_generator=system_prompt_generator, memo_config=config)

    return agent

async def query_rag_researcher(agent: MemoAgent, response_model, user_query: str, user_id: str, chunks_list: ChunkResponseList = None) -> str:
    # Check if there are any chunks and if so create a dictionary and append to async_run function of agent
    if chunks_list is not None:
        chunks_resp_list = []
        for chunk in chunks_list.chunks:
            chunk_dict = {chunk.title + str(chunk.score): {"chunk": chunk.chunk, "abstract": chunk.abstract, "score": chunk.score}}
            chunks_resp_list.append(chunk_dict)
        response = await agent.run_async(response_model=response_model, user_message=user_query, user_id=user_id, chunks_list=chunks_resp_list)
    else:
        response = await agent.run_async(response_model=response_model, user_message=user_query, user_id=user_id)
    
    return response.message

async def execute_arxiv_search(user_response, my_agent):
    chunk_responses = query_arxiv(user_response)
    chunk_response_list = ChunkResponseList(chunks=chunk_responses)            
    response = await query_rag_researcher(my_agent, Response, user_response, "researcher_1", chunk_response_list)
    print(f"Response from the arxiv researcher: {response}")

async def main():
    OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
    researcher_client = instructor.from_openai(openai.AsyncOpenAI(api_key=OPEN_API_KEY), mode=instructor.Mode.JSON)
    overseer_client = instructor.from_openai(openai.OpenAI(api_key=OPEN_API_KEY), mode=instructor.Mode.JSON)
    my_agent = create_rag_researcher(researcher_client)
    overseer_agent = create_overseer(overseer_client, Tools)
    is_prev_query_answered = True
    while True:
        mode = input("Enter o to use the overseer or m to use the manual mode. You can also type 'exit' to quit.:")
        if mode == "o":
            if is_prev_query_answered:
                print("Enter your query related to research papers or type 'exit' to quit.")
                user_response = input("What is your query?:")
                if user_response.lower() == "exit":
                    exit()
            # get the processed titles
            is_prev_query_answered = False
            processed_titles = str(get_processed_titles())
            print(f"Processed Titles: {processed_titles}")
            query_to_overseer = user_response + "Processed Titles: " + processed_titles
            # query the overseer
            overseer_response = query_overseer(overseer_agent, OverseerResponse, query_to_overseer)
            print(f"Overseer: {overseer_response}")
            if (overseer_response.isToolRequired):
                if overseer_response.toolName == "add_arxiv_ppr":
                    parameters_dict = {}
                    #extract the parameters for list of dictionaries
                    for parameter in overseer_response.parameters:
                        for key, value in parameter.items():
                            parameters_dict[key] = value
                    tool_execution = ToolExecution(tool_name=overseer_response.toolName, parameters=parameters_dict)
                    add_arxiv_response = await overseer_agent.run_tool(tool_execution)
                    # give the response to the overseer
                    overseer_response = query_overseer(overseer_agent, OverseerResponse, "Result of adding research papers: " + str(add_arxiv_response.success))
                    print(f"Overseer: {overseer_response.message}")
                    is_prev_query_answered = True
                    await execute_arxiv_search(overseer_response.refinedQuery, my_agent)
                else :
                    print("Tool not found")
            else:
                print(f"Overseer: {overseer_response.message}")
                is_prev_query_answered = True
                await execute_arxiv_search(overseer_response.refinedQuery, my_agent)
        elif mode == "m":
            print("Do you want to add research papers to the database? (y/n)")
            if input().lower() == "y":
                research_paper_query = input("Enter the query to search for research papers:")
                await add_arxiv_ppr(research_paper_query)
            print("Enter your query related to research papers or type 'exit' to quit.")
            user_response = input("What is your query?:")
            if user_response.lower() == "exit":
                exit()
            await execute_arxiv_search(user_response, my_agent)
        elif mode == "exit":
            exit()
        else:
            print("Invalid mode. Please enter 'o' or 'm'.")

if __name__ == "__main__":
    asyncio.run(main())