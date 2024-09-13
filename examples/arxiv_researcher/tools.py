import asyncio
import json
import os
import arxiv
from typing import List

from pydantic import Field

from LightAgents.tools.helper_classes import ToolInputSchema, ToolOutputSchema
from LightAgents.tools.tools import Tool
from examples.arxiv_researcher.helper_functions import process_paper


def get_processed_titles(filename="processed_titles.json") -> List[str]:
    file_path = os.path.join(os.getcwd(), "examples/arxiv_researcher/" + filename)
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            processed_titles = json.load(f)
    else:
        processed_titles = []
    return {"titles": processed_titles}


# Function to add reearch paper into qdrant
async def add_arxiv_ppr(query, req_limit=3) -> bool:
    # Load existing processed titles from JSON file
    if os.path.exists("examples/arxiv_researcher/processed_titles.json"):
        with open("examples/arxiv_researcher/processed_titles.json", "r") as f:
            processed_titles = json.load(f)
    else:
        processed_titles = []

    arxiv_client = arxiv.Client()
    arxiv_search = arxiv.Search(
        query=query, max_results=req_limit, sort_by=arxiv.SortCriterion.Relevance
    )

    results = arxiv_client.results(arxiv_search)
    tasks = [process_paper(result, processed_titles) for result in results]
    await asyncio.gather(*tasks)

    # Save processed titles to a JSON file
    with open("examples/arxiv_researcher/processed_titles.json", "w") as f:
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
    type="async",
)

Tools = [add_arxiv_ppr_tool]
