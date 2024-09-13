import os
import re
import shutil
from typing import List
import aiohttp
import instructor
import openai
from pydantic import BaseModel
import pymupdf4llm
from sentence_transformers import SentenceTransformer

from LightAgents.rag.emdedders import create_dense_vector_via_sentence_transformer
from LightAgents.rag.splitters import generate_chunks_using_markdown_splitter
from LightAgents.rag.utilities import (
    append_descriptions_to_markdown,
    encode_images_in_folder,
    get_image_description_ollama_or_openai_async,
)
from LightAgents.rag.vector_stores import (
    query_qdrant_dense_vectors,
    store_dense_vectors_in_qdrant,
)


class ChunkResponse(BaseModel):
    chunk: str
    title: str
    score: float
    abstract: str


class ChunkResponseList(BaseModel):
    chunks: List[ChunkResponse]


async def fetch_pdf(session, url):
    async with session.get(url) as response:
        return await response.read()


async def process_paper(result, processed_titles):
    sanitized_title = re.sub(r'[<>:"/\\|?*]', "_", result.title)
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
        content = pymupdf4llm.to_markdown(
            temp_pdf_path,
            page_chunks=True,
            image_format="jpeg",
            write_images=True,
            image_path=images_dir,
        )

        # Encode images
        encoded_images = encode_images_in_folder(images_dir)
        OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
        client = instructor.from_openai(
            openai.AsyncOpenAI(api_key=OPEN_API_KEY), mode=instructor.Mode.JSON
        )

        image_descriptions = {
            image: await get_image_description_ollama_or_openai_async(
                encoded_image, client
            )
            for image, encoded_image in encoded_images.items()
        }

        modified_content = []

        for page in content:
            modified_content.append(
                append_descriptions_to_markdown(
                    page["text"], image_descriptions, images_dir
                )
            )
        # Generate chunks
        max_tokens = 250
        list_of_chunks = []
        for page in modified_content:
            chunks = list(
                generate_chunks_using_markdown_splitter(
                    page, max_tokens, 20, sanitized_title
                )
            )
            list_of_chunks.extend(chunks)

        dense_vectors = [
            create_dense_vector_via_sentence_transformer(chunk)
            for _, chunk in list_of_chunks
        ]

        # Assuming result.authors is a list of arxiv.Result.Author objects
        author_names = [author.name for author in result.authors]

        metadata_regarding_paper = {
            "title": sanitized_title,
            "abstract": result.summary,
            "doi": result.doi,
            "pdf_url": result.pdf_url,
            "authors": author_names,
        }

        dense_vector_storage = "all-MiniLM-L6-v2"

        store_dense_vectors_in_qdrant(
            list_of_chunks,
            "arxiv-chunks",
            dense_vectors,
            dense_vector_storage,
            metadata_regarding_paper,
        )

        # Save the title to the processed_titles list
        processed_titles.append(result.title)

        # remove the temp pdf file
        os.remove(temp_pdf_path)
        # remove the images directory
        shutil.rmtree(images_dir)

        print(f"Done processing paper- {result.title}")


def query_arxiv(query, req_limit=3) -> List[ChunkResponse]:
    dense_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    collection_name = "arxiv-chunks"
    results = query_qdrant_dense_vectors(
        query, dense_encoder, "all-MiniLM-L6-v2", req_limit, collection_name, None
    )
    chunk_responses = []
    for result in results:
        for scored_point in result[1]:
            chunk_responses.append(
                ChunkResponse(
                    chunk=scored_point.payload["chunk"],
                    title=scored_point.payload["title"],
                    score=scored_point.score,
                    abstract=scored_point.payload["abstract"],
                )
            )

    return chunk_responses
