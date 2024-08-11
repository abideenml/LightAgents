
import os
import instructor
import openai
import dotenv
from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer
from LightAgents.rag.emdedders import create_dense_vector_via_sentence_transformer, create_sparse_vector_via_sparse_text_embedding_lib
from LightAgents.rag.extractors import collect_metadata
from LightAgents.rag.splitters import generate_chunks_using_markdown_splitter_hf
from LightAgents.rag.utilities import read_markdown_file
from LightAgents.rag.vector_stores import query_qdrant_dense_sparse_vectors, store_dense_sparse_vectors_in_qdrant

dotenv.load_dotenv()

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")
client = instructor.from_openai(openai.OpenAI(api_key=OPEN_API_KEY), mode=instructor.Mode.JSON)

pdf_path = "documents/human-nutrition-text.pdf"
output_md_path = "documents/output42-60.md"
image_output_folder = "images"
pages = list(range(42, 61))  

# md_text = extract_text_and_images(pdf_path, image_output_folder, pages)
# encoded_images = encode_images_in_folder(image_output_folder)
# image_descriptions = {image: get_image_description(client, encoded_image) for image, encoded_image in encoded_images.items()}
# updated_md_text = append_descriptions_to_markdown(md_text, image_descriptions, image_output_folder)

updated_md_text = read_markdown_file(output_md_path)
max_tokens = 500
chunks = list(generate_chunks_using_markdown_splitter_hf(updated_md_text, max_tokens, "document_name"))

metadata = collect_metadata(pdf_path)
dense_vectors = [create_dense_vector_via_sentence_transformer(chunk) for _, chunk in chunks]
sparse_vectors = create_sparse_vector_via_sparse_text_embedding_lib([chunk for _, chunk in chunks])

dense_encoder = SentenceTransformer("all-MiniLM-L6-v2")
sparse_encoder = SparseTextEmbedding(model_name="Qdrant/bm25")
dense_vector_storage_name = "all-MiniLM-L6-v2"
sparse_vector_storage_name = "bm25"
store_dense_sparse_vectors_in_qdrant(chunks,"chunkers" ,dense_vectors, sparse_vectors ,dense_vector_storage_name,sparse_vector_storage_name, metadata)

# Example query
query = "What parcentage of American purchase supplements?"
results = query_qdrant_dense_sparse_vectors(query, dense_encoder,sparse_encoder,dense_vector_storage_name,sparse_vector_storage_name,limit=2)
for result in results:
    print(result)