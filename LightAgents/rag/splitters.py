from semantic_text_splitter import MarkdownSplitter
from tokenizers import Tokenizer

from LightAgents.rag.utilities import create_unique_id


# This function is used to generate chunks from the input text using markdown splitter , it makes use of the huggingface tokenizer
def generate_chunks_using_markdown_splitter_hf(md_text, max_tokens, document_name , tokenizer = "BAAI/bge-en-icl"):
    tokenizer = Tokenizer.from_pretrained(tokenizer)
    splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer, max_tokens, overlap=50)
    chunks = splitter.chunks(md_text)
    for i, chunk in enumerate(chunks):
        chunk_name = f"{document_name}_chunk_{i+1}"
        generated_uuid = create_unique_id(chunk_name)
        yield generated_uuid, chunk

# This function is used to generate chunks from the input text using markdown splitter , it makes use of characters to split the text
def generate_chunks_using_markdown_splitter(md_text, max_tokens, overlap,document_name):
    splitter = MarkdownSplitter(max_tokens, overlap=overlap)
    chunks = splitter.chunks(md_text)
    for i, chunk in enumerate(chunks):
        chunk_name = f"{document_name}_chunk_{i+1}"
        generated_uuid = create_unique_id(chunk_name)
        yield generated_uuid, chunk