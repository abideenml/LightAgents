from semantic_text_splitter import MarkdownSplitter
from tokenizers import Tokenizer

from LightAgents.rag.utilities import create_unique_id


def generate_chunks_using_markdown_splitter_hf(md_text, max_tokens, document_name , tokenizer = "BAAI/bge-en-icl"):
    tokenizer = Tokenizer.from_pretrained(tokenizer)
    splitter = MarkdownSplitter.from_huggingface_tokenizer(tokenizer, max_tokens, overlap=50)
    chunks = splitter.chunks(md_text)
    no_of_chunks = len(chunks)
    for i, chunk in enumerate(chunks):
        chunk_name = f"{document_name}_chunk_{i+1}_{no_of_chunks}_{len(chunk)}_{chunk[:10]}"
        generated_uuid = create_unique_id(chunk_name)
        yield generated_uuid, chunk