from fastembed import SparseTextEmbedding
from sentence_transformers import SentenceTransformer

# This function is used to create dense vector representation of the input text chunk
def create_dense_vector_via_sentence_transformer(chunk , model_name = "all-MiniLM-L6-v2"):
    dense_encoder = SentenceTransformer(model_name)
    return dense_encoder.encode(chunk)

# This function is used to create sparse vector representation of the input text chunk
def create_sparse_vector_via_sparse_text_embedding_lib(chunk , model_name = "Qdrant/bm25"):
    model = SparseTextEmbedding(model_name)
    return list(model.embed(chunk))