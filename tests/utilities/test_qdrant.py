# from qdrant_client import QdrantClient
# from qdrant_client.models import  PointStruct , SparseVectorParams , Modifier , SparseVector
# from fastembed import SparseTextEmbedding


# client = QdrantClient(url="http://localhost:6333")

# documents = [
#     "You should stay, study and sprint.",
#     "History can only prepare us to be surprised yet again.",
#     "The quick brown fox jumps over the lazy dog.",
# ]

# model = SparseTextEmbedding(model_name="Qdrant/bm25")
# embeddings = list(model.embed(documents))

# print(embeddings)

# client.delete_collection(collection_name="sparse-embeds")

# client.create_collection(
#     collection_name="sparse-embeds",
#     vectors_config={},
#     sparse_vectors_config={
#         "bm25": SparseVectorParams(
#             modifier=Modifier.IDF
#         )
#     }
    

# )

# point1 = PointStruct(
#     id=13,
#     vector={
#          "bm25": embeddings[0].as_object()},
#     payload={"text": "You should stay, study and sprint."}
# )

# point2 = PointStruct(
#     id=12,
#     vector={
#          "bm25": embeddings[1].as_object()
#     },
#     payload={"text": "History can only prepare us to be surprised yet again."}
# )

# point3 = PointStruct(
#     id=14,
#     vector={
#          "bm25": embeddings[2].as_object()
#     },
#     payload={"text": "The quick brown fox jumps over the lazy dog."}
# )

# client.upsert(collection_name="sparse-embeds", points=[point1], wait=False)
# client.upsert(collection_name="sparse-embeds", points=[point2], wait=False)
# client.upsert(collection_name="sparse-embeds", points=[point3], wait=True)


# Query = "History can only prepare us thy be surprised yet again."
# query_vector = list(model.embed([Query]))

# print(query_vector)
# query_sparse_vector = SparseVector(
#     values=query_vector[0].values.tolist(),
#     indices=query_vector[0].indices.tolist()
# )

# result = client.query_points(
#     collection_name="sparse-embeds",
#     query= query_sparse_vector,
#     using="bm25",
#     with_payload=True,
#     limit=1
# )

# print(result)

# operation_info = client.upsert(
#     collection_name="quickstart",
#     wait=True,
#     points=[
#         PointStruct(id=1, vector=[0.05, 0.61, 0.76, 0.74], payload={"city": "Berlin"}),
#         PointStruct(id=2, vector=[0.19, 0.81, 0.75, 0.11], payload={"city": "London"}),
#         PointStruct(id=3, vector=[0.36, 0.55, 0.47, 0.94], payload={"city": "Moscow"}),
#         PointStruct(id=4, vector=[0.18, 0.01, 0.85, 0.80], payload={"city": "New York"}),
#         PointStruct(id=5, vector=[0.24, 0.18, 0.22, 0.44], payload={"city": "Beijing"}),
#         PointStruct(id=6, vector=[0.35, 0.08, 0.11, 0.44], payload={"city": "Mumbai"}),
#     ],
# )

# print(operation_info)

# search_result = client.search(
#     collection_name="quickstart", query_vector=[0.2, 0.1, 0.9, 0.7], limit=3
# )

# print(search_result)


# search_result = client.search(
#     collection_name="quickstart",
#     query_vector=[0.2, 0.1, 0.9, 0.7],
#     query_filter=Filter(
#         must=[FieldCondition(key="city", match=MatchValue(value="London"))]
#     ),
#     with_payload=True,
#     limit=3
# )

# print(search_result)

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

from LightAgents.rag.vector_stores import query_qdrant_dense_vectors


client = QdrantClient("http://localhost:6333", timeout=600)

collection_name = "arxiv-chunks"
    
#     # Check if the collection exists
# collections = client.get_collections()
# if collection_name in [collection.name for collection in collections.collections]:
#         print(f"Collection '{collection_name}' already exists.")

# client.delete_collection(collection_name=collection_name)
# Initialize the dense encoder
dense_encoder = SentenceTransformer("all-MiniLM-L6-v2")
limit = 3
collection_name = "arxiv-chunks"

# Metadata for filters
filter_params = {
    'authors[]': "Satish Vitta",
}

query = "Discount on Electric Vehicles"

# Call the function with the provided parameters
results = query_qdrant_dense_vectors(query, dense_encoder,"all-MiniLM-L6-v2" ,limit, collection_name, None)

for result in results:
    print("Result: ")
    print(result)
    for scored_point in result[1]:
        print(scored_point.payload['title'])
