from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, Modifier, PointStruct , Prefetch,FusionQuery,Fusion,FieldCondition,MatchValue,Filter
import tqdm


def store_dense_sparse_vectors_in_qdrant(chunks,collection_name ,dense_vectors,sparse_vectors ,dense_storage_name, sparse_storage_name,metadata):
    client = QdrantClient("http://localhost:6333", timeout=600)
    
    collections = client.get_collections()
    if collection_name in [collection.name for collection in collections.collections]:
        print(f"Collection '{collection_name}' already exists.")
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                dense_storage_name: VectorParams(
                    size=len(dense_vectors[0]),
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
            sparse_storage_name: SparseVectorParams(
                    modifier=Modifier.IDF,
                )
            }
        )

    for i, (chunk_uuid, chunk) in enumerate(chunks):
        client.upload_points(collection_name,
                             [
                                 PointStruct(
                                     id=chunk_uuid,
                                     vector={
                                         dense_storage_name: dense_vectors[i],
                                         sparse_storage_name: sparse_vectors[i].as_object()
                                     },
                                     payload={
                                         "chunk": chunk,
                                         "metadata": metadata
                                     }
                                 )
                             ])


def store_dense_vectors_in_qdrant(chunks,collection_name ,dense_vectors ,dense_storage_name,metadata):
    client = QdrantClient("http://localhost:6333", timeout=600)
    collections = client.get_collections()
    print(len(dense_vectors))
    print(len(chunks))
    if collection_name in [collection.name for collection in collections.collections]:
        print(f"Collection '{collection_name}' already exists.")
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                dense_storage_name: VectorParams(
                    size=len(dense_vectors[0]),
                    distance=Distance.COSINE,
                ),
            }
        )
    batch_size = 8  # Define the batch size

    # Iterate over chunks in batches
    for batch_start in tqdm.tqdm(range(0, len(chunks), batch_size), total=len(chunks) // batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        batch_chunks = chunks[batch_start:batch_end]
        batch_dense_vectors = dense_vectors[batch_start:batch_end]

        print(f"Uploading chunks {batch_start + 1} to {batch_end} of {len(chunks)}")
        print(batch_chunks)
        print(batch_dense_vectors)

        points = [
            PointStruct(
                id=chunk_uuid,  
                vector={
                    dense_storage_name: batch_dense_vectors[i],
                },
                payload={
                    "chunk": chunk,
                    **{key: value for key, value in metadata.items()}
                }
            )
            for i, (chunk_uuid, chunk) in enumerate(batch_chunks)
        ]

        client.upload_points(collection_name, points)
        print(f"Uploaded chunks {batch_start + 1} to {batch_end} of {len(chunks)}")

    


def query_qdrant_dense_sparse_vectors(query,  dense_encoder, sparse_encoder ,dense_vector_storage_name,sparse_vector_storage_name,limit,collection_name="document_chunks"):
    client = QdrantClient("http://localhost:6333")
    dense_query = dense_encoder.encode(query)
    sparse_query = list(sparse_encoder.embed(query))
    prefetch = [
        Prefetch(
            query=dense_query,
            using=dense_vector_storage_name,
            limit=limit,
        ),
        Prefetch(
            query=sparse_query[0].as_object(),
            using=sparse_vector_storage_name,
            limit=limit,
        ),
    ]
    search_results = client.query_points(collection_name,prefetch=prefetch,
                                         query=FusionQuery(
            fusion=Fusion.RRF,
        ),
        with_payload=True,
        limit=limit,)
    return search_results



def query_qdrant_dense_vectors(query, dense_encoder,dense_vector_storage_name ,limit, collection_name="document_chunks", filter_params=None):
    client = QdrantClient("http://localhost:6333")
    encoded_query = dense_encoder.encode(query)
    
    # Create filter object if filter_params are provided
    query_filter = None
    if filter_params:
        must_conditions = [
            FieldCondition(
                key=key,
                match=MatchValue(value=value)
            ) for key, value in filter_params.items()
        ]
        query_filter = Filter(must=must_conditions)

    search_results = client.query_points(collection_name,
                                            query=encoded_query,
                                            using=dense_vector_storage_name,
                                            with_payload=True,
                                            limit=limit,
                                            query_filter=query_filter)
    
    return search_results


# async def query_qdrant_dense_vectors_async(query, dense_encoder,dense_vector_storage_name ,limit, collection_name="document_chunks", filter_params=None):
#     client = QdrantClient("http://localhost:6333")
#     encoded_query = dense_encoder.encode(query)
    
#     # Create filter object if filter_params are provided
#     query_filter = None
#     if filter_params:
#         must_conditions = [
#             FieldCondition(
#                 key=key,
#                 match=MatchValue(value=value)
#             ) for key, value in filter_params.items()
#         ]
#         query_filter = Filter(must=must_conditions)

#     search_results = await client.query_points(collection_name,
#                                             query=encoded_query,
#                                             using=dense_vector_storage_name,
#                                             with_payload=True,
#                                             limit=limit,
#                                             query_filter=query_filter)
    
#     return search_results
    