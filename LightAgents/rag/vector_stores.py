from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, SparseVectorParams, Modifier, PointStruct , Prefetch,FusionQuery,Fusion,FieldCondition,MatchValue,Filter
import tqdm


def store_dense_sparse_vectors_in_qdrant(chunks,collection_name ,dense_vectors,sparse_vectors ,dense_storage_name, sparse_storage_name,metadata, url="http://localhost:6333"):
    client = QdrantClient(url, timeout=600)
    
    # Check if the collection already exists
    collections = client.get_collections()
    # If the collection already exists, print a message
    if collection_name in [collection.name for collection in collections.collections]:
        print(f"Collection '{collection_name}' already exists.")
    # If the collection does not exist, create a new collection
    else:
        # Create a new collection with the specified name and vector configurations
        client.create_collection(
            collection_name=collection_name,
            # Define the vector configurations for the dense and sparse vectors
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
    # Define the batch size
    for i, (chunk_uuid, chunk) in enumerate(chunks):
        # Upload the dense and sparse vectors to the collection
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


def store_dense_vectors_in_qdrant(chunks,collection_name ,dense_vectors ,dense_storage_name,metadata,url="http://localhost:6333"):
    client = QdrantClient(url, timeout=600)
    collections = client.get_collections()
    # If the collection already exists, print a message
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

        # Create a list of PointStruct objects to upload to the collection
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

    


def query_qdrant_dense_sparse_vectors(query,  dense_encoder, sparse_encoder ,dense_vector_storage_name,sparse_vector_storage_name,limit,collection_name="document_chunks", url="http://localhost:6333"):
    client = QdrantClient(url)
    # Encode the query using the dense and sparse encoders
    dense_query = dense_encoder.encode(query)
    sparse_query = list(sparse_encoder.embed(query))
    # Create a Prefetch object to specify the dense and sparse vector storage names
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
    # Query the collection using the dense and sparse vectors
    search_results = client.query_points(collection_name,prefetch=prefetch,
                                         query=FusionQuery(
            fusion=Fusion.RRF,
        ),
        with_payload=True,
        limit=limit,)
    return search_results



def query_qdrant_dense_vectors(query, dense_encoder,dense_vector_storage_name ,limit, collection_name="document_chunks", filter_params=None ,url="http://localhost:6333"):
    client = QdrantClient(url)
    # Encode the query using the dense encoder
    encoded_query = dense_encoder.encode(query)
    # Create a Filter object to specify the filter parameters
    query_filter = None
    if filter_params:
        must_conditions = [
            FieldCondition(
                key=key,
                match=MatchValue(value=value)
            ) for key, value in filter_params.items()
        ]
        query_filter = Filter(must=must_conditions)

   # Query the collection using the dense vectors
    search_results = client.query_points(collection_name,
                                            query=encoded_query,
                                            using=dense_vector_storage_name,
                                            with_payload=True,
                                            limit=limit,
                                            query_filter=query_filter)
    
    return search_results

