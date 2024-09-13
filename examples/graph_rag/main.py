

import sys
import os
import dotenv

# Add the root directory of your project to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from examples.graph_rag.pipeline import GraphRAGPipeline

dotenv.load_dotenv(".env")
OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

# Usage
pipeline = GraphRAGPipeline(api_key=OPEN_API_KEY, document_path="test.txt")
pipeline.convert_document_to_chunks()
pipeline.extract_entities_and_relationships()
pipeline.summarize_entities_and_relationships()
pipeline.construct_knowledge_graph()
pipeline.extract_communities()
pipeline.summarize_communities()
pipeline.relate_user_query_to_communities(user_query="What is the impact of climate change on agriculture?")
pipeline.save_knowledge_graph("knowledge_graph.gpickle")