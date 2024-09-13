## Graph Rag


### Overview
Graph-based Retrieval-Augmented Generation (Graph-RAG) is an advanced framework that enhances both knowledge retrieval and answer generation in AI systems. By combining retrieval-augmented generation with graph-based data structures, Graph-RAG enables more efficient and context-aware information retrieval from large datasets.

In this approach, the process begins by chunking a document source and extracting entities and relationships from each chunk to construct a knowledge graph. Communities are then identified within the knowledge graph, and each community is summarized. These summaries allow the system to relate user queries to relevant communities, ensuring more accurate and contextually rich responses.

### Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/itsAdee/Light-Agents
   cd Light-Agents
   ```

2. Install dependencies using Poetry:
   ```sh
   poetry install
   ```

3. Set up your environment variables:
   Create a `.env` file in the project root and add your Together API key:
   ```sh
   TOGETHER_API_KEY=your_api_key_here
   ```

### Running the Script

1. Activate the Poetry shell:
   ```sh
   poetry shell
   ```

2. Run the script:
   ```sh
   python examples/graph_rag/main.py
   ```

