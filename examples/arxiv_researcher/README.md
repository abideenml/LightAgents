# 🚀 Arxiv Researcher: Your AI-Powered Research Assistant 📚

Arxiv Researcher is an intelligent, interactive tool that supercharges your research process by leveraging the power of AI and the vast Arxiv database.

## ✨ Features

### 🤖 Overseer Mode
- Semantic verification of queries against existing research papers
- Intelligent addition of new relevant papers to the database
- Refined query processing for optimal results

### 🛠 Manual Mode
- Direct interaction with the Arxiv database
- Option to manually add new research papers
- Immediate query processing without overseer intervention

### 🧠 AI-Powered Analysis
- Utilizes OpenAI's GPT model for intelligent responses
- Semantic search capabilities for finding relevant research chunks
- Summarization and detailed explanations of research findings

### 📊 Vector Store Integration
- Efficient storage and retrieval of research paper information
- Utilizes Qdrant for high-performance vector similarity search

### 🎨 Rich Console Interface
- Colorful and emoji-filled console outputs
- Interactive prompts for user-friendly experience
- Clear presentation of research summaries and details

## 🏃‍♂️ Getting Started

### Prerequisites
- Python 3.7+
- Poetry
- Docker (for running Qdrant)

### Setup

1. Clone the repository:
   ```
   git clone  https://github.com/itsAdee/Light-Agents
   cd Light-Agents
   ```

2. Install dependencies using Poetry:
   ```
   poetry install
   ```

3. Set up Qdrant:
   ```
   docker pull qdrant/qdrant
   docker run -p 6333:6333 -p 6334:6334 ` -v ${PWD}/qdrant_storage:/qdrant/storage ` qdrant/qdrant
   ```

4. Set up your environment variables:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Script

1. Activate the Poetry shell:
   ```
   poetry shell
   ```

2. Run the script:
   ```
   python examples/arxiv_researcher/main.py
   ```
3. Follow the interactive prompts to choose between Overseer and Manual modes, input your queries, and explore research papers!

## 🙏 Acknowledgments

- OpenAI for their powerful GPT models
- Arxiv for providing access to a vast research database
- Qdrant for efficient vector similarity search capabilities

```