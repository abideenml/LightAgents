# Light Agent
A modular toolkit to craft AI agents and RAG Pipelines.

```bash

## Development log

2024-08-9
----------
Made Changes in the code to log LLM traces to  BrainTrust and to retrieve Input and Completion Tokens from LLMS. Still need to make changes to save those tokens and use them for later.

2024-08-05
----------
Modified the  test "Arxiv Researcher" . The overseer agent decides by viewing the processed titles if there is need to add more research papers to properly answer the user query , after that, the Arxiv Researcher Agent  properly answers the user query using the most appropriate RAG chunks given to it via similarity search. I still need to update the pipeline to save abstract with the titles as well , so the overseer agent has a better idea of the research paper.

2024-08-03
----------
Finished the document processing pipeline test and started to implement "Arxiv Researcher" ,a test pipeline , to utilize the Rag tools incombination with mem0 agent.

2024-08-02
----------
Created functions to save make dense embeddings via hugging face models and sparse vectors using fastembed library and to store these vectors in qdrant. Also made functions to implement hybrid search or hybrid search.

2024-08-01
----------
Created functions to help extract information from any document , process them into chunks , utilized gpt-4o model to gather information about images present in the documents and insert this information instead of the image url.

2024-07-28
----------
Implement various functions to calculate tokens through models available at hugging face and tiktoken.

2024-08-23
----------
Used "https://github.com/mem0ai/mem0" to create a new agent class "mem0-agent" which utilizes the dynamic memory provided by mem0.

2024-08-20
----------
created memory buffer and memory storage functions for the Agent class.

2024-08-21
----------
integrated instructor  into the agent class to get structured outputs.

2024-08-16
----------
Create an agent class and a tool class to create an AI agent which can utilize functions , and call them.

```


## Setup


Here are the steps to use this Respository!:

1. `git clone https://github.com/itsAdee/Light-Agents`
2. Navigate into project directory `cd Light-Agents`
3. Install poetry via `pip install poetry .`
4. Run the command `poetry install` to install all the required packages.
5. Run `poetry shell` to created a new venv with all the required packages.

That's it!<br/>

## Features

Here are the core features of Light Agents

#### ✅ Supported LLMS
* Open Source Models via Ollama
* OpenAI Models
* Anthropic Models
* Gemini Models
* Groq Models

#### ✅ Capabilities of Light Agents
* Get Structured Outputs. 
* Functions executions through *tool* abstraction.
* Schema validation for tools
* Executing the appropriate tool directly through the agent class

#### ✅ Rag Tools
* Process any document (pdfs,docs) etc
* Make chunks using characters limits or token limits
* Make dense embeddings for the chunks using hugging face models.
* Make sparse embeddings for the chunks by using models available through fastembed library.
* Storing vectors and querying via qdrant.

## 🤞 Todos

* Creating examples
* making agents classes more modular.



## 🦋 Citation

If you find this code useful, please cite the following:

```
@misc{Zain2024Light.Agents,
  author = {Zain, Abideen , Adeel,Qureshi },
  title = {Light.Agents},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/itsAdee/Light-Agents}},
}
```

## Connect with me

If you'd love to have some more AI-related content in your life :nerd_face:, consider:

* Connect and reach me on [LinkedIn](https://www.linkedin.com/in/zaiinulabideen/) and [Twitter](https://twitter.com/zaynismm)
* Follow me on 📚 [Medium](https://medium.com/@zaiinn440)
* Check out my 🤗 [HuggingFace](https://huggingface.co/abideen)