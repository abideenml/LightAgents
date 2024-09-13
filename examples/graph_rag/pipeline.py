import os
import pickle
import openai
import networkx as nx
from rich.console import Console
from pydantic import BaseModel
from cdlib import algorithms
import pymupdf4llm
from semantic_text_splitter import TextSplitter
import dotenv
import instructor

from LightAgents.agents.agent import Agent
dotenv.load_dotenv('.env')

OPEN_API_KEY = os.getenv("OPENAI_API_KEY")

class GraphRAGPipeline:
    def __init__(self, api_key, document_path, model_name="gpt-4o-mini"):
        dotenv.load_dotenv('.env')
        self.api_key = api_key
        self.document_path = document_path
        self.model_name = model_name
        self.client = instructor.from_openai(openai.OpenAI(api_key=self.api_key))
        self.console = Console()
        self.G = nx.Graph()
        self.chunks = []
        self.elements = []
        self.summaries = []
        self.communities = []
        self.community_summaries = []

    def convert_document_to_chunks(self):
        md_text = pymupdf4llm.to_markdown(
            self.document_path,
            image_format="jpeg",
            write_images=True,
        )
        splitter = TextSplitter(capacity=600, overlap=100, trim=False)
        self.chunks = splitter.chunks(md_text)

    def extract_entities_and_relationships(self):
        class ExtractEntitiesAndRelationship(BaseModel):
            entities: list[str]
            relationships: list[str]

        extractor_system_prompt = "Extract the entities and relationships from the text given by the user. Be comprehensive and provide all the entities and relationships present in the text. You are to provide the entities and relationships as a list of strings in JSON format."
        extractor_agent = Agent(
            client=self.client, model_name=self.model_name, system_prompt=extractor_system_prompt
        )
        for index, chunk in enumerate(self.chunks):
            self.console.print(f"[bold green]Processing chunk {index + 1}[/bold green]")
            response = extractor_agent.run(
                role="user", content=chunk, response_model=ExtractEntitiesAndRelationship
            )
            self.elements.append(response)

    def summarize_entities_and_relationships(self):
        class KnowledgeGraphEdge(BaseModel):
            source: str
            target: str
            relationship: str

        class ExtractRelationships(BaseModel):
            entities: list[str]
            relationships: list[KnowledgeGraphEdge]

        summarizor_system_prompt = "Your job is to summarize the entities and relationships provided to you in a structured format. You will be given list[str] of entities and list[str] relationships and you will transform them into a structured format in JSON."
        summarizor_agent = Agent(client=self.client, model_name=self.model_name, system_prompt=summarizor_system_prompt)
        for index, element in enumerate(self.elements):
            self.console.print(f"[bold blue]Processing element {index + 1}[/bold blue]")
            entities = element.entities
            relationships = element.relationships
            prompt = (
                f"Relationships: {str(relationships)}\nEntities: {str(entities)}"
            )
            response = summarizor_agent.run(
                role="user", content=prompt, response_model=ExtractRelationships
            )
            self.summaries.append(response)
            self.console.print(f"[bold yellow]Summary {index + 1}:[/bold yellow] {response}")

    def construct_knowledge_graph(self):
        for summary in self.summaries:
            for entity in summary.entities:
                self.G.add_node(entity)
                self.console.print(f"[bold cyan]Added node:[/bold cyan] {entity}")

            for edge in summary.relationships:
                self.G.add_edge(edge.source, edge.target, label=edge.relationship)
                self.console.print(f"[bold magenta]Added edge from {edge.source} to {edge.target} with relationship {edge.relationship}[/bold magenta]")

        self.console.print("[bold red]Knowledge graph construction completed.[/bold red]")

    def extract_communities(self):
        index = 0
        for component in nx.connected_components(self.G):
            self.console.print(f"[bold green]Component index {index} of {len(list(nx.connected_components(self.G)))}:[/bold green]")
            subgraph = self.G.subgraph(component)
            if len(subgraph.nodes) > 1:
                try:
                    subcommunities = algorithms.leiden(subgraph)
                    for subcommunity in subcommunities.communities:
                        self.communities.append(list(subcommunity))
                        self.console.print(f"[bold yellow]Community {index}:[/bold yellow] {subcommunity}")
                except Exception as e:
                    self.console.print(f"[bold red]Error in community detection for component {index}:[/bold red] {str(e)}")
            else:
                self.communities.append(list(subgraph.nodes))
            index += 1
        self.console.print(f"[bold blue]Communities: {self.communities}[/bold blue]")

    def summarize_communities(self):
        class CommunitySummary(BaseModel):
            description: str

        community_summarizor_system_prompt = "Your job is to summarize the community of entities and relationships provided to you in a structured format. The content will be given to you in a form of a string, with Entities and Relationships separated by a newline."
        summary_agent = Agent(client=self.client, model_name=self.model_name, system_prompt=community_summarizor_system_prompt)
        for index, community in enumerate(self.communities):
            self.console.print(f"Summarizing community {index} out of {len(self.communities)}")
            subgraph = self.G.subgraph(community)
            nodes = list(subgraph.nodes)
            edges = list(subgraph.edges(data=True))
            description = f"Entities: {nodes}\nRelationships:"
            relationships = []
            for edge in edges:
                relationships.append(f"{edge[0]} -> {edge[2]['label']} -> {edge[1]}")
            description += "\n".join(relationships)
            response = summary_agent.run(role="user", content=description, response_model=CommunitySummary)
            self.community_summaries.append(response)

    def relate_user_query_to_communities(self, user_query):
        class IntermediateResponse(BaseModel):
            intermediate_response: str

        class FinalAnswer(BaseModel):
            final_answer: str

        intermediate_system_prompt = "Relate the user query with the summary of each community provided to you. To better answer the user query."
        final_answer_system_prompt = "Provide the final answer to the user query based on the intermediate responses that are provided to you."

        intermediate_agent = Agent(client=self.client, model_name=self.model_name, system_prompt=intermediate_system_prompt)
        final_answer_agent = Agent(client=self.client, model_name=self.model_name, system_prompt=final_answer_system_prompt)
        intermediate_answers = []
        for index, community_summary in enumerate(self.community_summaries):
            self.console.print(f"Relating user query to community {index} out of {len(self.community_summaries)}")
            prompt = f"User Query: {user_query}\nCommunity Summary: {community_summary.description}"
            response = intermediate_agent.run(role="user", content=prompt, response_model=IntermediateResponse)
            intermediate_answers.append(response.intermediate_response)

        final_response = final_answer_agent.run(role="user", content=f"User Query: {user_query}\nIntermediate Responses: {intermediate_answers}", response_model=FinalAnswer)
        self.console.print(f"Final Answer: {final_response.final_answer}")

    def save_knowledge_graph(self, file_path):
        with open(file_path  , "wb") as file:
           pickle.dump(self.G, file, pickle.HIGHEST_PROTOCOL)

    def load_knowledge_graph(self, file_path):
        with open(file_path , "rb") as file:
            self.G = pickle.load(file)
