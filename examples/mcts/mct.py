import math
import random
import instructor
import numpy as np
import os
import dotenv
import openai
from rich.console import Console
from rich.tree import Tree
from rich.panel import Panel


from LightAgents.agents.agent import Agent
from examples.mcts.utilities import get_answer_from_llm, get_critique_from_llm, get_ranking_from_llm


max_children  = 2
console = Console()

dotenv.load_dotenv('.env')
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

# Initialize the OpenAI client (together API makes use of the OpenAI SDK under the hood)
client = openai.OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=TOGETHER_API_KEY,
)
# Convert the OpenAI client to an instructor client
client = instructor.from_openai(client)
model = "mistralai/Mixtral-8x22B-Instruct-v0.1"

agent = Agent(client=client, model_name=model , system_prompt=None)


class MCTNode:
    def __init__(self, question,answer, parent=None):
        self.question = question
        self.answer = answer
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
    
    def is_fully_expanded(self):
        return len(self.children) == max_children
    

    def calculate_uct(self, exploration_weight=1.44):
        if self.visits == 0:
            return float("inf")
        return self.value / self.visits + exploration_weight * math.sqrt(2.0 * math.log(self.parent.visits) / self.visits)
    
    def best_child(self, exploration_weight=1.44):
        choices = []
        for child in self.children:
            uct = child.calculate_uct(exploration_weight)
            choices.append(uct)
        return self.children[np.argmax(choices)]
    
    def most_visited_child(self):
        return max(self.children, key=lambda x: x.visits)
    
    def add_child(self, child_node):
        self.children.append(child_node)


class MCT:
    def __init__(self, question, seed_answers, iterations=10):
        self.question = question
        self.seed_answers = seed_answers
        self.iterations = iterations
        self.root = MCTNode(question, random.choice(self.seed_answers), None)

    def search(self):
        tree = Tree(f"[bold blue]Root: {self.root.answer[:30]}...")
        
        for i in range(self.iterations):
            console.print(f"\n[bold green]Iteration: {i+1}/{self.iterations}")
            node = self.select_node(self.root)
            console.print(Panel(f"Selected node: {node.answer[:50]}...", title="Selection", border_style="yellow"))
            
            if not node.is_fully_expanded():
                node = self.expand_node(node)
                console.print(Panel(f"Expanded node: {node.answer[:50]}...", title="Expansion", border_style="cyan"))
            else:
                while len(node.children) > 0:
                    node = node.best_child() 
            result = self.simulate(node)
            console.print(Panel(f"Simulated reward: {result}", title="Simulation", border_style="magenta"))
            
            self.backpropagate(node, result[0])
            
        
        console.print(tree)
        console.print(f"[bold red]Visits to the most visited child: {self.root.most_visited_child().visits}")
        return self.root.most_visited_child().answer


    def select_node(self, node):
        while node.is_fully_expanded() and len(node.children) > 0:
            node = node.best_child()
        return node
    
    def expand_node(self, node):
        for i in range(max_children - len(node.children)):
           child_node = MCTNode(self.question, node.answer, node)
           node.add_child(child_node)
           critique = get_critique_from_llm(agent, self.question, child_node.answer)
           console.print(Panel(f"Critique: {critique}", title="Critique", border_style="red"))
           improved_answer = get_answer_from_llm(agent, self.question, child_node.answer, critique)
           console.print(Panel(f"Improved Answer: {improved_answer[:50]}...", title="Improvement", border_style="green"))
           child_node.answer = improved_answer
        return random.choice(node.children)
    
    def simulate(self, node):
        rating = get_ranking_from_llm(agent, self.question, node.answer)
        return rating
    
    def backpropagate(self, node, result):
        while node is not None:
            node.visits += 1
            node.value += result
            console.print(f"Backpropagating: {node.answer[:30]}... , visits: {node.visits}")
            node = node.parent


