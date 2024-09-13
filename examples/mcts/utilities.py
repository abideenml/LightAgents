import re

from LightAgents.agents.agent import Agent
from examples.mcts.system_prompts import get_ranking_prompt, improve_answer_prompt

seed_answers = [
    "I don't know the answer",
    "I'm not sure",
    "I can't help with that",
]

def extract_ranking(response: str):
    match = re.search(r"Rating:\s*(\d+)", response)
    try:
        if match:
            match = match.group(1)
            return float(match)/100
        else:
            raise ValueError("Rating not found in response")
    except Exception as e:
        print(f"Error extracting response: {e}")
        print(f"The llm Response which was used to extract the ratings from was: {response}")
        return 0.0
        
def get_answer_from_llm(agent: Agent , question:str,answer:str,critique:str):
     system_prompt = improve_answer_prompt(question, answer, critique)
     prompt = (
            f"Question: {question}\n Provide the improved answer to the question."
            
        )
     agent.system_prompt = system_prompt
     response = agent.run(role="user", content=prompt, response_model=None , last_k_messages=0)
     retrieved_response = response.choices[0].message.content
     try:
         match = re.search(r"Final Answer:\s*(.*)", retrieved_response)
         if match is not None:
            final_answer = match.group(1).strip()
         else:
            raise ValueError("Final Answer not found in response")
     except Exception as e:
            print(f"Error extracting response: {e}")
            print(f"The llm Response which was used to extract the final answer from was: {response}")
            final_answer = None
     return final_answer,retrieved_response

def get_ranking_from_llm(agent: Agent , question:str, answer:str):
        system_prompt =  get_ranking_prompt(question, answer)
        prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            "provide rating from 1 to 100 based on the correctness, relevance, and quality of the answer."
        )
        agent.system_prompt = system_prompt
        response = agent.run(role="user", content=prompt, response_model=None , last_k_messages=0)
        retrieved_response = response.choices[0].message.content
        return extract_ranking(retrieved_response),retrieved_response

def get_critique_from_llm(agent: Agent , question:str, answer:str):
        system_prompt =  get_ranking_prompt(question, answer)
        user_prompt = (
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            "Please provide a detailed critique of the answer with respect to the question."
            "Provide only a critique of the answer, not a revised answer."
        )
        agent.system_prompt = system_prompt
        response = agent.run(role="user", content=user_prompt, response_model=None, last_k_messages=0)
        retrieved_response = response.choices[0].message.content
        return retrieved_response
    



