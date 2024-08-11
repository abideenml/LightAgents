# Tools setup
from typing import List
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from LightAgents.agents.agent import Agent
from LightAgents.systemprompt.systemprompt import ChainOfThought


chain_of_thought = ChainOfThought( goal="Provide answer by solving the steps provided to you.", background="Your are a math tutor for pre-schoolers, you need to explain the problem in easiest way possible and derive the solution in steps.",steps=["Breakdown the complex probelm into small problems", "Solve the small problems step by step", "Combine the solutions to get the final answer."])

# Client setup
client = instructor.from_openai(OpenAI(
 base_url='http://localhost:11434/v1',
    api_key='llama3',  # required, but unused
),
mode=instructor.Mode.JSON)



# Example usage
agent = Agent(client=client, model_name="llama3", system_prompt_generator=chain_of_thought )

class MathProblemAnswer(BaseModel):
    answer: List[str] = Field(description="The answer to the math problem.")

resp = agent.run(
    response_model=MathProblemAnswer,
    user_message="Saddie has 5 cricket bats. She buys 10 more cricket kits, each containing 3 cricket balls adn 5 cricket bats. How many cricket balls and bats does Saddie have now?",
    examples={"Question": "Roger has 5 tennis balls. He buys 2 more cans of tennis balls, each containing 3 tennis balls. How many tennis balls does Roger have now?", "Answer": ["Step 1: Roger has 5 tennis balls." ,"Step 2: Roger buys 2 cans of tennis balls, each containing 3 tennis balls."," Step 3: Roger has 5 + 2 * 3 = 11 tennis balls."]},
)
print(resp)