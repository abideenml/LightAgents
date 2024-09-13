
def get_critique_prompt(question:str, answer: str) -> str:
    prompt = (
        f"Question: {question}\n"
        f"Draft Answer: {answer}\n"
        "Please critique the draft answer."
        "Do a careful assessment whether the answer is correct or not, and why."
        "Consider multiple ways of verifying the correctness of the answer."
        "Do point out every flaw and hold the answer to high standards."
        "Do provide specific recommendations to improve the answer."
        "Do think step by step."
        "Do not provide a revised answer."
    )
    return prompt


def improve_answer_prompt(question:str, answer:str , critique:str) -> str:
    prompt = (
        f"Question: {question}\n"
        f"Draft Answer: {answer}\n"
        f"Critique: {critique}\n"
        "Please improve the draft answer based on the critique.Follow this format:\n"
        "Reasoning Process: <Step by step reasoning process>\n"
        "Verification: <Verification of the facts>\n"
        "Final Answer: <Final improved answer>"   
    )
    return prompt

def get_ranking_prompt(question:str, answer: str) -> str:
    prompt = (
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "As an expert on this topic , provide a detailed critique of the answer with respect to the question."
        "Provide only a critique of the answer, not a revised answer.",
        "Rate the answer from 1 to 100 based on the correctness, relevance, and quality of the answer."
        "The response should be in the following format:\n"
        "Critique: <Detailed critique of the answer>\n"
        "Rating: <Rating from 1 to 100>"
    )
    return prompt