from abc import ABC, abstractmethod
from typing import List, Dict

# Base class for context providers
class ContextProvider(ABC):
    @abstractmethod
    def provide_context(self) -> str:
        pass

# Base class for prompting techniques
class SystemPromptGenerator(ABC):
    
    def __init__(self , goal: str , background: str):
        self.context_providers: Dict[str, ContextProvider] = {}  # For dynamic context addition
        self.goal = goal
        self.background = background

    def add_context_provider(self, key: str, provider: ContextProvider):
        self.context_providers[key] = provider

    def generate_context(self) -> List[str]:
        return [provider.provide_context() for provider in self.context_providers.values()]

    @abstractmethod
    def generate_additional_sections(self) -> List[str]:
        pass

    def generate(self) -> str:
        # Enhanced version for clarity and readability
        prompt_parts = [
            f"**What you're trying to achieve:** {self.goal}",
            f"**The necessary background information:** {self.background}",
        ]
        # Adding a clear separator for additional sections if they exist
        if self.generate_additional_sections():
            prompt_parts.append("**Additional Details:**")
            prompt_parts.extend(self.generate_additional_sections())

        # Adding context with a clear introduction
        if self.generate_context():
            prompt_parts.append("**Contextual Information:**")
            prompt_parts.extend(self.generate_context())

        return "\n\n".join(prompt_parts).strip()


## Generic System Prompt Generator
class SimpleSystemPromptGenerator(SystemPromptGenerator):
    def generate_additional_sections(self) -> List[str]:
        return []



# Chain of Thought
class ChainOfThought(SystemPromptGenerator):
    
    def __init__(self,goal: str , background: str,steps: List[str]):
        super().__init__(goal, background)
        self.steps = steps
    
    def generate_additional_sections(self) -> List[str]:
        steps_intro = ["Please follow these steps carefully:"]
        steps_formatted = [f"Step {i+1}: {step}" for i, step in enumerate(self.steps)]
        return steps_intro + steps_formatted



# Few Shot
class FewShot(SystemPromptGenerator):
    
    def __init__(self,goal: str,background:str ,examples: List[str]):
        super().__init__(goal , background)
        self.examples = examples
    
    def generate_additional_sections(self) -> List[str]:
        examples_intro = ["Please utilize the examples given below to understand the context and structure of your response:"]
        examples_formatted = [f"Example {i+1}: {example}" for i, example in enumerate(self.examples)]
        return examples_intro + examples_formatted
    
