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


    def generate(self) -> str:
        # Enhanced version for clarity and readability
        prompt_parts = [
            f"**What you're trying to achieve:** {self.goal}",
            f"**The necessary background information:** {self.background}",
        ]

        # Adding context with a clear introduction
        if self.generate_context():
            prompt_parts.append("**Contextual Information:**")
            prompt_parts.extend(self.generate_context())

        return "\n\n".join(prompt_parts).strip()


## Generic System Prompt Generator
class SimpleSystemPromptGenerator(SystemPromptGenerator):
    def __init__(self, goal: str, background: str):
        super().__init__(goal, background)
    def generate(self) -> str:
        return super().generate()



# Chain of Thought
class ChainOfThought(SystemPromptGenerator):
    
    def __init__(self,goal: str , background: str,steps: List[str]):
        super().__init__(goal, background)
        self.steps = steps
    
    def generate(self , intro: str) -> str:
        return super().generate() + "\n\n" + intro + "\n" + "\n".join([f"Step {i+1}: {step}" for i, step in enumerate(self.steps)])


# Few Shot
class FewShot(SystemPromptGenerator):
    
    def __init__(self,goal: str,background:str ,examples: List[str]):
        super().__init__(goal , background)
        self.examples = examples
    
    
    def generate(self , intro: str) -> str:
        return super().generate() + "\n\n" + intro + "\n" + "\n".join([f"Example {i+1}: {example}" for i, example in enumerate(self.examples)])
    
