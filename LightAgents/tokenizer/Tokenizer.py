import os
from transformers import PreTrainedTokenizerFast as PreTrainedTokenizer
from typing import List
from openai import BaseModel
import tiktoken
from LightAgents.tokenizer.Models import OpenAIModels, OpenSourceModels




class TokenizerResult(BaseModel):
    name: str
    tokens: List[int]
    count: int

class Tokenizer:
    def __init__(self, name: str):
        self.name = name

    def tokenize(self, text: str) -> TokenizerResult:
        raise NotImplementedError

class TiktokenTokenizer(Tokenizer):
    def __init__(self, model: str):
        super().__init__(model)
        self.enc = tiktoken.encoding_for_model(model)

    def tokenize(self, text: str) -> TokenizerResult:
        tokens = self.enc.encode(text)
        return TokenizerResult(name=self.name, tokens=tokens, count=len(tokens))

class OpenSourceTokenizer(Tokenizer):
    def __init__(self, tokenizer: PreTrainedTokenizer, name: str):
        super().__init__(name)
        self.tokenizer = tokenizer

    @staticmethod
    def load(model: str) -> PreTrainedTokenizer:
        hf_token = os.getenv("HUGGING_FACE_API_TOKEN")
        if hf_token:
            return PreTrainedTokenizer.from_pretrained(model,token=hf_token , cache_dir="./cache") 
        else:
            raise ValueError("Hugging Face API token not found")

    def tokenize(self, text: str) -> TokenizerResult:
        tokens = self.tokenizer.encode(text)
        return TokenizerResult(name=self.name, tokens=tokens, count=len(tokens))

async def create_tokenizer(name: str) -> Tokenizer:
    if name in OpenAIModels._value2member_map_:
        return TiktokenTokenizer(name)
    elif name in OpenSourceModels._value2member_map_:
        tokenizer = OpenSourceTokenizer.load(name)
        return OpenSourceTokenizer(tokenizer, name)
    else:
        raise ValueError("Invalid model name")