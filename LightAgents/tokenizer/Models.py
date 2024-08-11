from enum import Enum

class OpenAIModels(Enum):
    GPT_4O = "gpt-4o"
    GPT_3_5_TURBO = "gpt-3.5-turbo"
    GPT_4 = "gpt-4"
    GPT_4_32K = "gpt-4-32k"
    GPT_4_1106_PREVIEW = "gpt-4-1106-preview"
class OpenSourceModels(Enum):
    CODELLAMA_CODELLAMA_7B_HF = "codellama/CodeLlama-7b-hf"
    CODELLAMA_CODELLAMA_70B_HF = "codellama/CodeLlama-70b-hf"
    META_LLAMA_META_LLAMA_3_8B = "meta-llama/Meta-Llama-3-8B"
    META_LLAMA_META_LLAMA_3_70B = "meta-llama/Meta-Llama-3-70B"
    MICROSOFT_PHI_2 = "microsoft/phi-2"
    GOOGLE_GEMMA_7B = "google/gemma-7b"
    TIIUAE_FALCON_7B = "tiiuae/falcon-7b"
    AI_YI_6B = "01-ai/Yi-6B"