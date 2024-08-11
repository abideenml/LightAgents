import asyncio

from dotenv import load_dotenv
from LightAgents.tokenizer.Models import OpenAIModels, OpenSourceModels
from LightAgents.tokenizer.Tokenizer import create_tokenizer

load_dotenv(".env" , override=True)

async def main():
    # Example text to tokenize
    text = "Hello, world! This is a test."

    # Test with an OpenAI model
    openai_model_name = OpenAIModels.GPT_4O.value
    try:
        openai_tokenizer = await create_tokenizer(openai_model_name)
        openai_result = openai_tokenizer.tokenize(text)
        print(f"OpenAI Model: {openai_model_name}")
        print(f"Tokens: {openai_result.tokens}")
        print(f"Token Count: {openai_result.count}")
    except ValueError as e:
        print(e)

    # Test with an Open Source model
    open_source_model_name = OpenSourceModels.GOOGLE_GEMMA_7B.value
 
    open_source_tokenizer = await create_tokenizer(open_source_model_name)
    open_source_result = open_source_tokenizer.tokenize(text)
    print(f"Open Source Model: {open_source_model_name}")
    print(f"Tokens: {open_source_result.tokens}")
    print(f"Token Count: {open_source_result.count}")

# Run the test
asyncio.run(main())