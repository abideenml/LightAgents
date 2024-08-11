import inspect
from typing import Any, Callable, Dict
from pydantic import BaseModel
from LightAgents.tools.helper_classes import ToolInputSchema, ToolOutputSchema

class Tool:
    def __init__(self, function: Callable, input_schema: ToolInputSchema, output_schema: ToolOutputSchema, description: str = "", type: str = "sync"):
        self.function = function
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.description = description
        self.type = type

    def run_sync(self, *args, **kwargs) -> Any:
        validated_input = self.input_schema(*args, **kwargs)
        result = self.function(**validated_input.dict())
        
        if isinstance(result, dict):
            return self.output_schema(**result)
        else:
            return self.output_schema(result=result)

    async def run_async(self, *args, **kwargs) -> Any:
        validated_input = self.input_schema(*args, **kwargs)
        result = await self.function(**validated_input.dict())
        
        if isinstance(result, dict):
            return self.output_schema(**result)
        else:
            return self.output_schema(result=result)

    def run(self, *args, **kwargs) -> Any:
        if inspect.iscoroutinefunction(self.function):
            return self.run_async(*args, **kwargs)
        else:
            return self.run_sync(*args, **kwargs)

    def validate_input(self, *args, **kwargs) -> Dict:
        validated_input = self.input_schema(*args, **kwargs)
        return validated_input

    def dump_output_schema(self) -> Dict:
        return self.output_schema.model_json_schema()

    def dump_input_schema(self) -> Dict:
        return self.input_schema.model_json_schema()

    def generate_schemas(self) -> Dict[str, Dict]:
        """Generates and returns the function schema including input and output JSON schemas."""
        input_schema = self.dump_input_schema()
        input_properties = input_schema.get("properties", {})
        required_fields = list(self.input_schema.model_fields.keys()) if input_properties else []

        function_schema = {
            "type": self.type,
            "function": {
                "name": self.function.__name__,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": input_properties,
                    "required": required_fields
                },
            }
        }
        return function_schema

# Example usage
class ExampleInputSchema(ToolInputSchema):
    param1: int
    param2: str

class ExampleOutputSchema(ToolOutputSchema):
    result: str

async def example_async_function(param1: int, param2: str) -> Dict[str, str]:
    return {"result": f"Processed {param1} and {param2}"}

def example_sync_function(param1: int, param2: str) -> Dict[str, str]:
    return {"result": f"Processed {param1} and {param2}"}

# Function that does not return a dictionary
def example_non_dict_function(param1: int, param2: str) -> str:
    return f"Processed {param1} and {param2}"

# Function that does not need any inputs
def example_no_input_function() -> Dict[str, str]:
    return {"result": "No input needed"}

async def main():
    async_tool = Tool(example_async_function, ExampleInputSchema, ExampleOutputSchema, "Example async function", "async")
    sync_tool = Tool(example_sync_function, ExampleInputSchema, ExampleOutputSchema, "Example sync function", "sync")
    non_dict_tool = Tool(example_non_dict_function, ExampleInputSchema, ExampleOutputSchema, "Example non-dict function", "sync")
    no_input_tool = Tool(example_no_input_function, ToolInputSchema, ExampleOutputSchema, "Example no input function", "sync")

    async_result = await async_tool.run(param1=1, param2="test")
    print(async_result)

    sync_result = sync_tool.run(param1=1, param2="test")
    print(sync_result)

    non_dict_result = non_dict_tool.run(param1=1, param2="test")
    print(non_dict_result)

    no_input_result = no_input_tool.run()
    print(no_input_result)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())