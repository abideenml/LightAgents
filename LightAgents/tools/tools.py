import inspect
from typing import Any, Callable, Dict
from LightAgents.tools.helper_classes import ToolInputSchema, ToolOutputSchema


class Tool:
    """
    A class that wraps a function and provides input and output schemas for the function.
    """

    def __init__(
        self,
        function: Callable, # The function to be wrapped
        input_schema: ToolInputSchema, # The input schema for the function
        output_schema: ToolOutputSchema, # The output schema for the function
        description: str = "", # A description of the function
        type: str = "sync", # The type of the function (sync or async)
    ):
        self.function = function
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.description = description
        self.type = type

    # Run the function synchronously
    def run_sync(self, *args, **kwargs) -> Any:
        validated_input = self.input_schema(*args, **kwargs)
        result = self.function(**validated_input.dict())

        if isinstance(result, dict):
            return self.output_schema(**result)
        else:
            return self.output_schema(result=result)

    # Run the function asynchronously
    async def run_async(self, *args, **kwargs) -> Any:
        validated_input = self.input_schema(*args, **kwargs)
        result = await self.function(**validated_input.dict())

        if isinstance(result, dict):
            return self.output_schema(**result)
        else:
            return self.output_schema(result=result)

    # Run the function
    def run(self, *args, **kwargs) -> Any:
        if inspect.iscoroutinefunction(self.function):
            return self.run_async(*args, **kwargs)
        else:
            return self.run_sync(*args, **kwargs)

    # Validate the input
    def validate_input(self, *args, **kwargs) -> Dict:
        validated_input = self.input_schema(*args, **kwargs)
        return validated_input

    # Dump the output schema
    def dump_output_schema(self) -> Dict:
        return self.output_schema.model_json_schema()

    # Dump the input schema
    def dump_input_schema(self) -> Dict:
        return self.input_schema.model_json_schema()

    def generate_schemas(self) -> Dict[str, Dict]:
        """Generates and returns the function schema including input and output JSON schemas."""
        input_schema = self.dump_input_schema()
        input_properties = input_schema.get("properties", {})
        required_fields = (
            list(self.input_schema.model_fields.keys()) if input_properties else []
        )

        function_schema = {
            "type": self.type,
            "function": {
                "name": self.function.__name__,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": input_properties,
                    "required": required_fields,
                },
                "output": self.dump_output_schema(),
            },
        }
        return function_schema
