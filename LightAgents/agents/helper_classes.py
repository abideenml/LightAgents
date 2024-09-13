
from typing import Optional
from pydantic import BaseModel


# Pydantic model for the input schema of a tool call
class ToolExecution(BaseModel):
    tool_name: str
    parameters: dict

# Pydantic model for the response of a tool call
class ToolResponse(BaseModel):
    success: bool
    message: str
    tool_output: Optional[dict] = None