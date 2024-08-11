
from typing import Optional
from pydantic import BaseModel


class ToolExecution(BaseModel):
    tool_name: str
    parameters: dict


class ToolResponse(BaseModel):
    success: bool
    message: str
    tool_output: Optional[dict] = None