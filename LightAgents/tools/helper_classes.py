# Tools are wrappers around functions that provide input and output schemas.
# These schemas are used to validate the input and output of the function.

# The ToolInputSchema and ToolOutputSchema classes are used to define the input and output schemas for a tool. They are subclasses of Pydantic's BaseModel class, which provides a way to define data models with type hints.
from pydantic import BaseModel


class ToolInputSchema(BaseModel):
    pass

class ToolOutputSchema(BaseModel):
    pass