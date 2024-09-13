# Tools are wrappers around functions that provide input and output schemas.
# These schemas are used to validate the input and output of the function.

# The ToolInputSchema and ToolOutputSchema classes are used to define the input and output schemas for a tool. They are subclasses of Pydantic's BaseModel class, which provides a way to define data models with type hints.
from pydantic import BaseModel

# Used to define the input schema for a tool
# if no input is required, the ToolInputSchema class can be left empty
class ToolInputSchema(BaseModel):
    pass

# used to define the output schema for a tool
# if return type is of type dict , specify the fields of the dict 
# else define a field as 'result' and specify the type of the result
class ToolOutputSchema(BaseModel):
    pass