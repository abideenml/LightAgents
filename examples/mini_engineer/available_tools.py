
import os

# import sys
# # Add the root directory of your project to the Python path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from pydantic import Field
from LightAgents.tools.helper_classes import ToolInputSchema, ToolOutputSchema
from LightAgents.tools.tools import Tool
from tavily import TavilyClient

tavily_key = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=tavily_key)

class CreateFolderInputSchema(ToolInputSchema):
    path: str = Field(..., title="Path", description="The path of the folder to be created.")

# Function to create a folder
def create_folder(path):
    try:
        os.makedirs(path, exist_ok=True)
        return f"Folder created: {path}"
    except Exception as e:
        return f"Error creating folder: {str(e)}"
   
class CreateFolderOutputSchema(ToolOutputSchema):
    result : str = Field(..., title="Result", description="The result of the operation.If successful , the message will be 'Folder created: {path}'. If unsuccessful, the message will be 'Error creating folder: {error message}'.")

create_folder_tool = Tool(
    function=create_folder,
    input_schema=CreateFolderInputSchema,
    output_schema=CreateFolderOutputSchema,
    description="Create a folder at the specified path."
)

class CreateFileInputSchema(ToolInputSchema):
    path: str = Field(..., title="Path", description="The path of the file to be created.")
    content: str = Field("", title="Content", description="The content to be written to the file.")

# Function to create a file
def create_file(path, content=""):
    try:
        with open(path, 'w') as f:
            f.write(content)
        return f"File created: {path}"
    except Exception as e:
        return f"Error creating file: {str(e)}"
    
class CreateFileOutputSchema(ToolOutputSchema):
    result : str = Field(..., title="Result", description="The result of the operation. If successful, the message will be 'File created: {path}'. If unsuccessful, the message will be 'Error creating file: {error message}'.")

create_file_tool = Tool(
    function=create_file,
    input_schema=CreateFileInputSchema,
    output_schema=CreateFileOutputSchema,
    description="Create a file at the specified path with optional content."
)

class WriteToFileInputSchema(ToolInputSchema):
    path: str = Field(..., title="Path", description="The path of the file to write to.")
    content: str = Field(..., title="Content", description="The content to write to the file.")

# Function to write to a file
def write_to_file(path, content):
    try:
        with open(path, 'w') as f:
            f.write(content)
        return f"Content written to file: {path}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"

class WriteToFileOutputSchema(ToolOutputSchema):
    result : str = Field(..., title="Result", description="The result of the operation. If successful, the message will be 'Content written to file: {path}'. If unsuccessful, the message will be 'Error writing to file: {error message}'.")


write_to_file_tool = Tool(
    function=write_to_file,
    input_schema=WriteToFileInputSchema,
    output_schema=WriteToFileOutputSchema,
    description="Write content to a file at the specified path."
)

class ReadFileInputSchema(ToolInputSchema):
    path: str = Field(..., title="Path", description="The path of the file to read.")

# Function to read a file
def read_file(path):
    try:
        with open(path, 'r') as f:
            content = f.read()
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"
    
class ReadFileOutputSchema(ToolOutputSchema):
    result : str = Field(..., title="Content", description="The content of the file. If the file is read successfully, the content will be returned. If an error occurs, the message will be 'Error reading file: {error message}'.")

read_file_tool = Tool(
    function=read_file,
    input_schema=ReadFileInputSchema,
    output_schema=ReadFileOutputSchema,
    description="Read the contents of a file at the specified path."
)

class ListFilesInputSchema(ToolInputSchema):
    path: str = Field(".", title="Path", description="The path of the directory to list files from.")

# Function to list files in the root directory
def list_files(path="."):
    try:
        files = os.listdir(path)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {str(e)}"
    
class ListFilesOutputSchema(ToolOutputSchema):
    result : str = Field(..., title="Files", description="The list of files in the directory. If the files are listed successfully,a string containing  the list of files will be returned. If an error occurs, the message will be 'Error listing files: {error message}'.")


list_files_tool = Tool(
    function=list_files,
    input_schema=ListFilesInputSchema,
    output_schema=ListFilesOutputSchema,
    description="List the files in the root directory of the project."
)

tools_list = [create_folder_tool, create_file_tool, write_to_file_tool, read_file_tool, list_files_tool]

class TavilySearchInputSchema(ToolInputSchema):
    query: str = Field(..., title="Query", description="The search query to be sent to Tavily.")

class TavilySearchOutputSchema(ToolOutputSchema):
    result: str = Field(..., title="Answer", description="The answer generated in response to the query.")
# Function to perform a Tavily search
def tavily_search(query):
    try:
        response = tavily.qna_search(query=query, search_depth="advanced")
        return response
    except Exception as e:
        return f"Error performing search: {str(e)}"
    

tavily_search_tool = Tool(
    function=tavily_search,
    input_schema=TavilySearchInputSchema,
    output_schema=TavilySearchOutputSchema,
    description="Performs a Tavily search and returns an answer to the provided query. Optimal for use as an AI agent tool for question answering."
)
    
tools_list.append(tavily_search_tool)