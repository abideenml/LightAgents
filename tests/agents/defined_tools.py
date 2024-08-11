import os
import sys

from pydantic import BaseModel, Field
from typing import  List
import json

from LightAgents.tools.tools import Tool


class CheckAvailabilityInputSchema(BaseModel):
    item_name: str = Field(..., description="The name of the item to check for availability")

class CheckAvailabilityOutputSchema(BaseModel):
    result: List[str] = Field(..., description="The availability of the item")

class AddItemInputSchema(BaseModel):
    item_name: str = Field(..., description="The name of the item to add")
    quantity: int = Field(..., description="The quantity of the item to add")

class AddItemOutputSchema(BaseModel):
    success: bool = Field(..., description="Whether the item was successfully added")
    message: str = Field(..., description="Updated Quantity in the inventory")

def check_availability(item_name: str) -> dict:
    with open("./tests/agents/hotel_data.json", "r") as file:
        data = json.load(file)
    available = item_name in data
    return {"result": ["Available" if available else "Not available"]}

def add_item(item_name: str, quantity: int) -> dict:
    try:
        with open("./tests/agents/hotel_data.json", "r+") as file:
            data = json.load(file)
            if item_name in data:
                data[item_name] += quantity
                return {"success": True, "message": f"{data[item_name]}"}
            data[item_name] = quantity
            file.seek(0)
            json.dump(data, file)
        return {"success": True, "message": "Item added successfully."}
    except Exception as e:
        return {"success": False, "message": str(e)}



check_availability_tool = Tool(
    function=check_availability,
    input_schema=CheckAvailabilityInputSchema,
    output_schema=CheckAvailabilityOutputSchema,
    description="Check the availability of an item in the hotel",
    type="sync"
)

add_item_tool = Tool(
    function=add_item,
    input_schema=AddItemInputSchema,
    output_schema=AddItemOutputSchema,
    description="Add an item to the hotel inventory",
    type="sync"
)
