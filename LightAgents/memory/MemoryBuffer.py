from collections import deque
from itertools import islice

class MemoryBuffer:
    def __init__(self , storage_db = None):
        self.messages = deque()
        self.storage_db = storage_db

    def store_message(self, message: str, role: str):
        self.messages.append({"role": role, "content": message})
        if self.storage_db:
            print("Storing message in db")
            self.storage_db.store_message(message, role)

    def get_previous_messages(self):
        return list(self.messages)
    
    def load_messages(self):
        messages_from_db = self.storage_db.load_messages()
        for message in messages_from_db:
            self.messages.append({"role": message[1], "content": message[2]})
    
    def get_latest_messages(self, k: int):
        start_index = max(0, len(self.messages) - k)
        return list(islice(self.messages, start_index, len(self.messages)))