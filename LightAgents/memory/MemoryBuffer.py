from collections import deque
from itertools import islice

class MemoryBuffer:
    '''
    MemoryBuffer class is used to store the messages exchanged between the agents.
    '''
    def __init__(self , storage_db = None):
        self.messages = deque() # deque is used to store the messages in a list
        self.storage_db = storage_db # storage_db is used to store the messages in a database

    # store_message method is used to store the messages in the memory buffer
    def store_message(self, message: str | list, role: str):
        # If the message is a list, then it is stored as a list of messages
        self.messages.append({"role": role, "content": message})
        # If the storage_db is not None, then the message is stored in the database
        if self.storage_db:
            # print the message that is stored in the database
            print("Storing message in db")
            self.storage_db.store_message(message, role)

    # reset_memory method is used to reset the memory buffer
    def reset_memory(self):
        # Clear the messages in the memory buffer
        self.messages.clear()
        if self.storage_db:
            self.storage_db.clear_memory()

    # get_previous_messages method is used to get the previous messages from the memory buffer
    def get_previous_messages(self):
        return list(self.messages)

    # load_messages method is used to load the messages from the database   
    def load_messages(self):
        messages_from_db = self.storage_db.load_messages()
        for message in messages_from_db:
            self.messages.append({"role": message[1], "content": message[2]})
    
    # get_latest_messages method is used to get the latest messages from the memory buffer
    def get_latest_messages(self, k: int):
        if k == 0:
            return []
        start_index = max(0, len(self.messages) - k)
        return list(islice(self.messages, start_index, len(self.messages)))