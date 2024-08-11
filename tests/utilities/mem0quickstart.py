from dotenv import load_dotenv
from mem0 import Memory


#load key from .env
load_dotenv()

config = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": "localhost",
            "port": 6333,
        }
    },
}

m = Memory.from_config(config)


# For a user
# result = m.add("Likes to play cricket on weekends", user_id="alice", metadata={"category": "hobbies"})
# print(result)

# Get all memories
all_memories = m.get_all()
print(all_memories)

