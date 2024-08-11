import sqlite3

class MemoryStorage:
    def __init__(self, db_name='memory_buffer.db'):
        self.db_connection = sqlite3.connect(db_name)
        self.db_cursor = self.db_connection.cursor()
        self.initialize_database()

    def initialize_database(self):
        self.db_cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY,
                role TEXT NOT NULL,
                content TEXT NOT NULL
            )
        ''')
        self.db_connection.commit()

    def store_message(self, message: str, role: str):
        self.db_cursor.execute('''
            INSERT INTO messages (role, content) VALUES (?, ?)
        ''', (role, message))
        self.db_connection.commit()

    def load_messages(self):
        self.db_cursor.execute('''
            SELECT * FROM messages
        ''')
        return self.db_cursor.fetchall()

    def close_connection(self):
        self.db_connection.close()