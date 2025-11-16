import sqlite3

def create_table():
    conn = sqlite3.connect('RobotDialogs.db')
    cursor = conn.cursor()

    # Create table with movie name, character name, dialogue, and synopsis columns
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS dialogues (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        movie_name TEXT,
        character_name TEXT,
        dialogue TEXT,
        synopsis TEXT
    )
    ''')

    # Enable foreign key support (if needed later for movie_id, speaker_id relations, etc.)
    cursor.execute('PRAGMA foreign_keys = ON')

    # Adding optimization for large inserts by turning off the foreign key checks and enabling faster inserts.
    cursor.execute('PRAGMA journal_mode=WAL')
    cursor.execute('PRAGMA synchronous=OFF')

    # Commit changes and close connection
    conn.commit()
    conn.close()

    print("Database and table created successfully!")

# Running the function to create the table
create_table()
