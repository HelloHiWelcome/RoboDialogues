import json
import sqlite3

# Define the robot characters and their speaker IDs and movie IDs
robot_characters = {
    "HAL 9000": {"speaker_ids": ["u56"], "movie_ids": ["m3"]},
    "Bishop": {"speaker_ids": ["u229"], "movie_ids": ["m15"]},
    "Simone/Viktor": {"speaker_ids": ["u2783"], "movie_ids": ["m181"]},
    
    # Merge Data across all movies
    "Data": {
        "speaker_ids": ["u2899", "u2993", "u3025"], 
        "movie_ids": ["m191", "m196", "m198"]
    },
    
    "Borg Queen": {"speaker_ids": ["u2992"], "movie_ids": ["m196"]},
    
    # Consolidate C-3PO for all appearances
    "C-3PO": {
        "speaker_ids": ["u5103", "u7826", "u7251"], 
        "movie_ids": ["m337", "m529", "m489"]
    },
    
    # Consolidate Terminator for both movies
    "Terminator": {
        "speaker_ids": ["u8077", "u8098"], 
        "movie_ids": ["m547", "m549"]
    },
    
    "Leeloo": {"speaker_ids": ["u85"], "movie_ids": ["m5"]},
    
    # New AI characters from The Matrix
    "Agent Smith": {"speaker_ids": ["u6517"], "movie_ids": ["m433"]},
    "Agent Jones": {"speaker_ids": ["u6516"], "movie_ids": ["m433"]},
    "Oracle": {"speaker_ids": ["u6524"], "movie_ids": ["m433"]}
}

# Movie ID dictionary to map movie IDs to movie names
movie_id_dict = {
    "m3": "2001: A Space Odyssey",
    "m15": "Aliens",
    "m181": "Simone",
    "m191": "Star Trek: Generations",
    "m196": "Star Trek: First Contact",
    "m198": "Star Trek: Nemesis",
    "m337": "Star Wars: The Empire Strikes Back",
    "m529": "Star Wars",
    "m489": "Star Wars: Episode VI - Return of the Jedi",
    "m547": "Terminator 2: Judgment Day",
    "m549": "The Terminator",
    "m5": "The Fifth Element",
    "m433": "The Matrix"
}

# Load the speakers data (speaker_id to character_name mapping)
with open(r'C:\Users\Hp\.convokit\saved-corpora\movie-corpus\speakers.json', 'r') as file:
    speakers_data = json.load(file)

# Load the conversations data (metadata for each line, including movie_idx and movie_name)
with open(r'C:\Users\Hp\.convokit\saved-corpora\movie-corpus\conversations.json', 'r') as file:
    conversations_data = json.load(file)

# Load the utterances data (dialogues with speaker_id and movie_id)
with open(r'C:\Users\Hp\.convokit\saved-corpora\movie-corpus\utterances.jsonl', 'r') as file:
    utterances_data = [json.loads(line) for line in file]

# Connect to SQLite database (or create one if it doesn't exist)
conn = sqlite3.connect('RobotDialogs.db')
cursor = conn.cursor()

# Clear the previous dialogues (if any) before inserting new ones
cursor.execute("DELETE FROM dialogues;")
conn.commit()

# Reset the auto-increment ID counter
cursor.execute("DELETE FROM sqlite_sequence WHERE name='dialogues';")
conn.commit()

# Create a table for storing the dialogues (movie_name, character_name, dialogue, synopsis)
cursor.execute(''' 
CREATE TABLE IF NOT EXISTS dialogues ( 
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_name TEXT, 
    character_name TEXT, 
    dialogue TEXT, 
    synopsis TEXT 
)
''')

# Insert dialogues into the database
def insert_dialogue(movie_name, character_name, dialogue):
    cursor.execute("""
        INSERT INTO dialogues (movie_name, character_name, dialogue)
        VALUES (?, ?, ?)
    """, (movie_name, character_name, dialogue))
    conn.commit()

# Process the utterances data and filter for the AI/robot characters
inserted_count = 0
data_to_insert = {}

# Read the utterances.jsonl file
for utterance in utterances_data:
    speaker_id = utterance['speaker']
    dialogue = utterance['text']
    movie_id = utterance['meta']['movie_id']
    
    # Only process the relevant movies (those in relevant_movie_ids)
    if movie_id not in movie_id_dict:
        continue  # Skip this utterance if the movie isn't in the relevant list
    
    # Ensure we get the correct movie name by looking up the movie_id in the movie_id_dict
    movie_name = movie_id_dict.get(movie_id)
    
    if not movie_name:
        print(f"Warning: Movie ID {movie_id} is not in the movie_id_dict. Skipping this entry.")
        continue  # Skip this dialogue if no movie name is found for the movie_id
    
    # Check if this utterance corresponds to an AI/robot character
    for character_name, data in robot_characters.items():
        if speaker_id in data['speaker_ids'] and movie_id in data['movie_ids']:
            # Collect data for bulk insert if movie name is found
            if character_name not in data_to_insert:
                data_to_insert[character_name] = []
            data_to_insert[character_name].append((movie_name, dialogue))

# Insert dialogues for all relevant character names
for character_name, dialogues in data_to_insert.items():
    for movie_name, dialogue in dialogues:
        insert_dialogue(movie_name, character_name, dialogue)
        inserted_count += 1

# Output how many dialogues were inserted
print(f"\nInserted {inserted_count} new dialogues.")

# Close the database connection
conn.close()
