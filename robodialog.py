import json
import sqlite3

# AI/robot characters dictionary with their respective speaker IDs and movie IDs
robot_characters = {
    "HAL 9000": {"speaker_id": "u56", "movie_id": "m3"},
    "Bishop": {"speaker_id": "u229", "movie_id": "m15"},
    "Simone/Viktor": {"speaker_id": "u2783", "movie_id": "m181"},
    "Data": {"speaker_id": "u2899", "movie_id": "m191"},
    "Borg Queen": {"speaker_id": "u2992", "movie_id": "m196"},
    "Data": {"speaker_id": "u2993", "movie_id": "m196"},
    "Data": {"speaker_id": "u3025", "movie_id": "m198"},
    "C-3PO": {"speaker_id": "u5103", "movie_id": "m337"},
    "Terminator": {"speaker_id": "u8077", "movie_id": "m547"},
    "Terminator": {"speaker_id": "u8098", "movie_id": "m549"},
    "Leeloo": {"speaker_id": "u85", "movie_id": "m5"},
    
    # New AI characters from The Matrix
    "Agent Smith": {"speaker_id": "u6517", "movie_id": "m433"},
    "Agent Jones": {"speaker_id": "u6516", "movie_id": "m433"},
    "Oracle": {"speaker_id": "u6524", "movie_id": "m433"}
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
    
    # Add The Matrix movie
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

# Create a table for storing the dialogues (movie_name, character_name, dialogue)
cursor.execute('''
CREATE TABLE IF NOT EXISTS dialogues (
    movie_name TEXT,
    character_name TEXT,
    dialogue TEXT
)
''')

# Function to insert dialogues into the database
def insert_dialogue(movie_name, character_name, dialogue):
    cursor.execute('''
    INSERT INTO dialogues (movie_name, character_name, dialogue)
    VALUES (?, ?, ?)
    ''', (movie_name, character_name, dialogue))
    conn.commit()

# Process the utterances data and filter for the AI/robot characters
inserted_count = 0

# Now process all the utterances
for utterance in utterances_data:
    speaker_id = utterance['speaker']
    dialogue = utterance['text']
    conversation_id = utterance['conversation_id']
    
    # Get the movie ID from the utterance's metadata and map it to the movie name
    movie_id = utterance['meta']['movie_id']
    movie_name = movie_id_dict.get(movie_id, 'Unknown Movie')
    
    # Check if this utterance corresponds to an AI/robot character
    for character_name, data in robot_characters.items():
        if data['speaker_id'] == speaker_id and data['movie_id'] == movie_id:
            # Check if movie_name is correctly assigned
            if movie_name == 'Unknown Movie':
                # If movie name is still 'Unknown Movie', check if it is in conversations_data
                for conversation in conversations_data.values():
                    if conversation['meta']['movie_idx'] == movie_id:
                        movie_name = conversation['meta'].get('movie_name', 'Unknown Movie')
                        break
            
            # Insert the dialogue into the database if the movie name is found
            if movie_name != 'Unknown Movie':
                insert_dialogue(movie_name, character_name, dialogue)
                inserted_count += 1

# Output how many dialogues were inserted
print(f"\nInserted {inserted_count} new dialogues.")

# Close the database connection
conn.close()

print("Dialogues for AI/robot characters have been successfully extracted and inserted into the RobotDialogs.db database.")
