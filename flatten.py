import sqlite3
from collections import OrderedDict

# Connect to the database
conn = sqlite3.connect('RobotDialogs.db')
cursor = conn.cursor()

def flatten_dialogues():
    # Step 1: Read all existing rows (this is the state after robodialog.py)
    cursor.execute("""
        SELECT id, movie_name, character_name, dialogue
        FROM dialogues
        ORDER BY character_name, movie_name, id
    """)
    rows = cursor.fetchall()

    # Build: character_name -> OrderedDict(movie_name -> [dialogue, dialogue, ...])
    characters = {}

    for _id, movie_name, character_name, dialogue in rows:
        if character_name not in characters:
            characters[character_name] = OrderedDict()
        movie_dict = characters[character_name]

        # Ensure movie_name bucket exists
        if movie_name not in movie_dict:
            movie_dict[movie_name] = []

        # Collect each utterance text
        movie_dict[movie_name].append(dialogue)

    # Step 2: For each character, flatten their dialogues:
    # - Within each movie: join lines with " | "
    # - Between movies: join movie chunks with " @@ "
    flattened_entries = []  # list of (movie_name_combined, character_name, dialogue_combined)

    for character_name, movie_dict in characters.items():
        movie_name_parts = []
        dialogue_chunks = []

        for movie_name, utterances in movie_dict.items():
            # Dialogues from this character in this specific movie
            # joined by " | " (different lines in same movie)
            per_movie_dialogue = " | ".join(utterances)
            dialogue_chunks.append(per_movie_dialogue)

            # Record which movies this character appears in
            movie_name_parts.append(movie_name)

        # Join movie names by "@@" to show multiple movies if needed
        combined_movie_name = " @@ ".join(movie_name_parts)
        # Join per-movie dialogue chunks by "@@" (between movies)
        combined_dialogue = " @@ ".join(dialogue_chunks)

        flattened_entries.append((combined_movie_name, character_name, combined_dialogue))

    # Step 3: Replace the table contents with the flattened version

    # Delete all existing rows
    cursor.execute("DELETE FROM dialogues;")
    conn.commit()

    # Reset AUTOINCREMENT so ids start from 1 again
    cursor.execute("DELETE FROM sqlite_sequence WHERE name = 'dialogues';")
    conn.commit()

    # Insert one row per character (movie_name, character_name, dialogue)
    cursor.executemany("""
        INSERT INTO dialogues (movie_name, character_name, dialogue)
        VALUES (?, ?, ?)
    """, flattened_entries)
    conn.commit()

    print(f"Flattened dialogues for {len(flattened_entries)} characters into one row each.")

if __name__ == "__main__":
    flatten_dialogues()
    conn.close()
