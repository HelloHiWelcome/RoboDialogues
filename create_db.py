import os
import re
import sqlite3

# ============
# CONFIG
# ============

# This is the folder containing all of the movie folders (e.g. Alien_0078748, Her_1798709, etc.)
ROOT_DIR = r"C:/Users/Hp/Documents/robodialogs/movie_character_texts"  # <--- CHANGE THIS TO YOUR PATH

# Mapping from movie folder names (as they appear in your dataset) to a list of AI/robot
# characters that appear in that movie.  This per‑movie configuration allows us to
# include characters only in the films where they are actually robots.  For example,
# "Vanessa Kensington" is only listed for the second Austin Powers movie, because
# she is a Fembot only in that film.
ai_config = {
    # Westworld TV series (hosts are robots/AI)
    "Westworld_0475784": [
        "Armistice", "Bernard Lowe", "Hector Escaton",
        "Dolores Abernathy", "Lawrence", "Maeve Millay",
    ],
    # Austin Powers films
    # In the first film, only generic Fembots count as robots; Vanessa is human.
    "Austin Powers International Man of Mystery_0118655": [
        "Fembot",
    ],
    # In the second film Vanessa is revealed to be a Fembot as well.
    "Austin Powers The Spy Who Shagged Me_0145660": [
        "Fembot", "Vanessa Kensington",
    ],
    # Interstellar – NASA‑built robots
    "Interstellar_0816692": ["TARS", "CASE"],
    # Moon – lunar base AI/robots
    "Moon_1182345": ["Eve", "GERTY"],
    # Star Trek: The Motion Picture – V'Ger's probe companion Ilia
    "Star Trek The Motion Picture_0079945": ["Ilia"],
    # WarGames – NORAD supercomputer
    "WarGames_0086567": ["Joshua"],
    # Her – intelligent operating system
    "Her_1798709": ["Samantha"],
    # Alien: Resurrection – Auton crew member
    "Alien Resurrection_0118583": ["Call"],
}

# Output SQLite database (it will be created if it doesn't exist)
#
# The database will be created in the same directory as this script.  You can
# change the filename here if you prefer a different location.  For example,
# to save it in a specific folder, use an absolute path instead of
# ``os.path.dirname(__file__)``.
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MovieScript.db")

# ============
# MAIN LOGIC
# ============

def extract_dialogue_lines(file_path: str) -> list:
    """
    Extracts all dialogue lines from a character file.
    Looks for lines containing 'dialog:' or 'dialogue:' (case-insensitive).

    The Kaggle MovieScripts files use "dialog:" to denote a character's
    spoken lines.  We support both "dialog" and "dialogue" as prefixes
    to be flexible.

    Parameters
    ----------
    file_path : str
        The full path to the character's text file.

    Returns
    -------
    list[str]
        A list of dialogue strings extracted from the file.
    """
    dialogues = []
    # Compile regex once for efficiency.  Match either "dialog" or
    # "dialogue" followed by a colon and capture the rest of the line.
    pattern = re.compile(r"dialog(?:ue)?\s*:\s*(.*)", re.IGNORECASE)
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    text = m.group(1).strip()
                    if text:
                        dialogues.append(text)
    except FileNotFoundError:
        # If the file can't be read, return empty list; caller will skip
        return []
    return dialogues

def create_database(db_path):
    """
    Creates the SQLite database and the table if it doesn't already exist.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS robot_dialogues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            movie_name TEXT,
            character_name TEXT,
            dialogue TEXT,
            synopsis TEXT
        );
    """)
    conn.commit()
    conn.close()

def insert_row(db_path, movie_name, character_name, dialogue_text):
    """
    Inserts a single row into the database.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        INSERT INTO robot_dialogues (movie_name, character_name, dialogue, synopsis)
        VALUES (?, ?, ?, NULL);
    """, (movie_name, character_name, dialogue_text))
    conn.commit()
    conn.close()

def main():
    """
    Main entry point.  Creates the database and then iterates over the
    per‑movie AI configuration.  For each configured movie, the script tries
    to locate the corresponding folder in ``ROOT_DIR`` by matching on the
    numeric ID suffix.  It then extracts dialogue lines for each listed
    AI/robot character and inserts them into the SQLite database.

    Using ``ai_config`` ensures that characters are only treated as robots in
    the movies where they actually appear as such (for example, Vanessa
    Kensington is only considered a Fembot in the second Austin Powers film).
    """
    # Create the database & table if it doesn't exist
    create_database(DB_PATH)

    # Iterate over each movie and its associated robot characters
    for configured_folder, character_names in ai_config.items():
        # Derive the human‑readable movie title by stripping the numeric ID
        # and replacing underscores with spaces.  If you prefer a custom
        # mapping, you can define a separate dictionary, but this fallback
        # should work for most cases.
        if "_" in configured_folder:
            title_part, movie_id = configured_folder.rsplit("_", 1)
        else:
            title_part, movie_id = configured_folder, ""
        # Replace underscores with spaces for readability
        movie_title = title_part.replace("_", " ")

        # Attempt to locate the folder in the dataset.  Start by joining
        # ROOT_DIR and the configured folder name exactly.
        matched_folder_path = os.path.join(ROOT_DIR, configured_folder)
        # If it doesn't exist, search for any subfolder that ends with the
        # same numeric ID.  This makes the search resilient to differences
        # in spacing or underscores in the dataset folder names.
        if not os.path.isdir(matched_folder_path):
            for candidate in os.listdir(ROOT_DIR):
                full_candidate = os.path.join(ROOT_DIR, candidate)
                if not os.path.isdir(full_candidate):
                    continue
                if movie_id and candidate.rsplit("_", 1)[-1] == movie_id:
                    matched_folder_path = full_candidate
                    break
        # If the matched path still isn't a directory, warn and continue
        if not os.path.isdir(matched_folder_path):
            print(f"[WARN] Movie folder not found for '{movie_title}': attempted {matched_folder_path}")
            continue

        # For each character associated with this movie, search for dialogue
        for character_name in character_names:
            matched_files = []
            # Build a normalised pattern: remove spaces and underscores and lower‑case
            raw_pattern = character_name.lower().replace(" ", "").replace("_", "")
            # Scan through all text files in the matched folder
            for fname in os.listdir(matched_folder_path):
                if not fname.lower().endswith(".txt"):
                    continue
                name_no_ext = os.path.splitext(fname)[0]
                normalised_fname = name_no_ext.lower().replace(" ", "").replace("_", "")
                if raw_pattern in normalised_fname:
                    matched_files.append(os.path.join(matched_folder_path, fname))
            if not matched_files:
                continue  # no files for this character
            # Extract all dialogue lines from matched files
            all_dialogues = []
            for file_path in matched_files:
                dialogues = extract_dialogue_lines(file_path)
                all_dialogues.extend(dialogues)
            if not all_dialogues:
                continue  # no spoken lines
            # Join dialogues with a separator
            flattened_dialogue = " | ".join(all_dialogues)
            # Insert into database
            insert_row(DB_PATH, movie_title, character_name, flattened_dialogue)
            print(f"[OK] Inserted {len(all_dialogues)} lines for {character_name} in {movie_title}")
    # Final status message
    print("\nAll done! Check for results.")
    print(f"Results are saved to: {DB_PATH}")



if __name__ == "__main__":
    main()
