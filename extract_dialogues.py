import sqlite3
from datasets import load_dataset

# Mapping of movie names (MovieSum format: Title_Year) to the AI/robot characters to extract
character_map = {
    "Blade Runner_1982": ["BATTY", "PRIS", "RACHAEL"],
    "TRON: Legacy_2010": ["KROD"],
    "I, Robot_2004": ["SONNY"],
    "Lost in Space_1998": ["ROBOT"],    
    "Dark Star_1974": ["BOMB #20"],
    "WALL·E_2008": ["AUTO"],
    "Alien: Covenant_2017": ["DAVID", "WALTER"],
    "Upgrade_2018": ["STEM"],
    "Arcade_1993": ["ARCADE"],
    "Prometheus_2012": ["DAVID"],
    "The Hitchhiker's Guide to the Galaxy_2005": ["MARVIN"],
    "Ex Machina_2014": ["AVA"],
    "Iron Man_2008": ["JARVIS"],
    "The Mitchells vs the Machines_2021": ["PAL", "DEBORAHBOT 5000"],
    "Spaceballs_1987": ["DOT"],
}

# Load MovieSum dataset. If you have already downloaded the dataset locally, set
# `data_dir` in load_dataset. Otherwise, this will download the splits from the hub.
print("Loading MovieSum dataset...")
dataset = load_dataset("rohitsaxena/MovieSum", split="train+validation+test")

# Build a lookup dictionary from movie name to script for quick access
movie_lookup = {}
for row in dataset:
    # The dataset uses either 'movie_name' or 'Name' depending on version; same for 'script'/'Script'
    name = row.get('movie_name') or row.get('Name')
    script = row.get('script') or row.get('Script')
    if name and script:
        movie_lookup[name] = script

required_movies = [
    "Blade Runner_1982", "TRON: Legacy_2010", "I, Robot_2004", 
    "Lost in Space_1998", "Nine_2009", "Dark Star_1974", "WALL·E_2008", 
    "The Matrix Reloaded_2003", "Alien_1979", "Alien: Covenant_2017", 
    "Star Trek: The Motion Picture_1979", "Upgrade_2018", "Arcade_1993", 
    "Prometheus_2012", "The Hitchhiker's Guide to the Galaxy_2005", 
    "Ex Machina_2014", "Iron Man_2008", "The Mitchells vs the Machines_2021", 
    "Spaceballs_1987", "The Day the Earth Stood Still_1951", "Virtuosity_1995"
]

# Filter the movie_lookup dictionary
filtered_movie_lookup = {name: script for name, script in movie_lookup.items() if name in required_movies}
# Create/connect to SQLite database
conn = sqlite3.connect('RoboD.db')
cursor = conn.cursor()

# Create table if it doesn't exist.  A new column, `synopsys`, has been added
# to allow storing character synopses later.  The user will populate
# this column in a separate script, so we insert an empty string for now.
cursor.execute('''
CREATE TABLE IF NOT EXISTS dialogues (
    movie_name TEXT,
    character_name TEXT,
    dialogue TEXT,
    synopsys TEXT
)
''')


def extract_dialogues(script: str, character: str) -> str:
    """Extract lines of dialogue spoken by `character` from `script`.

    The MovieSum dataset stores screenplays in a lightweight XML format
    under a root <script> element.  Each <scene> contains a sequence of
    child tags such as <stage_direction>, <scene_description>,
    <character> and <dialogue>.  A <character> tag indicates the
    speaker for the next <dialogue> tag.  This function parses the
    script as XML and extracts the text of <dialogue> tags where the
    preceding <character> matches the provided character name (case-
    insensitive).  If parsing fails (e.g., the script isn't valid XML),
    it falls back to simple heuristics that look for "Character: line"
    patterns and lines where a character name is followed by a dialogue
    line on the next line.

    Returns a single string where individual utterances are separated
    by ' | '.
    """
    import xml.etree.ElementTree as ET

    target = character.strip().lower()
    dialogues: list[str] = []
    parsed = False
    try:
        root = ET.fromstring(script)
        # iterate through all scenes
        for scene in root.findall('.//scene'):
            last_speaker = None
            for elem in scene:
                tag = elem.tag.lower() if hasattr(elem, 'tag') else ''
                text = elem.text.strip() if elem.text else ''
                if tag == 'character':
                    last_speaker = text.lower()
                elif tag == 'dialogue' and last_speaker == target:
                    if text:
                        dialogues.append(text)
        parsed = True
    except Exception:
        # not valid XML; will fall back to heuristic parsing below
        parsed = False

    if not parsed:
        # fallback to simple heuristic parsing of plain text
        lines = script.splitlines()
        for idx, line in enumerate(lines):
            stripped = line.strip()
            # Pattern 1: CHARACTER: dialogue
            if stripped.lower().startswith(target + ":"):
                parts = stripped.split(":", 1)
                if len(parts) > 1 and parts[1].strip():
                    dialogues.append(parts[1].strip())
            # Pattern 2: line is just character name; next line assumed dialogue
            elif stripped.lower() == target and idx + 1 < len(lines):
                next_line = lines[idx + 1].strip()
                if next_line:
                    dialogues.append(next_line)

    return ' | '.join(dialogues)

# Process each movie and extract dialogues for the specified characters
for movie_name, characters in character_map.items():
    script = movie_lookup.get(movie_name)
    if not script:
        print(f"Warning: Script for movie '{movie_name}' not found in dataset.")
        continue
    for character in characters:
        dialogue_text = extract_dialogues(script, character)
        # Insert into database.  Insert an empty string for `synopsys` since
        # summaries will be added later by a separate script.
        cursor.execute(
            'INSERT INTO dialogues (movie_name, character_name, dialogue, synopsys) VALUES (?, ?, ?, ?)',
            (movie_name, character, dialogue_text, '')
        )
        print(f"Extracted {len(dialogue_text.split(' | ')) if dialogue_text else 0} utterances for {character} in {movie_name}.")

# Commit and close database
conn.commit()
conn.close()
print("Extraction complete. Dialogues stored in RobotDialogs.db")
