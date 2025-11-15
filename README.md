# RoboDialogues
Robos in Movies

## How to Use
- Clone the repository.
- Run `dialog.py` to extract the dialogues and store them in an SQLite database.

## Description
This project extracts dialogues for AI/robot characters from movies, stores them in a database, and offers easy access to these dialogues for analysis.

## Setup Instructions

1. **Install the required dependencies**:
    - You can install all required Python packages by running the following command:

    ```bash
    pip install -r requirements.txt
    ```

2. **Download the Cornell Movie-Dialogs Corpus**:
    - You will need to download the Cornell Movie-Dialogs Corpus. You can do this by running the following code:

    ```python
    from convokit import Corpus, download
    corpus = Corpus(filename=download("movie-corpus"))
    ```

    - For more details on how to use the corpus, visit the official documentation: [Cornell Movie-Dialogs Corpus](https://convokit.cornell.edu/documentation/movie.html)

3. **Additional Setup**:
    - Follow the installation instructions on the Convokit documentation page for further setup: [Convokit Installation](https://convokit.cornell.edu/documentation/install.html)
