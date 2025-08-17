from pathlib import Path

MALLET_HOME = Path("./mallet-2.0.8")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# --- File Paths ---
INPUT_CSV_PATH = DATA_DIR / "data.csv"
MALLET_BIN_PATH = str(MALLET_HOME / "bin" / "mallet")
CLEAN_TEXT_PATH= RESULTS_DIR / "clean_text.txt"

# --- Model Parameters ---
NUM_RUNS = 3
TOPIC_START = 5
TOPIC_LIMIT = 71
TOPIC_STEP = 5

# --- Data Columns ---
TEXT_COLUMN = 'Body'
TITLE_COLUMN = 'Title'
