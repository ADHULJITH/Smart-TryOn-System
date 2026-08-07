"""
config.py

Central configuration for the Smart-TryOn-System project.
"""

from pathlib import Path
import torch


# ==========================================================
# Project Root
# ==========================================================

# Current Colab project location
PROJECT_ROOT = Path("/content/Smart-TryOn-System")


# ==========================================================
# Data
# ==========================================================

DATA_DIR = PROJECT_ROOT / "Data"

TEST_PAIRS_DIR = DATA_DIR / "test_pairs"

PERSON_DIR = TEST_PAIRS_DIR / "person"

GARMENT_DIR = TEST_PAIRS_DIR / "garment"

PAIRS_MANIFEST = TEST_PAIRS_DIR / "pairs_manifest.csv"


# ==========================================================
# Edge Cases
# ==========================================================

EDGE_CASES_DIR = DATA_DIR / "edge_cases"


# ==========================================================
# Output Directories
# ==========================================================

OUTPUT_DIR = DATA_DIR / "Output"

Q1_OUTPUT = OUTPUT_DIR / "Q1"

Q2_OUTPUT = OUTPUT_DIR / "Q2"

Q3_OUTPUT = OUTPUT_DIR / "Q3"

Q4_OUTPUT = OUTPUT_DIR / "Q4"

Q5_OUTPUT = OUTPUT_DIR / "Q5"


# ==========================================================
# Q2 Output Directories
# ==========================================================

Q2_PARSING_DIR = Q2_OUTPUT / "parsing_maps"

Q2_AGNOSTIC_DIR = Q2_OUTPUT / "agnostic"

Q2_GARMENT_MASK_DIR = Q2_OUTPUT / "garment_masks"

Q2_GARMENT_BINARY_DIR = Q2_OUTPUT / "garment_masks_binary"

Q2_VISUALIZATION_DIR = Q2_OUTPUT / "visualizations"

Q2_EDGE_INPUT_DIR = Q2_OUTPUT / "edge_case_input"

Q2_EDGE_PARSING_DIR = Q2_OUTPUT / "edge_case_parsing"

Q2_EDGE_AGNOSTIC_DIR = Q2_OUTPUT / "edge_case_agnostic"


# ==========================================================
# Create Output Directories
# ==========================================================

for folder in [
    Q1_OUTPUT,
    Q2_OUTPUT,
    Q3_OUTPUT,
    Q4_OUTPUT,
    Q5_OUTPUT,

    Q2_PARSING_DIR,
    Q2_AGNOSTIC_DIR,
    Q2_GARMENT_MASK_DIR,
    Q2_GARMENT_BINARY_DIR,
    Q2_VISUALIZATION_DIR,
    Q2_EDGE_INPUT_DIR,
    Q2_EDGE_PARSING_DIR,
    Q2_EDGE_AGNOSTIC_DIR
]:

    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================================================
# Florence-2 Model
# ==========================================================

FLORENCE_MODEL = "microsoft/Florence-2-base"


# ==========================================================
# Device
# ==========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DTYPE = (
    torch.float16
    if DEVICE == "cuda"
    else torch.float32
)


# ==========================================================
# Image
# ==========================================================

IMAGE_WIDTH = 768

IMAGE_HEIGHT = 1024


# ==========================================================
# Generation
# ==========================================================

MAX_NEW_TOKENS = 128

NUM_BEAMS = 3


# ==========================================================
# Seed
# ==========================================================

SEED = 42