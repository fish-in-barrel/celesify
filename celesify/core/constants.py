"""Shared constants for preprocessing and training."""

RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COLUMN = "class"
CLASS_ENCODING = {"STAR": 0, "GALAXY": 1, "QSO": 2}
CLASS_LABEL_ORDER = [0, 1, 2]

NON_INFORMATIVE_COLUMNS = [
    "obj_ID",
    "run_ID",
    "rerun_ID",
    "cam_col",
    "field_ID",
    "spec_obj_ID",
    "plate",
    "MJD",
    "fiber_ID",
]

SKEW_CHECK_COLUMNS = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift"]
SKEW_THRESHOLD = 1.5

KAGGLE_DATASET = "fedesoriano/stellar-classification-dataset-sdss17"
KAGGLE_EXPECTED_FILE = "star_classification.csv"
