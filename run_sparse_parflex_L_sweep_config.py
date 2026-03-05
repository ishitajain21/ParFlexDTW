"""
Config for running Sparse_Parflex.ipynb for multiple L and saving to experiments_train.
Do not run the sweep here; this file only defines constants.
"""

from pathlib import Path

# Chunk lengths to run (as in the user request).
L_VALUES = [
    4000,6000
]

# Root directory for saving systems (one subdir per L: experiments_train/sparse_parflex/L_100, ...).
EXPERIMENTS_TRAIN_ROOT = Path(__file__).resolve().parent / "experiments_test"
SPARSE_PARFLEX_SUBDIR = "sparse_parflex"

# Default feature root for loading F1, F2 (optional; notebook may set file_1, file_2 directly).
FEATURES_ROOT = Path("/home/ijain/ttmp/Chopin_Mazurkas_features/matching")
