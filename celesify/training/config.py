"""
Configuration and environment handling for training pipeline.

This module handles:
- Reading and validating environment variables
- Managing training configuration (n_iter, cv_splits, etc.)
- Logging configuration details
"""

from __future__ import annotations

import os

from celesify.core.constants import RANDOM_STATE
from celesify.core.logging import log

SERVICE = "training"


class TrainingConfig:
    """Configuration for the training pipeline.

    Attributes:
        n_iter: Number of iterations for RandomizedSearchCV (default: 20).
        cv_splits: Number of cross-validation folds (default: 5).
        n_jobs: Parallel jobs (-1 for all cores, default: -1).
        max_train_rows: Max rows for training (0 = no limit, default: 0).
        random_state: Random seed for reproducibility (always RANDOM_STATE).
    """

    def __init__(
        self,
        n_iter: int = 20,
        cv_splits: int = 5,
        n_jobs: int = -1,
        max_train_rows: int = 0,
    ):
        """
        Initialize training configuration from arguments or environment variables.

        Environment variables (if set, override defaults):
        - TRAINING_N_ITER: Number of search iterations
        - TRAINING_CV_SPLITS: Number of CV folds
        - TRAINING_N_JOBS: Number of parallel jobs
        - TRAINING_MAX_TRAIN_ROWS: Max training rows for subsampling

        Args:
            n_iter: Default number of iterations.
            cv_splits: Default number of CV splits.
            n_jobs: Default number of jobs.
            max_train_rows: Default max training rows.
        """
        self.n_iter = self._get_int_env("TRAINING_N_ITER", max(1, n_iter))
        self.cv_splits = self._get_int_env("TRAINING_CV_SPLITS", max(2, cv_splits))
        self.n_jobs = self._get_int_env("TRAINING_N_JOBS", n_jobs)
        self.max_train_rows = self._get_int_env("TRAINING_MAX_TRAIN_ROWS", max_train_rows)
        self.random_state = RANDOM_STATE

        self._log_summary()

    @staticmethod
    def _get_int_env(name: str, default: int) -> int:
        """
        Read integer environment variable with fallback.

        Args:
            name: Environment variable name.
            default: Default value if not set or invalid.

        Returns:
            Parsed integer value or default.
        """
        raw = os.getenv(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError:
            log(SERVICE, f"Invalid {name}={raw!r}; using default {default}.")
            return default

    def _log_summary(self) -> None:
        """Log current configuration for debugging."""
        log(
            SERVICE,
            f"Training config: n_iter={self.n_iter}, cv_splits={self.cv_splits}, "
            f"n_jobs={self.n_jobs}, max_train_rows={self.max_train_rows}, "
            f"random_state={self.random_state}",
        )
