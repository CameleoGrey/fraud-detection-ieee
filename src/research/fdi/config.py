import numpy as np
import pandas as pd
import gc
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import catboost as cb
import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuration for the fraud detection pipeline"""

    # Data paths
    data_dir: str = Path("..", "..", "Datasets", "ieee-fraud-detection")
    output_dir: str = "./output"

    # Parquet file paths (stored in same dir as raw files)
    train_transaction_parquet: str = None
    train_identity_parquet: str = None
    test_transaction_parquet: str = None
    test_identity_parquet: str = None

    # Model parameters
    n_estimators: int = 5000
    max_depth: int = 12
    learning_rate: float = 0.02
    subsample: float = 0.8
    colsample_bytree: float = 0.4

    # Training parameters
    n_folds: int = 6
    early_stopping_rounds: int = 200
    eval_metric: str = "AUC"
    task_type: str = "GPU"  # or "GPU"

    # Feature configuration
    start_date: str = "2017-11-30"

    # V columns to use
    v_columns: List[int] = None

    def __post_init__(self):
        """Initialize V columns if not provided"""

        if self.v_columns is None:

            # fmt: off
            self.v_columns = [
                1, 3, 4, 6, 8, 11, 13, 14, 17, 20, 23, 26, 27, 30,
                36, 37, 40, 41, 44, 47, 48, 54, 56, 59, 62, 65, 67, 68, 70,
                76, 78, 80, 82, 86, 88, 89, 91, 107, 108, 111, 115, 117, 120, 121, 123,
                124, 127, 129, 130, 136, 138, 139, 142, 147, 156, 162, 165, 160, 166,
                178, 176, 173, 182, 187, 203, 205, 207, 215, 169, 171, 175, 180, 185,
                188, 198, 210, 209, 218, 223, 224, 226, 228, 229, 235, 240, 258, 257,
                253, 252, 260, 261, 264, 266, 267, 274, 277, 220, 221, 234, 238, 250,
                271, 294, 284, 285, 286, 291, 297, 303, 305, 307, 309, 310, 320, 281,
                283, 289, 296, 301, 314
            ]
            # fmt: on
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Initialize parquet file paths
        if self.train_transaction_parquet is None:
            self.train_transaction_parquet = str(Path(self.data_dir) / "train_transaction.parquet")
        if self.train_identity_parquet is None:
            self.train_identity_parquet = str(Path(self.data_dir) / "train_identity.parquet")
        if self.test_transaction_parquet is None:
            self.test_transaction_parquet = str(Path(self.data_dir) / "test_transaction.parquet")
        if self.test_identity_parquet is None:
            self.test_identity_parquet = str(Path(self.data_dir) / "test_identity.parquet")


config = Config(
    data_dir=Path("..", "..", "Datasets", "ieee-fraud-detection"),
    output_dir=Path("data", "interim"),
    n_estimators=5000,
    max_depth=4,
    learning_rate=0.02,
    n_folds=6,
    task_type="GPU",
)
