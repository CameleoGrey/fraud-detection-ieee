import numpy as np
import pandas as pd
import gc
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
import os
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import catboost as cb
import datetime
from src.research.fdi.config import config, Config, logger


class DataLoader:
    """Handles loading and initial preprocessing of data"""

    def __init__(self, config: Config):
        self.config = config
        # fmt: off
        self.str_type = [
            "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
            "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
            "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29",
            "id_30", "id_31", "id_33", "id_34", "id_35", "id_36", "id_37", "id_38",
            "DeviceType", "DeviceInfo"
        ]
        # fmt: on
        self.str_type += [
            f"id-{i}"
            for i in [12, 15, 16, 23, 27, 28, 29, 30, 31, 33, 34, 35, 36, 37, 38]
        ]

    def _get_base_columns(self) -> List[str]:
        """Get base columns to load"""

        # fmt: off
        cols = [
            "TransactionID", "TransactionDT", "TransactionAmt",
            "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
            "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain",
            "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11",
            "C12", "C13", "C14", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8",
            "D9", "D10", "D11", "D12", "D13", "D14", "D15", "M1", "M2", "M3", "M4",
            "M5", "M6", "M7", "M8", "M9"
        ]
        # fmt: on
        cols += [f"V{x}" for x in self.config.v_columns]
        return cols

    def _get_dtypes(self, cols: List[str]) -> Dict[str, str]:
        """Get data types for columns"""
        dtypes = {}
        id_cols = [f"id_0{x}" for x in range(1, 10)] + [
            f"id_{x}" for x in range(10, 34)
        ]
        id_cols += [f"id-0{x}" for x in range(1, 10)] + [
            f"id-{x}" for x in range(10, 34)
        ]

        for c in cols + id_cols:
            dtypes[c] = "float32"
        for c in self.str_type:
            dtypes[c] = "category"

        return dtypes

    def _check_parquet_exists(self) -> bool:
        """Check if all required parquet files exist"""
        parquet_files = [
            self.config.train_transaction_parquet,
            self.config.train_identity_parquet,
            self.config.test_transaction_parquet,
            self.config.test_identity_parquet
        ]
        return all(os.path.exists(f) for f in parquet_files)

    def _load_from_parquet(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Load data from parquet files"""
        logger.info("Loading data from parquet files...")

        # Load train data
        x_train = pd.read_parquet(self.config.train_transaction_parquet)
        train_id = pd.read_parquet(self.config.train_identity_parquet)
        x_train = x_train.merge(train_id, how="left", left_index=True, right_index=True)

        # Load test data
        x_test = pd.read_parquet(self.config.test_transaction_parquet)
        test_id = pd.read_parquet(self.config.test_identity_parquet)

        # Fix column names for test identity
        fix = {o: n for o, n in zip(test_id.columns, train_id.columns)}
        test_id.rename(columns=fix, inplace=True)
        x_test = x_test.merge(test_id, how="left", left_index=True, right_index=True)

        # Extract target
        y_train = x_train["isFraud"].copy()
        x_train.drop("isFraud", axis=1, inplace=True)

        del train_id, test_id
        gc.collect()

        logger.info(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
        return x_train, x_test, y_train

    def _build_parquet_files(self) -> None:
        """Build parquet files from CSV files"""
        logger.info("Building parquet files from CSV...")

        cols = self._get_base_columns()
        dtypes = self._get_dtypes(cols)

        # Load and save train transaction
        train_transaction = pd.read_csv(
            f"{self.config.data_dir}/train_transaction.csv",
            index_col="TransactionID",
            dtype=dtypes,
            usecols=cols + ["isFraud"],
        )
        train_transaction.to_parquet(self.config.train_transaction_parquet)
        logger.info(f"Saved train_transaction to {self.config.train_transaction_parquet}")

        # Load and save train identity
        train_identity = pd.read_csv(
            f"{self.config.data_dir}/train_identity.csv",
            index_col="TransactionID",
            dtype=dtypes,
        )
        train_identity.to_parquet(self.config.train_identity_parquet)
        logger.info(f"Saved train_identity to {self.config.train_identity_parquet}")

        # Load and save test transaction
        test_transaction = pd.read_csv(
            f"{self.config.data_dir}/test_transaction.csv",
            index_col="TransactionID",
            dtype=dtypes,
            usecols=cols,
        )
        test_transaction.to_parquet(self.config.test_transaction_parquet)
        logger.info(f"Saved test_transaction to {self.config.test_transaction_parquet}")

        # Load and save test identity
        test_identity = pd.read_csv(
            f"{self.config.data_dir}/test_identity.csv",
            index_col="TransactionID",
            dtype=dtypes,
        )
        test_identity.to_parquet(self.config.test_identity_parquet)
        logger.info(f"Saved test_identity to {self.config.test_identity_parquet}")

        del train_transaction, train_identity, test_transaction, test_identity
        gc.collect()

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Load train and test data from parquet files or build them from CSV"""
        logger.info("Loading data...")

        # Check if parquet files exist, if not build them
        if not self._check_parquet_exists():
            logger.info("Parquet files not found, building from CSV files...")
            # Ensure data directory exists
            Path(self.config.data_dir).mkdir(parents=True, exist_ok=True)
            self._build_parquet_files()
        else:
            logger.info("Loading from existing parquet files...")

        # Load from parquet files
        return self._load_from_parquet()
