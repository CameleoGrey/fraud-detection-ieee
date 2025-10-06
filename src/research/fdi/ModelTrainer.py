import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import catboost as cb
import datetime
from src.research.fdi.config import config, Config, logger


class ModelTrainer:
    """Handles model training and evaluation"""

    def __init__(self, config: Config):
        self.config = config

    def train_single_model(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        cols: List[str],
        train_idx: np.ndarray,
        val_idx: np.ndarray,
    ) -> cb.CatBoostClassifier:
        """Train a single CatBoost model"""
        clf = cb.CatBoostClassifier(
            iterations=self.config.n_estimators,
            depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            nan_mode='Min',
            eval_metric=self.config.eval_metric,
            task_type=self.config.task_type,
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose=100,
        )

        clf.fit(
            x_train[cols].iloc[train_idx],
            y_train.iloc[train_idx],
            eval_set=[(x_train[cols].iloc[val_idx], y_train.iloc[val_idx])],
        )

        return clf

    def train_cv_model(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        y_train: pd.Series,
        cols: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Train model with cross-validation"""
        logger.info("Training with cross-validation...")

        oof = np.zeros(len(x_train))
        preds = np.zeros(len(x_test))

        skf = GroupKFold(n_splits=self.config.n_folds)

        for i, (train_idx, val_idx) in enumerate(
            skf.split(x_train, y_train, groups=x_train["DT_M"])
        ):
            month = x_train.iloc[val_idx]["DT_M"].iloc[0]
            logger.info(f"Fold {i} - Withholding month {month}")
            logger.info(
                f"Train rows: {len(train_idx)}, Validation rows: {len(val_idx)}"
            )

            clf = self.train_single_model(x_train, y_train, cols, train_idx, val_idx)

            oof[val_idx] = clf.predict_proba(x_train[cols].iloc[val_idx])[:, 1]
            preds += clf.predict_proba(x_test[cols])[:, 1] / self.config.n_folds

        cv_score = roc_auc_score(y_train, oof)
        logger.info(f"OOF CV Score: {cv_score:.6f}")

        return oof, preds
