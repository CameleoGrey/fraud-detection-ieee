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
from src.research.fdi.DataLoader import DataLoader
from src.research.fdi.FeatureEngineer import FeatureEngineer
from src.research.fdi.ModelTrainer import ModelTrainer


class FraudDetectionPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config: Config):
        self.config = config
        self.data_loader = DataLoader(config)
        self.feature_engineer = FeatureEngineer(config)
        self.model_trainer = ModelTrainer(config)

    def validate_pipeline_inputs(self) -> None:
        """Validate that all necessary inputs and configurations are available"""
        self._validate_config()
        self._validate_data_availability()
        self._validate_output_directory()

    def _validate_config(self) -> None:
        """Validate configuration parameters"""
        if not self.config:
            raise ValueError("Configuration object is required")

        if not hasattr(self.config, 'data_dir') or not self.config.data_dir:
            raise ValueError("Data directory must be specified in config")

        if not hasattr(self.config, 'output_dir') or not self.config.output_dir:
            raise ValueError("Output directory must be specified in config")

    def _validate_data_availability(self) -> None:
        """Validate that required data files are available"""
        required_files = [
            f"{self.config.data_dir}/sample_submission.csv"
        ]

        # Additional validation could be added here for train/test files
        # based on the specific data loading implementation

    def _validate_output_directory(self) -> None:
        """Ensure output directory exists or create it"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    def run_stage1(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Run stage 1 of the pipeline - base model with basic features"""
        self._log_stage_header("STAGE 1: Base Model")

        # Load and prepare data
        x_train, x_test, y_train = self._load_and_preprocess_data()

        # Apply stage 1 feature engineering
        self._apply_stage1_feature_engineering(x_train, x_test)

        # Get feature columns for modeling
        feature_columns = self.feature_engineer.get_feature_columns(x_train, stage=1)

        # Add time features for cross-validation
        self.feature_engineer.add_time_features(x_train, x_test)

        # Train model and generate predictions
        out_of_fold_predictions, test_predictions = self._train_and_predict(
            x_train, x_test, y_train, feature_columns
        )

        # Save results
        self._save_stage1_results(x_train, out_of_fold_predictions, test_predictions)

        logger.info("Stage 1 complete!")
        return x_train, x_test, y_train

    def _load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        """Load data and apply basic preprocessing"""
        return self.data_loader.load_data()

    def _apply_stage1_feature_engineering(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame
    ) -> None:
        """Apply stage 1 feature engineering steps"""
        self.feature_engineer.normalize_d_columns(x_train, x_test)
        self.feature_engineer.label_encode_and_reduce_memory(x_train, x_test)
        self.feature_engineer.engineer_features_stage1(x_train, x_test)

    def _train_and_predict(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame,
        y_train: pd.Series, feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Train model and generate predictions"""
        return self.model_trainer.train_cv_model(x_train, x_test, y_train, feature_columns)

    def _save_stage1_results(
        self, x_train: pd.DataFrame,
        out_of_fold_predictions: np.ndarray,
        test_predictions: np.ndarray
    ) -> None:
        """Save stage 1 results to output files"""
        # Save out-of-fold predictions
        x_train["oof"] = out_of_fold_predictions
        oof_output_path = f"{self.config.output_dir}/oof_cb_stage1.csv"
        x_train.reset_index()[["TransactionID", "oof"]].to_csv(oof_output_path, index=False)

        # Save test predictions
        self._save_test_predictions(test_predictions, "sub_cb_stage1.csv")

    def _save_test_predictions(self, predictions: np.ndarray, filename: str) -> None:
        """Save test predictions to submission file"""
        submission_path = f"{self.config.data_dir}/sample_submission.csv"
        sample_submission = pd.read_csv(submission_path)
        sample_submission["isFraud"] = predictions

        output_path = f"{self.config.output_dir}/{filename}"
        sample_submission.to_csv(output_path, index=False)

    def run_stage2(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame, y_train: pd.Series
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run stage 2 of the pipeline - enhanced model with advanced features"""
        self._log_stage_header("STAGE 2: Enhanced Model")

        # Apply stage 2 feature engineering (advanced features)
        self.feature_engineer.engineer_features_stage2(x_train, x_test)

        # Get feature columns for stage 2 modeling
        feature_columns = self.feature_engineer.get_feature_columns(x_train, stage=2)

        # Train enhanced model
        out_of_fold_predictions, test_predictions = self._train_and_predict(
            x_train, x_test, y_train, feature_columns
        )

        # Save stage 2 results
        self._save_stage2_results(x_train, out_of_fold_predictions, test_predictions)

        logger.info("Stage 2 complete!")
        return out_of_fold_predictions, test_predictions

    def _save_stage2_results(
        self, x_train: pd.DataFrame,
        out_of_fold_predictions: np.ndarray,
        test_predictions: np.ndarray
    ) -> None:
        """Save stage 2 results to output files"""
        # Save out-of-fold predictions
        x_train["oof"] = out_of_fold_predictions
        oof_output_path = f"{self.config.output_dir}/oof_cb_stage2.csv"
        x_train.reset_index()[["TransactionID", "oof"]].to_csv(oof_output_path, index=False)

        # Save test predictions
        self._save_test_predictions(test_predictions, "sub_cb_stage2.csv")

    def _log_stage_header(self, stage_name: str) -> None:
        """Log a standardized stage header"""
        header_line = "=" * 50
        logger.info(header_line)
        logger.info(stage_name)
        logger.info(header_line)

    def run_full_pipeline(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run the complete fraud detection pipeline"""
        logger.info("Starting Fraud Detection Pipeline")

        # Validate inputs before starting
        self.validate_pipeline_inputs()

        try:
            # Execute stage 1: Base model
            x_train, x_test, y_train = self.run_stage1()

            # Execute stage 2: Enhanced model
            out_of_fold_predictions, test_predictions = self.run_stage2(
                x_train, x_test, y_train
            )

            self._log_pipeline_completion()

            return out_of_fold_predictions, test_predictions

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise

    def _log_pipeline_completion(self) -> None:
        """Log successful pipeline completion"""
        completion_header = "=" * 50
        logger.info(completion_header)
        logger.info("Pipeline Complete!")
        logger.info(completion_header)
