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


# Constants for feature engineering
SECONDS_PER_DAY = 24 * 60 * 60
D_COLUMN_SKIP_LIST = [1, 2, 3, 5, 9]
D_COLUMN_RANGE = range(1, 16)
C_COLUMN_SKIP = 3
C_COLUMN_RANGE = range(1, 15)
M_COLUMN_RANGE = range(1, 10)
STAGE1_REMOVE_COLS = [
    "TransactionDT", "D6", "D7", "D8", "D9", "D12", "D13", "D14",
    "C3", "M5", "id_08", "id_33", "card4", "id_07", "id_14", "id_21",
    "id_30", "id_32", "id_34"
]
STAGE1_REMOVE_ID_COLS = [f"id_{x}" for x in range(22, 28)]
STAGE2_REMOVE_COLS = ["oof", "DT_M", "day", "uid"]
OUTSIDER_THRESHOLD = 3
BASE_YEAR = 2017


class FeatureEngineer:
    """Handles all feature engineering operations"""

    def __init__(self, config: Config):
        self.config = config

    def normalize_d_columns(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> None:
        """Normalize D columns by subtracting normalized transaction time"""
        logger.info("Normalizing D columns by transaction time...")

        for column_index in D_COLUMN_RANGE:
            if column_index in D_COLUMN_SKIP_LIST:
                continue

            column_name = f"D{column_index}"
            normalized_time = x_train["TransactionDT"] / np.float32(SECONDS_PER_DAY)

            x_train[column_name] = x_train[column_name] - normalized_time
            x_test[column_name] = x_test[column_name] - normalized_time

    def label_encode_and_reduce_memory(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame
    ) -> None:
        """Label encode categorical variables and reduce memory usage"""
        logger.info("Performing label encoding and memory optimization...")

        # Columns that should not be normalized
        excluded_columns = {"TransactionAmt", "TransactionDT"}

        for column_name in x_train.columns:
            if self._is_categorical_column(x_train[column_name]):
                self._encode_categorical_column(x_train, x_test, column_name)

            elif column_name not in excluded_columns:
                self._normalize_numeric_column(x_train, x_test, column_name)

    def _is_categorical_column(self, column: pd.Series) -> bool:
        """Check if a column is categorical or object type"""
        return str(column.dtype) in ["category", "object"]

    def _encode_categorical_column(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame, column_name: str
    ) -> None:
        """Encode a categorical column using factorization"""
        combined_data = pd.concat([x_train[column_name], x_test[column_name]], axis=0)
        encoded_data, _ = combined_data.factorize(sort=True)

        x_train[column_name] = encoded_data[: len(x_train)].astype("int32")
        x_test[column_name] = encoded_data[len(x_train) :].astype("int32")

    def _normalize_numeric_column(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame, column_name: str
    ) -> None:
        """Normalize numeric column by subtracting minimum value"""
        min_value = min(x_train[column_name].min(), x_test[column_name].min())

        x_train[column_name] = x_train[column_name] - np.float32(min_value)
        x_test[column_name] = x_test[column_name] - np.float32(min_value)

        # Fill any missing values with -1
        x_train[column_name].fillna(-1, inplace=True)
        x_test[column_name].fillna(-1, inplace=True)

    def frequency_encode(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame, columns: List[str]
    ) -> None:
        """Apply frequency encoding to specified columns"""
        logger.info(f"Applying frequency encoding to columns: {columns}")

        for column_name in columns:
            frequency_map = self._calculate_frequency_map(x_train, x_test, column_name)
            encoded_column_name = f"{column_name}_FE"

            x_train[encoded_column_name] = x_train[column_name].map(frequency_map).astype("float32")
            x_test[encoded_column_name] = x_test[column_name].map(frequency_map).astype("float32")

    def _calculate_frequency_map(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame, column_name: str
    ) -> Dict:
        """Calculate frequency map for a column"""
        combined_data = pd.concat([x_train[column_name], x_test[column_name]])
        frequency_map = combined_data.value_counts(dropna=True, normalize=True).to_dict()

        # Add default value for missing/NaN cases
        frequency_map[-1] = -1

        return frequency_map

    def combine_features(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame, col1: str, col2: str
    ) -> str:
        """Combine two features by concatenation and apply label encoding"""
        combined_column_name = f"{col1}_{col2}"

        # Create combined feature by concatenating string representations
        x_train[combined_column_name] = (
            x_train[col1].astype(str) + "_" + x_train[col2].astype(str)
        )
        x_test[combined_column_name] = (
            x_test[col1].astype(str) + "_" + x_test[col2].astype(str)
        )

        # Apply label encoding to the combined feature
        self._encode_categorical_column(x_train, x_test, combined_column_name)

        logger.info(f"Created combined feature: {combined_column_name}")
        return combined_column_name

    def aggregate_features(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        main_columns: List[str],
        groupby_columns: List[str],
        aggregations: List[str] = ["mean"],
        fillna: bool = True,
        use_na_replacement: bool = False,
    ) -> None:
        """Aggregate features by group using specified aggregation functions"""
        logger.info(f"Aggregating {main_columns} by {groupby_columns} with {aggregations}")

        for main_column in main_columns:
            for groupby_column in groupby_columns:
                for aggregation in aggregations:
                    self._create_aggregated_feature(
                        x_train, x_test, main_column, groupby_column,
                        aggregation, fillna, use_na_replacement
                    )

    def _create_aggregated_feature(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        main_column: str,
        groupby_column: str,
        aggregation: str,
        fillna: bool,
        use_na_replacement: bool
    ) -> None:
        """Create a single aggregated feature"""
        feature_name = f"{main_column}_{groupby_column}_{aggregation}"

        # Combine train and test data for consistent aggregation
        combined_data = pd.concat([
            x_train[[groupby_column, main_column]],
            x_test[[groupby_column, main_column]]
        ])

        # Replace -1 with NaN if requested (for certain aggregations)
        if use_na_replacement:
            combined_data.loc[combined_data[main_column] == -1, main_column] = np.nan

        # Calculate aggregation
        aggregated_data = (
            combined_data.groupby([groupby_column])[main_column]
            .agg([aggregation])
            .reset_index()
        )

        # Rename the aggregation column and create mapping dictionary
        aggregated_data = aggregated_data.rename(columns={aggregation: feature_name})
        aggregated_data.index = aggregated_data[groupby_column]
        aggregation_map = aggregated_data[feature_name].to_dict()

        # Apply aggregation to train and test sets
        x_train[feature_name] = x_train[groupby_column].map(aggregation_map).astype("float32")
        x_test[feature_name] = x_test[groupby_column].map(aggregation_map).astype("float32")

        # Fill missing values if requested
        if fillna:
            x_train[feature_name].fillna(-1, inplace=True)
            x_test[feature_name].fillna(-1, inplace=True)

    def aggregate_nunique(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        main_columns: List[str],
        groupby_columns: List[str],
    ) -> None:
        """Create features based on unique count aggregations"""
        logger.info(f"Creating nunique aggregations for {main_columns} by {groupby_columns}")

        for main_column in main_columns:
            for groupby_column in groupby_columns:
                self._create_nunique_feature(x_train, x_test, main_column, groupby_column)

    def _create_nunique_feature(
        self,
        x_train: pd.DataFrame,
        x_test: pd.DataFrame,
        main_column: str,
        groupby_column: str
    ) -> None:
        """Create a single nunique feature"""
        feature_name = f"{groupby_column}_{main_column}_ct"

        # Combine train and test data for consistent calculation
        combined_data = pd.concat([
            x_train[[groupby_column, main_column]],
            x_test[[groupby_column, main_column]]
        ], axis=0)

        # Calculate unique count mapping
        nunique_map = (
            combined_data.groupby(groupby_column)[main_column]
            .agg(["nunique"])["nunique"]
            .to_dict()
        )

        # Apply nunique mapping to train and test sets
        x_train[feature_name] = x_train[groupby_column].map(nunique_map).astype("float32")
        x_test[feature_name] = x_test[groupby_column].map(nunique_map).astype("float32")

    def add_time_features(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> None:
        """Add time-based features including month and day calculations"""
        logger.info("Adding time-based features...")

        start_date = datetime.datetime.strptime(self.config.start_date, "%Y-%m-%d")

        for dataframe in [x_train, x_test]:
            self._add_time_features_to_dataframe(dataframe, start_date)

    def _add_time_features_to_dataframe(
        self, dataframe: pd.DataFrame, start_date: datetime.datetime
    ) -> None:
        """Add time features to a single dataframe"""
        # Convert transaction timestamp to datetime
        dataframe["DT_M"] = dataframe["TransactionDT"].apply(
            lambda timestamp: start_date + datetime.timedelta(seconds=timestamp)
        )

        # Calculate month offset from base year
        dataframe["DT_M"] = (
            (dataframe["DT_M"].dt.year - BASE_YEAR) * 12 +
            dataframe["DT_M"].dt.month
        )

        # Calculate day as fraction of day
        dataframe["day"] = dataframe["TransactionDT"] / SECONDS_PER_DAY

    def create_uid_features(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> None:
        """Create features based on unique identifier (UID) aggregations"""
        logger.info("Creating UID-based features...")

        # Create UID column
        self._create_uid_column(x_train, x_test)

        # Apply frequency encoding to UID
        self.frequency_encode(x_train, x_test, ["uid"])

        # Create various UID-based aggregations
        self._create_uid_transaction_aggregations(x_train, x_test)
        self._create_uid_categorical_aggregations(x_train, x_test)
        self._create_uid_email_aggregations(x_train, x_test)
        self._create_uid_additional_aggregations(x_train, x_test)

        # Create outsider feature
        self._create_outsider_feature(x_train, x_test)

    def _create_uid_column(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> None:
        """Create unique identifier column based on card and time features"""
        for dataframe in [x_train, x_test]:
            dataframe["uid"] = (
                dataframe["card1_addr1"].astype(str)
                + "_"
                + np.floor(dataframe["day"] - dataframe["D1"]).astype(str)
            )

    def _create_uid_transaction_aggregations(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame
    ) -> None:
        """Create UID aggregations for transaction-related columns"""
        transaction_columns = ["TransactionAmt", "D4", "D9", "D10", "D15"]

        self.aggregate_features(
            x_train, x_test, transaction_columns, ["uid"],
            ["mean", "std"], fillna=True, use_na_replacement=True
        )

    def _create_uid_categorical_aggregations(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame
    ) -> None:
        """Create UID aggregations for categorical C columns"""
        c_columns = [f"C{x}" for x in C_COLUMN_RANGE if x != C_COLUMN_SKIP]

        self.aggregate_features(
            x_train, x_test, c_columns, ["uid"],
            ["mean"], fillna=True, use_na_replacement=True
        )

    def _create_uid_email_aggregations(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame
    ) -> None:
        """Create UID aggregations for email and M columns"""
        m_columns = [f"M{x}" for x in M_COLUMN_RANGE]

        self.aggregate_features(
            x_train, x_test, m_columns, ["uid"],
            ["mean"], fillna=True, use_na_replacement=True
        )

    def _create_uid_additional_aggregations(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame
    ) -> None:
        """Create additional UID-based aggregations"""
        # Nunique aggregations for various columns
        nunique_columns_1 = ["P_emaildomain", "dist1", "DT_M", "id_02", "cents"]
        self.aggregate_nunique(x_train, x_test, nunique_columns_1, ["uid"])

        # C14 standard deviation
        self.aggregate_features(
            x_train, x_test, ["C14"], ["uid"],
            ["std"], fillna=True, use_na_replacement=True
        )

        # Additional nunique aggregations
        nunique_columns_2 = ["C13", "V314"]
        self.aggregate_nunique(x_train, x_test, nunique_columns_2, ["uid"])

        nunique_columns_3 = ["V127", "V136", "V309", "V307", "V320"]
        self.aggregate_nunique(x_train, x_test, nunique_columns_3, ["uid"])

    def _create_outsider_feature(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> None:
        """Create feature indicating if D1 and D15 values differ significantly"""
        for dataframe in [x_train, x_test]:
            dataframe["outsider15"] = (
                np.abs(dataframe["D1"] - dataframe["D15"]) > OUTSIDER_THRESHOLD
            ).astype("int8")

    def engineer_features_stage1(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame
    ) -> None:
        """Stage 1 feature engineering - basic feature creation and aggregations"""
        logger.info("Starting Stage 1 feature engineering...")

        # Create cents feature (decimal part of transaction amount)
        self._create_cents_feature(x_train, x_test)

        # Apply frequency encoding to basic categorical features
        basic_categorical_columns = ["addr1", "card1", "card2", "card3", "P_emaildomain"]
        self.frequency_encode(x_train, x_test, basic_categorical_columns)

        # Create combined features
        card_addr_combined = self.combine_features(x_train, x_test, "card1", "addr1")
        card_addr_email_combined = self.combine_features(
            x_train, x_test, card_addr_combined, "P_emaildomain"
        )

        # Apply frequency encoding to combined features
        combined_columns = [card_addr_combined, card_addr_email_combined]
        self.frequency_encode(x_train, x_test, combined_columns)

        # Create aggregations by card-based groupings
        self._create_stage1_aggregations(x_train, x_test, combined_columns)

    def _create_cents_feature(self, x_train: pd.DataFrame, x_test: pd.DataFrame) -> None:
        """Create cents feature from transaction amount decimal part"""
        for dataframe in [x_train, x_test]:
            dataframe["cents"] = (
                dataframe["TransactionAmt"] - np.floor(dataframe["TransactionAmt"])
            ).astype("float32")

    def _create_stage1_aggregations(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame, combined_columns: List[str]
    ) -> None:
        """Create aggregations for stage 1 features"""
        aggregation_columns = ["TransactionAmt", "D9", "D11"]
        groupby_columns = ["card1"] + combined_columns

        self.aggregate_features(
            x_train, x_test, aggregation_columns, groupby_columns,
            ["mean", "std"], use_na_replacement=True
        )

    def engineer_features_stage2(
        self, x_train: pd.DataFrame, x_test: pd.DataFrame
    ) -> None:
        """Stage 2 feature engineering - advanced time and UID-based features"""
        logger.info("Starting Stage 2 feature engineering...")

        # Add time-based features
        self.add_time_features(x_train, x_test)

        # Create advanced UID-based features
        self.create_uid_features(x_train, x_test)

    def get_feature_columns(self, x_train: pd.DataFrame, stage: int = 1) -> List[str]:
        """Get list of feature columns to use for modeling, excluding problematic ones"""
        all_columns = list(x_train.columns)

        # Base columns to remove (always excluded)
        columns_to_remove = self._get_base_excluded_columns()

        # Add stage-specific exclusions
        if stage == 2:
            columns_to_remove.extend(STAGE2_REMOVE_COLS)

        # Remove additional problematic columns
        columns_to_remove.extend(self._get_additional_excluded_columns())

        # Filter columns
        feature_columns = [
            col for col in all_columns if col not in columns_to_remove
        ]

        logger.info(f"Using {len(feature_columns)} features for stage {stage}")
        return feature_columns

    def _get_base_excluded_columns(self) -> List[str]:
        """Get columns that are always excluded from modeling"""
        return [
            "TransactionDT", "D6", "D7", "D8", "D9", "D12", "D13", "D14"
        ]

    def _get_additional_excluded_columns(self) -> List[str]:
        """Get additional columns that are excluded due to data quality issues"""
        return (
            STAGE1_REMOVE_COLS +
            STAGE1_REMOVE_ID_COLS
        )
