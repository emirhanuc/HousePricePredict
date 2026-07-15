"""
House Price Prediction - Data Preprocessing Pipeline

This module performs:

1. Raw train/test data loading
2. Training outlier removal
3. Target transformation
4. Semantic missing-value handling
5. Ordinal feature encoding
6. Feature engineering
7. One-Hot Encoding
8. Numerical feature scaling
9. Processed dataset export
10. Preprocessor artifact export
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    PREPROCESSING_SUMMARY_PATH,
    PREPROCESSOR_PATH,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    REMOVE_OUTLIERS,
    SCALE_NUMERIC_FEATURES,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH
)

from src.utils import (
    create_output_directories,
    get_logger,
    load_csv,
    save_dataframe,
    save_joblib_artifact,
    save_text
)


logger = get_logger(__name__)


# ============================================================
# DATA LOADING
# ============================================================

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and validate raw train and test datasets.

    Returns
    -------
    train:
        Training dataset containing SalePrice.

    test:
        Test dataset without SalePrice.
    """

    logger.info("Loading raw datasets.")

    train = load_csv(
        TRAIN_DATA_PATH,
        "Raw training dataset"
    )

    test = load_csv(
        TEST_DATA_PATH,
        "Raw test dataset"
    )

    required_train_columns = {
        "Id",
        "SalePrice",
        "GrLivArea"
    }

    missing_train_columns = (
        required_train_columns - set(train.columns)
    )

    if missing_train_columns:
        raise ValueError(
            "Missing required training columns: "
            f"{sorted(missing_train_columns)}"
        )

    if "Id" not in test.columns:
        raise ValueError(
            "The raw test dataset does not contain the Id column."
        )

    logger.info(
        "Raw datasets loaded successfully: train=%s, test=%s",
        train.shape,
        test.shape
    )

    return train, test


# ============================================================
# OUTLIER REMOVAL
# ============================================================

def remove_training_outliers(
    train: pd.DataFrame
) -> tuple[pd.DataFrame, int]:
    """
    Remove extreme training observations.

    Houses with GrLivArea above 4000 and SalePrice below 300000
    are treated as unusual observations.

    Test data is never modified.
    """

    if not REMOVE_OUTLIERS:
        logger.info("Training outlier removal is disabled.")
        return train.copy(), 0

    outlier_condition = (
        (train["GrLivArea"] > 4000)
        & (train["SalePrice"] < 300000)
    )

    removed_count = int(outlier_condition.sum())

    cleaned_train = (
        train.loc[~outlier_condition]
        .reset_index(drop=True)
        .copy()
    )

    logger.info(
        "Removed %d training outliers.",
        removed_count
    )

    return cleaned_train, removed_count


# ============================================================
# DATA-TYPE CORRECTION
# ============================================================

def correct_data_types(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame
) -> None:
    """
    Convert numeric-looking category codes into string variables.

    MSSubClass is a building-class code.
    MoSold is the month of sale.

    These values are categorical rather than continuous quantities.
    """

    categorical_numeric_columns = [
        "MSSubClass",
        "MoSold"
    ]

    for dataframe in (train_features, test_features):
        for column in categorical_numeric_columns:
            if column not in dataframe.columns:
                raise KeyError(
                    f"Required column was not found: {column}"
                )

            dataframe[column] = (
                dataframe[column]
                .astype("string")
            )


# ============================================================
# COLUMN TYPE HELPERS
# ============================================================

def get_categorical_columns(
    dataframe: pd.DataFrame
) -> list[str]:
    """
    Return all non-numeric columns.

    This approach avoids deprecated object-only pandas behavior.
    """

    return [
        column
        for column in dataframe.columns
        if not pd.api.types.is_numeric_dtype(
            dataframe[column]
        )
    ]


def get_numeric_columns(
    dataframe: pd.DataFrame
) -> list[str]:
    """Return all numerical columns."""

    return [
        column
        for column in dataframe.columns
        if pd.api.types.is_numeric_dtype(
            dataframe[column]
        )
    ]


# ============================================================
# MISSING-VALUE HANDLING
# ============================================================

def fill_semantic_missing_values(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame
) -> None:
    """
    Handle missing values where NaN represents absence.

    Examples:
    - Missing GarageType means no garage.
    - Missing BsmtQual means no basement.
    - Missing PoolQC means no pool.
    """

    none_columns = [
        "Alley",
        "PoolQC",
        "Fence",
        "MiscFeature",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "MasVnrType"
    ]

    zero_columns = [
        "GarageYrBlt",
        "GarageArea",
        "GarageCars",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "MasVnrArea"
    ]

    for column in none_columns:
        if column not in train_features.columns:
            raise KeyError(
                f"Required categorical column was not found: {column}"
            )

        train_features[column] = (
            train_features[column]
            .fillna("None")
        )

        test_features[column] = (
            test_features[column]
            .fillna("None")
        )

    for column in zero_columns:
        if column not in train_features.columns:
            raise KeyError(
                f"Required numeric column was not found: {column}"
            )

        train_features[column] = (
            train_features[column]
            .fillna(0)
        )

        test_features[column] = (
            test_features[column]
            .fillna(0)
        )


def fill_lot_frontage(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame
) -> None:
    """
    Fill LotFrontage using neighborhood-level training medians.

    Test statistics are never used.
    """

    neighborhood_medians = (
        train_features
        .groupby("Neighborhood")["LotFrontage"]
        .median()
    )

    global_median = float(
        train_features["LotFrontage"].median()
    )

    train_features["LotFrontage"] = (
        train_features["LotFrontage"]
        .fillna(
            train_features["Neighborhood"].map(
                neighborhood_medians
            )
        )
        .fillna(global_median)
    )

    test_features["LotFrontage"] = (
        test_features["LotFrontage"]
        .fillna(
            test_features["Neighborhood"].map(
                neighborhood_medians
            )
        )
        .fillna(global_median)
    )


def fill_remaining_missing_values(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame
) -> None:
    """
    Fill remaining missing values using training statistics.

    Categorical columns:
        Training mode.

    Numerical columns:
        Training median.
    """

    categorical_columns = get_categorical_columns(
        train_features
    )

    for column in categorical_columns:
        mode_values = (
            train_features[column]
            .mode(dropna=True)
        )

        fill_value = (
            "Unknown"
            if mode_values.empty
            else mode_values.iloc[0]
        )

        train_features[column] = (
            train_features[column]
            .fillna(fill_value)
        )

        test_features[column] = (
            test_features[column]
            .fillna(fill_value)
        )

    numeric_columns = get_numeric_columns(
        train_features
    )

    for column in numeric_columns:
        median_value = float(
            train_features[column].median()
        )

        train_features[column] = (
            train_features[column]
            .fillna(median_value)
        )

        test_features[column] = (
            test_features[column]
            .fillna(median_value)
        )


# ============================================================
# ORDINAL ENCODING
# ============================================================

def encode_ordinal_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame
) -> None:
    """
    Convert naturally ordered categories into numeric rankings.
    """

    quality_mapping = {
        "None": 0,
        "Po": 1,
        "Fa": 2,
        "TA": 3,
        "Gd": 4,
        "Ex": 5
    }

    ordinal_mappings: dict[str, dict[str, int]] = {
        "ExterQual": quality_mapping,
        "ExterCond": quality_mapping,
        "HeatingQC": quality_mapping,
        "KitchenQual": quality_mapping,
        "FireplaceQu": quality_mapping,
        "GarageQual": quality_mapping,
        "GarageCond": quality_mapping,
        "BsmtQual": quality_mapping,
        "BsmtCond": quality_mapping,
        "PoolQC": quality_mapping,

        "BsmtExposure": {
            "None": 0,
            "No": 1,
            "Mn": 2,
            "Av": 3,
            "Gd": 4
        },

        "BsmtFinType1": {
            "None": 0,
            "Unf": 1,
            "LwQ": 2,
            "Rec": 3,
            "BLQ": 4,
            "ALQ": 5,
            "GLQ": 6
        },

        "BsmtFinType2": {
            "None": 0,
            "Unf": 1,
            "LwQ": 2,
            "Rec": 3,
            "BLQ": 4,
            "ALQ": 5,
            "GLQ": 6
        },

        "GarageFinish": {
            "None": 0,
            "Unf": 1,
            "RFn": 2,
            "Fin": 3
        },

        "PavedDrive": {
            "N": 0,
            "P": 1,
            "Y": 2
        },

        "CentralAir": {
            "N": 0,
            "Y": 1
        },

        "LotShape": {
            "IR3": 0,
            "IR2": 1,
            "IR1": 2,
            "Reg": 3
        },

        "Utilities": {
            "ELO": 0,
            "NoSeWa": 1,
            "NoSewr": 2,
            "AllPub": 3
        },

        "LandSlope": {
            "Sev": 0,
            "Mod": 1,
            "Gtl": 2
        },

        "Electrical": {
            "Mix": 0,
            "FuseP": 1,
            "FuseF": 2,
            "FuseA": 3,
            "SBrkr": 4
        },

        "Functional": {
            "Sal": 0,
            "Sev": 1,
            "Maj2": 2,
            "Maj1": 3,
            "Mod": 4,
            "Min2": 5,
            "Min1": 6,
            "Typ": 7
        },

        "Fence": {
            "None": 0,
            "MnWw": 1,
            "GdWo": 2,
            "MnPrv": 3,
            "GdPrv": 4
        }
    }

    for column, mapping in ordinal_mappings.items():
        for dataframe_name, dataframe in (
            ("train", train_features),
            ("test", test_features)
        ):
            if column not in dataframe.columns:
                raise KeyError(
                    f"Required ordinal column was not found: {column}"
                )

            observed_values = set(
                dataframe[column]
                .dropna()
                .astype(str)
                .unique()
            )

            unknown_values = (
                observed_values - set(mapping.keys())
            )

            if unknown_values:
                raise ValueError(
                    f"Unknown categories in {dataframe_name} "
                    f"column '{column}': "
                    f"{sorted(unknown_values)}"
                )

            dataframe[column] = (
                dataframe[column]
                .map(mapping)
                .astype(float)
            )


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_engineered_features(
    dataframe: pd.DataFrame
) -> None:
    """
    Create additional explanatory housing features.
    """

    dataframe["TotalSF"] = (
        dataframe["TotalBsmtSF"]
        + dataframe["1stFlrSF"]
        + dataframe["2ndFlrSF"]
    )

    dataframe["TotalBathrooms"] = (
        dataframe["FullBath"]
        + 0.5 * dataframe["HalfBath"]
        + dataframe["BsmtFullBath"]
        + 0.5 * dataframe["BsmtHalfBath"]
    )

    dataframe["TotalPorchSF"] = (
        dataframe["OpenPorchSF"]
        + dataframe["EnclosedPorch"]
        + dataframe["3SsnPorch"]
        + dataframe["ScreenPorch"]
        + dataframe["WoodDeckSF"]
    )

    dataframe["HouseAge"] = (
        dataframe["YrSold"]
        - dataframe["YearBuilt"]
    ).clip(lower=0)

    dataframe["RemodelAge"] = (
        dataframe["YrSold"]
        - dataframe["YearRemodAdd"]
    ).clip(lower=0)

    calculated_garage_age = (
        dataframe["YrSold"]
        - dataframe["GarageYrBlt"]
    ).clip(lower=0)

    dataframe["GarageAge"] = np.where(
        dataframe["GarageYrBlt"] > 0,
        calculated_garage_age,
        0
    )

    dataframe["HasPool"] = (
        dataframe["PoolArea"] > 0
    ).astype(int)

    dataframe["HasGarage"] = (
        dataframe["GarageArea"] > 0
    ).astype(int)

    dataframe["HasBsmt"] = (
        dataframe["TotalBsmtSF"] > 0
    ).astype(int)

    dataframe["HasFireplace"] = (
        dataframe["Fireplaces"] > 0
    ).astype(int)

    dataframe["WasRemodeled"] = (
        dataframe["YearRemodAdd"]
        != dataframe["YearBuilt"]
    ).astype(int)


# ============================================================
# FINAL TRANSFORMER
# ============================================================

def build_preprocessor(
    train_features: pd.DataFrame
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """
    Create the final sklearn ColumnTransformer.
    """

    categorical_columns = get_categorical_columns(
        train_features
    )

    numeric_columns = get_numeric_columns(
        train_features
    )

    numeric_transformer = (
        StandardScaler()
        if SCALE_NUMERIC_FEATURES
        else "passthrough"
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_columns
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                ),
                categorical_columns
            )
        ],
        remainder="drop"
    )

    return (
        preprocessor,
        numeric_columns,
        categorical_columns
    )


def clean_feature_names(
    feature_names: np.ndarray
) -> list[str]:
    """
    Remove sklearn transformer prefixes.
    """

    return [
        str(feature_name)
        .replace("numeric__", "")
        .replace("categorical__", "")
        for feature_name in feature_names
    ]


# ============================================================
# VALIDATION
# ============================================================

def validate_processed_outputs(
    processed_train: pd.DataFrame,
    processed_test: pd.DataFrame
) -> None:
    """
    Validate final train and test outputs.
    """

    train_missing_count = int(
        processed_train.isna().sum().sum()
    )

    test_missing_count = int(
        processed_test.isna().sum().sum()
    )

    if train_missing_count != 0:
        raise ValueError(
            "Missing values remain in processed training data: "
            f"{train_missing_count}"
        )

    if test_missing_count != 0:
        raise ValueError(
            "Missing values remain in processed test data: "
            f"{test_missing_count}"
        )

    train_feature_count = (
        processed_train.shape[1] - 1
    )

    test_feature_count = (
        processed_test.shape[1] - 1
    )

    if train_feature_count != test_feature_count:
        raise ValueError(
            "Processed train and test feature counts do not match. "
            f"Train={train_feature_count}, "
            f"Test={test_feature_count}"
        )


# ============================================================
# SUMMARY REPORT
# ============================================================

def save_preprocessing_summary(
    original_train_shape: tuple[int, int],
    original_test_shape: tuple[int, int],
    processed_train: pd.DataFrame,
    processed_test: pd.DataFrame,
    outlier_count: int,
    numeric_count: int,
    categorical_count: int
) -> None:
    """
    Save a human-readable preprocessing summary.
    """

    train_feature_count = (
        processed_train.shape[1] - 1
    )

    test_feature_count = (
        processed_test.shape[1] - 1
    )

    summary = f"""
HOUSE PRICE PREPROCESSING SUMMARY
=================================

Original train shape : {original_train_shape}
Original test shape  : {original_test_shape}

Removed outliers     : {outlier_count}

Processed train shape: {processed_train.shape}
Processed test shape : {processed_test.shape}

Numeric features before final transform    : {numeric_count}
Categorical features before final transform: {categorical_count}

Final train feature count: {train_feature_count}
Final test feature count : {test_feature_count}

Missing train values: {processed_train.isna().sum().sum()}
Missing test values : {processed_test.isna().sum().sum()}

Target transformation:
SalePrice_log = log1p(SalePrice)

Processed train output:
{PROCESSED_TRAIN_PATH}

Processed test output:
{PROCESSED_TEST_PATH}

Saved preprocessing transformer:
{PREPROCESSOR_PATH}
""".strip()

    save_text(
        summary,
        PREPROCESSING_SUMMARY_PATH
    )

    print("\n" + summary)


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_preprocessing() -> None:
    """
    Execute the complete preprocessing pipeline.
    """

    logger.info("Preprocessing pipeline started.")

    create_output_directories()

    train, test = load_raw_data()

    original_train_shape = train.shape
    original_test_shape = test.shape

    test_ids = test["Id"].copy()

    train, outlier_count = remove_training_outliers(
        train
    )

    target = (
        np.log1p(train["SalePrice"])
        .rename("SalePrice_log")
    )

    train_features = (
        train
        .drop(columns=["SalePrice", "Id"])
        .copy()
    )

    test_features = (
        test
        .drop(columns=["Id"])
        .copy()
    )

    correct_data_types(
        train_features,
        test_features
    )

    fill_semantic_missing_values(
        train_features,
        test_features
    )

    fill_lot_frontage(
        train_features,
        test_features
    )

    fill_remaining_missing_values(
        train_features,
        test_features
    )

    encode_ordinal_features(
        train_features,
        test_features
    )

    create_engineered_features(
        train_features
    )

    create_engineered_features(
        test_features
    )

    remaining_train_missing = int(
        train_features.isna().sum().sum()
    )

    remaining_test_missing = int(
        test_features.isna().sum().sum()
    )

    if remaining_train_missing != 0:
        raise ValueError(
            "Missing values remain before final train transform: "
            f"{remaining_train_missing}"
        )

    if remaining_test_missing != 0:
        raise ValueError(
            "Missing values remain before final test transform: "
            f"{remaining_test_missing}"
        )

    (
        preprocessor,
        numeric_columns,
        categorical_columns
    ) = build_preprocessor(train_features)

    logger.info(
        "Applying final transformation: numeric=%d, categorical=%d",
        len(numeric_columns),
        len(categorical_columns)
    )

    processed_train_array = (
        preprocessor.fit_transform(
            train_features
        )
    )

    processed_test_array = (
        preprocessor.transform(
            test_features
        )
    )

    feature_names = clean_feature_names(
        preprocessor.get_feature_names_out()
    )

    processed_train = pd.DataFrame(
        processed_train_array,
        columns=feature_names
    )

    processed_test = pd.DataFrame(
        processed_test_array,
        columns=feature_names
    )

    processed_train.insert(
        0,
        "SalePrice_log",
        target.to_numpy()
    )

    processed_test.insert(
        0,
        "Id",
        test_ids.to_numpy()
    )

    validate_processed_outputs(
        processed_train,
        processed_test
    )

    save_dataframe(
        processed_train,
        PROCESSED_TRAIN_PATH
    )

    save_dataframe(
        processed_test,
        PROCESSED_TEST_PATH
    )

    save_joblib_artifact(
        preprocessor,
        PREPROCESSOR_PATH
    )

    save_preprocessing_summary(
        original_train_shape=original_train_shape,
        original_test_shape=original_test_shape,
        processed_train=processed_train,
        processed_test=processed_test,
        outlier_count=outlier_count,
        numeric_count=len(numeric_columns),
        categorical_count=len(categorical_columns)
    )

    logger.info(
        "Processed datasets and transformer saved successfully."
    )

    logger.info(
        "Preprocessing pipeline completed successfully."
    )


if __name__ == "__main__":
    run_preprocessing()