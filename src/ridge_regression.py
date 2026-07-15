"""
House Price Prediction - Ridge Regression Pipeline

This module trains and evaluates a Ridge Regression model using
the preprocessed Ames Housing dataset.
"""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split

from src.config import (
    PROCESSED_TRAIN_PATH,
    RANDOM_STATE,
    RIDGE_ACTUAL_VS_PREDICTED_PATH,
    RIDGE_ALPHA,
    RIDGE_COEFFICIENTS_PATH,
    RIDGE_METRICS_PATH,
    RIDGE_MODEL_PATH,
    RIDGE_RESIDUAL_PLOT_PATH,
    RIDGE_VALIDATION_PREDICTIONS_PATH,
    TARGET_COLUMN,
    TEST_SIZE
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


def load_processed_training_data() -> pd.DataFrame:
    """Load the processed training dataset."""

    logger.info("Loading processed training dataset.")

    dataframe = load_csv(
        PROCESSED_TRAIN_PATH,
        "Processed training dataset"
    )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found."
        )

    logger.info(
        "Processed training dataset loaded: %s",
        dataframe.shape
    )

    return dataframe


def separate_features_and_target(
    dataframe: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and logarithmic target."""

    features = dataframe.drop(
        columns=[TARGET_COLUMN]
    ).copy()

    target = dataframe[
        TARGET_COLUMN
    ].copy()

    if features.empty:
        raise ValueError(
            "No model features were found."
        )

    if features.isna().any().any():
        raise ValueError(
            "Missing values were found in model features."
        )

    if target.isna().any():
        raise ValueError(
            "Missing values were found in the target."
        )

    non_numeric_columns = (
        features
        .select_dtypes(exclude=["number"])
        .columns
        .tolist()
    )

    if non_numeric_columns:
        raise TypeError(
            "All model features must be numeric. "
            f"Invalid columns: {non_numeric_columns}"
        )

    return features, target


def create_train_validation_split(
    features: pd.DataFrame,
    target: pd.Series
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series
]:
    """Create reproducible training and validation subsets."""

    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )


def train_ridge_regression(
    training_features: pd.DataFrame,
    training_target: pd.Series
) -> tuple[Ridge, float]:
    """Train Ridge Regression using L2 regularization."""

    model = Ridge(
        alpha=RIDGE_ALPHA
    )

    start_time = perf_counter()

    model.fit(
        training_features,
        training_target
    )

    duration = perf_counter() - start_time

    return model, duration


def generate_predictions(
    model: Ridge,
    validation_features: pd.DataFrame
) -> tuple[np.ndarray, float]:
    """Generate validation predictions."""

    start_time = perf_counter()

    predictions = model.predict(
        validation_features
    )

    duration = perf_counter() - start_time

    if not np.isfinite(predictions).all():
        raise ValueError(
            "The Ridge model generated invalid predictions."
        )

    return predictions, duration


def calculate_metrics(
    actual_log_values: pd.Series,
    predicted_log_values: np.ndarray
) -> dict[str, float]:
    """Calculate log-scale and original-scale metrics."""

    actual_log = actual_log_values.to_numpy(
        dtype=float
    )

    predicted_log = np.asarray(
        predicted_log_values,
        dtype=float
    )

    log_mae = mean_absolute_error(
        actual_log,
        predicted_log
    )

    log_mse = mean_squared_error(
        actual_log,
        predicted_log
    )

    log_rmse = np.sqrt(log_mse)

    log_r2 = r2_score(
        actual_log,
        predicted_log
    )

    actual_prices = np.expm1(
        actual_log
    )

    predicted_prices = np.maximum(
        np.expm1(predicted_log),
        0
    )

    price_mae = mean_absolute_error(
        actual_prices,
        predicted_prices
    )

    price_mse = mean_squared_error(
        actual_prices,
        predicted_prices
    )

    price_rmse = np.sqrt(
        price_mse
    )

    price_r2 = r2_score(
        actual_prices,
        predicted_prices
    )

    return {
        "log_mae": float(log_mae),
        "log_mse": float(log_mse),
        "log_rmse": float(log_rmse),
        "log_r2": float(log_r2),
        "price_mae": float(price_mae),
        "price_mse": float(price_mse),
        "price_rmse": float(price_rmse),
        "price_r2": float(price_r2)
    }


def create_validation_results(
    actual_log_values: pd.Series,
    predicted_log_values: np.ndarray
) -> pd.DataFrame:
    """Create row-level Ridge validation results."""

    actual_log = actual_log_values.to_numpy(
        dtype=float
    )

    predicted_log = np.asarray(
        predicted_log_values,
        dtype=float
    )

    actual_prices = np.expm1(
        actual_log
    )

    predicted_prices = np.maximum(
        np.expm1(predicted_log),
        0
    )

    log_residuals = (
        actual_log - predicted_log
    )

    price_differences = (
        predicted_prices - actual_prices
    )

    absolute_errors = np.abs(
        price_differences
    )

    percentage_errors = np.divide(
        absolute_errors,
        actual_prices,
        out=np.zeros_like(
            absolute_errors,
            dtype=float
        ),
        where=actual_prices != 0
    ) * 100

    return pd.DataFrame(
        {
            "Actual_Log_Price": actual_log,
            "Predicted_Log_Price": predicted_log,
            "Log_Residual": log_residuals,
            "Actual_Price": actual_prices,
            "Predicted_Price": predicted_prices,
            "Price_Difference": price_differences,
            "Absolute_Price_Error": absolute_errors,
            "Absolute_Percentage_Error": percentage_errors
        }
    )


def create_coefficient_table(
    model: Ridge,
    feature_names: list[str]
) -> pd.DataFrame:
    """Create a Ridge coefficient table."""

    table = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": model.coef_
        }
    )

    table["AbsoluteCoefficient"] = (
        table["Coefficient"].abs()
    )

    return (
        table
        .sort_values(
            by="AbsoluteCoefficient",
            ascending=False
        )
        .reset_index(drop=True)
    )


def plot_actual_vs_predicted(
    validation_results: pd.DataFrame
) -> None:
    """Save actual vs predicted Ridge plot."""

    actual = validation_results[
        "Actual_Price"
    ]

    predicted = validation_results[
        "Predicted_Price"
    ]

    minimum = min(
        actual.min(),
        predicted.min()
    )

    maximum = max(
        actual.max(),
        predicted.max()
    )

    plt.figure(figsize=(9, 7))

    plt.scatter(
        actual,
        predicted,
        alpha=0.65,
        edgecolors="none"
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=2
    )

    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title("Ridge Regression: Actual vs Predicted")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        RIDGE_ACTUAL_VS_PREDICTED_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_residuals(
    validation_results: pd.DataFrame
) -> None:
    """Save Ridge residual plot."""

    predicted = validation_results[
        "Predicted_Log_Price"
    ]

    residuals = validation_results[
        "Log_Residual"
    ]

    plt.figure(figsize=(9, 7))

    plt.scatter(
        predicted,
        residuals,
        alpha=0.65,
        edgecolors="none"
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=2
    )

    plt.xlabel("Predicted Log Sale Price")
    plt.ylabel("Log Residual")
    plt.title("Ridge Regression Residual Plot")
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        RIDGE_RESIDUAL_PLOT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_coefficients(
    coefficient_table: pd.DataFrame,
    top_n: int = 20
) -> None:
    """Save largest Ridge coefficient plot."""

    selected = (
        coefficient_table
        .head(top_n)
        .sort_values(
            by="Coefficient",
            ascending=True
        )
    )

    plt.figure(figsize=(11, 9))

    plt.barh(
        selected["Feature"],
        selected["Coefficient"]
    )

    plt.axvline(
        x=0,
        linewidth=1
    )

    plt.xlabel("Ridge Coefficient")
    plt.ylabel("Feature")
    plt.title(
        f"Top {top_n} Ridge Coefficients"
    )

    plt.grid(
        axis="x",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        RIDGE_COEFFICIENTS_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def save_model(
    model: Ridge,
    feature_names: list[str]
) -> None:
    """Save Ridge model and metadata."""

    artifact = {
        "model": model,
        "feature_names": feature_names,
        "model_name": "RidgeRegression",
        "alpha": RIDGE_ALPHA,
        "target_column": TARGET_COLUMN,
        "target_transformation": "log1p",
        "inverse_transformation": "expm1",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE
    }

    save_joblib_artifact(
        artifact,
        RIDGE_MODEL_PATH
    )


def save_metrics_report(
    metrics: dict[str, float],
    training_duration: float,
    prediction_duration: float,
    dataset_shape: tuple[int, int],
    training_shape: tuple[int, int],
    validation_shape: tuple[int, int]
) -> None:
    """Save Ridge evaluation metrics."""

    report = f"""
RIDGE REGRESSION EVALUATION REPORT
==================================

MODEL INFORMATION
-----------------
Model            : Ridge Regression
Alpha            : {RIDGE_ALPHA}
Target           : {TARGET_COLUMN}
Validation ratio : {TEST_SIZE:.0%}
Random state     : {RANDOM_STATE}

DATASET INFORMATION
-------------------
Complete feature shape : {dataset_shape}
Training subset shape  : {training_shape}
Validation subset shape: {validation_shape}
Feature count          : {dataset_shape[1]}

PERFORMANCE - LOG SCALE
-----------------------
MAE  : {metrics["log_mae"]:.6f}
MSE  : {metrics["log_mse"]:.6f}
RMSE : {metrics["log_rmse"]:.6f}
R2   : {metrics["log_r2"]:.6f}

PERFORMANCE - ORIGINAL PRICE SCALE
----------------------------------
MAE  : {metrics["price_mae"]:.2f}
MSE  : {metrics["price_mse"]:.2f}
RMSE : {metrics["price_rmse"]:.2f}
R2   : {metrics["price_r2"]:.6f}

EXECUTION TIME
--------------
Training time   : {training_duration:.6f} seconds
Prediction time : {prediction_duration:.6f} seconds
""".strip()

    save_text(
        report,
        RIDGE_METRICS_PATH
    )

    print("\n" + report)


def run_ridge_regression_training() -> None:
    """Execute the complete Ridge Regression pipeline."""

    logger.info(
        "Ridge Regression training pipeline started."
    )

    create_output_directories()

    dataframe = load_processed_training_data()

    features, target = separate_features_and_target(
        dataframe
    )

    (
        training_features,
        validation_features,
        training_target,
        validation_target
    ) = create_train_validation_split(
        features,
        target
    )

    logger.info(
        "Dataset split completed: train=%s, validation=%s",
        training_features.shape,
        validation_features.shape
    )

    model, training_duration = train_ridge_regression(
        training_features,
        training_target
    )

    predictions, prediction_duration = (
        generate_predictions(
            model,
            validation_features
        )
    )

    metrics = calculate_metrics(
        validation_target,
        predictions
    )

    validation_results = create_validation_results(
        validation_target,
        predictions
    )

    coefficient_table = create_coefficient_table(
        model,
        features.columns.tolist()
    )

    plot_actual_vs_predicted(
        validation_results
    )

    plot_residuals(
        validation_results
    )

    plot_coefficients(
        coefficient_table
    )

    save_model(
        model,
        features.columns.tolist()
    )

    save_dataframe(
        validation_results,
        RIDGE_VALIDATION_PREDICTIONS_PATH
    )

    save_metrics_report(
        metrics=metrics,
        training_duration=training_duration,
        prediction_duration=prediction_duration,
        dataset_shape=features.shape,
        training_shape=training_features.shape,
        validation_shape=validation_features.shape
    )

    logger.info(
        "Ridge Regression pipeline completed successfully."
    )


if __name__ == "__main__":
    run_ridge_regression_training()