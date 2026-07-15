"""
House Price Prediction - Linear Regression Pipeline

This module performs:

1. Processed training data loading
2. Feature-target separation
3. Reproducible train-validation split
4. Linear Regression model training
5. Log-scale and original-scale evaluation
6. Validation prediction export
7. Model artifact export
8. Evaluation report export
9. Model visualization generation
"""

from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split

from src.config import (
    ACTUAL_VS_PREDICTED_PATH,
    FEATURE_COEFFICIENTS_PATH,
    LINEAR_METRICS_PATH,
    LINEAR_MODEL_PATH,
    PROCESSED_TRAIN_PATH,
    RANDOM_STATE,
    RESIDUAL_DISTRIBUTION_PATH,
    RESIDUAL_PLOT_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    VALIDATION_PREDICTIONS_PATH
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

def load_processed_training_data() -> pd.DataFrame:
    """
    Load and validate the processed training dataset.
    """

    logger.info(
        "Loading processed training dataset."
    )

    dataframe = load_csv(
        PROCESSED_TRAIN_PATH,
        "Processed training dataset"
    )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found "
            "in the processed training dataset."
        )

    logger.info(
        "Processed training dataset loaded: %s",
        dataframe.shape
    )

    return dataframe


# ============================================================
# FEATURE-TARGET SEPARATION
# ============================================================

def separate_features_and_target(
    dataframe: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate model features and the logarithmic target.
    """

    features = (
        dataframe
        .drop(columns=[TARGET_COLUMN])
        .copy()
    )

    target = (
        dataframe[TARGET_COLUMN]
        .copy()
    )

    if features.empty:
        raise ValueError(
            "The processed dataset contains no model features."
        )

    feature_missing_count = int(
        features.isna().sum().sum()
    )

    target_missing_count = int(
        target.isna().sum()
    )

    if feature_missing_count != 0:
        raise ValueError(
            "Missing values were found in model features: "
            f"{feature_missing_count}"
        )

    if target_missing_count != 0:
        raise ValueError(
            "Missing values were found in the target: "
            f"{target_missing_count}"
        )

    non_numeric_columns = (
        features
        .select_dtypes(exclude=["number"])
        .columns
        .tolist()
    )

    if non_numeric_columns:
        raise TypeError(
            "All processed features must be numeric. "
            "Non-numeric columns found: "
            f"{non_numeric_columns}"
        )

    if not np.isfinite(
        features.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite values were found in model features."
        )

    if not np.isfinite(
        target.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Non-finite values were found in the target."
        )

    return features, target


# ============================================================
# TRAIN-VALIDATION SPLIT
# ============================================================

def create_train_validation_split(
    features: pd.DataFrame,
    target: pd.Series
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series
]:
    """
    Create reproducible training and validation subsets.
    """

    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )


# ============================================================
# MODEL TRAINING
# ============================================================

def train_linear_regression(
    training_features: pd.DataFrame,
    training_target: pd.Series
) -> tuple[LinearRegression, float]:
    """
    Train a Linear Regression model.

    Returns
    -------
    model:
        Fitted Linear Regression estimator.

    training_duration:
        Model fitting time in seconds.
    """

    model = LinearRegression()

    start_time = perf_counter()

    model.fit(
        training_features,
        training_target
    )

    training_duration = (
        perf_counter() - start_time
    )

    return model, training_duration


# ============================================================
# PREDICTION
# ============================================================

def generate_predictions(
    model: LinearRegression,
    validation_features: pd.DataFrame
) -> tuple[np.ndarray, float]:
    """
    Generate validation predictions on the logarithmic scale.
    """

    start_time = perf_counter()

    predictions = model.predict(
        validation_features
    )

    prediction_duration = (
        perf_counter() - start_time
    )

    if not np.isfinite(predictions).all():
        raise ValueError(
            "The model generated non-finite predictions."
        )

    return predictions, prediction_duration


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    actual_log_values: pd.Series,
    predicted_log_values: np.ndarray
) -> dict[str, float]:
    """
    Calculate metrics on logarithmic and original price scales.
    """

    actual_log_array = (
        actual_log_values
        .to_numpy(dtype=float)
    )

    predicted_log_array = np.asarray(
        predicted_log_values,
        dtype=float
    )

    # --------------------------------------------------------
    # Log-scale metrics
    # --------------------------------------------------------

    log_mae = mean_absolute_error(
        actual_log_array,
        predicted_log_array
    )

    log_mse = mean_squared_error(
        actual_log_array,
        predicted_log_array
    )

    log_rmse = float(
        np.sqrt(log_mse)
    )

    log_r2 = r2_score(
        actual_log_array,
        predicted_log_array
    )

    # --------------------------------------------------------
    # Original price scale
    # --------------------------------------------------------

    actual_prices = np.expm1(
        actual_log_array
    )

    predicted_prices = np.expm1(
        predicted_log_array
    )

    predicted_prices = np.maximum(
        predicted_prices,
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

    price_rmse = float(
        np.sqrt(price_mse)
    )

    price_r2 = r2_score(
        actual_prices,
        predicted_prices
    )

    return {
        "log_mae": float(log_mae),
        "log_mse": float(log_mse),
        "log_rmse": log_rmse,
        "log_r2": float(log_r2),
        "price_mae": float(price_mae),
        "price_mse": float(price_mse),
        "price_rmse": price_rmse,
        "price_r2": float(price_r2)
    }


# ============================================================
# VALIDATION RESULTS
# ============================================================

def create_validation_results(
    actual_log_values: pd.Series,
    predicted_log_values: np.ndarray
) -> pd.DataFrame:
    """
    Create row-level validation prediction results.
    """

    actual_log_array = (
        actual_log_values
        .to_numpy(dtype=float)
    )

    predicted_log_array = np.asarray(
        predicted_log_values,
        dtype=float
    )

    actual_prices = np.expm1(
        actual_log_array
    )

    predicted_prices = np.expm1(
        predicted_log_array
    )

    predicted_prices = np.maximum(
        predicted_prices,
        0
    )

    log_residuals = (
        actual_log_array
        - predicted_log_array
    )

    price_differences = (
        predicted_prices
        - actual_prices
    )

    absolute_price_errors = np.abs(
        price_differences
    )

    percentage_errors = np.divide(
        absolute_price_errors,
        actual_prices,
        out=np.zeros_like(
            absolute_price_errors,
            dtype=float
        ),
        where=actual_prices != 0
    ) * 100

    return pd.DataFrame(
        {
            "Actual_Log_Price": actual_log_array,
            "Predicted_Log_Price": predicted_log_array,
            "Log_Residual": log_residuals,
            "Actual_Price": actual_prices,
            "Predicted_Price": predicted_prices,
            "Price_Difference": price_differences,
            "Absolute_Price_Error": absolute_price_errors,
            "Absolute_Percentage_Error": percentage_errors
        }
    )


# ============================================================
# COEFFICIENT ANALYSIS
# ============================================================

def create_coefficient_table(
    model: LinearRegression,
    feature_names: list[str]
) -> pd.DataFrame:
    """
    Create a ranked table of Linear Regression coefficients.
    """

    if len(model.coef_) != len(feature_names):
        raise ValueError(
            "Model coefficient count does not match "
            "the feature count."
        )

    coefficient_table = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": model.coef_
        }
    )

    coefficient_table["AbsoluteCoefficient"] = (
        coefficient_table["Coefficient"].abs()
    )

    return (
        coefficient_table
        .sort_values(
            by="AbsoluteCoefficient",
            ascending=False
        )
        .reset_index(drop=True)
    )


# ============================================================
# VISUALIZATIONS
# ============================================================

def plot_actual_vs_predicted(
    validation_results: pd.DataFrame
) -> None:
    """
    Plot actual house prices against predicted house prices.
    """

    actual_prices = validation_results[
        "Actual_Price"
    ]

    predicted_prices = validation_results[
        "Predicted_Price"
    ]

    minimum_value = min(
        float(actual_prices.min()),
        float(predicted_prices.min())
    )

    maximum_value = max(
        float(actual_prices.max()),
        float(predicted_prices.max())
    )

    plt.figure(figsize=(9, 7))

    plt.scatter(
        actual_prices,
        predicted_prices,
        alpha=0.65,
        edgecolors="none"
    )

    plt.plot(
        [minimum_value, maximum_value],
        [minimum_value, maximum_value],
        linestyle="--",
        linewidth=2,
        label="Perfect prediction"
    )

    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title("Actual vs Predicted House Prices")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        ACTUAL_VS_PREDICTED_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_residuals(
    validation_results: pd.DataFrame
) -> None:
    """
    Plot predicted log prices against log residuals.
    """

    predicted_log_prices = validation_results[
        "Predicted_Log_Price"
    ]

    residuals = validation_results[
        "Log_Residual"
    ]

    plt.figure(figsize=(9, 7))

    plt.scatter(
        predicted_log_prices,
        residuals,
        alpha=0.65,
        edgecolors="none"
    )

    plt.axhline(
        y=0,
        linestyle="--",
        linewidth=2,
        label="Zero residual"
    )

    plt.xlabel("Predicted Log Sale Price")
    plt.ylabel("Log Residual")
    plt.title("Linear Regression Residual Plot")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        RESIDUAL_PLOT_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_residual_distribution(
    validation_results: pd.DataFrame
) -> None:
    """
    Plot the distribution of logarithmic residuals.
    """

    residuals = validation_results[
        "Log_Residual"
    ]

    plt.figure(figsize=(9, 7))

    plt.hist(
        residuals,
        bins=30,
        alpha=0.80,
        edgecolor="black"
    )

    plt.axvline(
        x=0,
        linestyle="--",
        linewidth=2,
        label="Zero residual"
    )

    plt.xlabel("Log Residual")
    plt.ylabel("Frequency")
    plt.title("Distribution of Linear Regression Residuals")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        RESIDUAL_DISTRIBUTION_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def plot_feature_coefficients(
    coefficient_table: pd.DataFrame,
    top_n: int = 20
) -> None:
    """
    Plot features with the largest absolute coefficients.
    """

    if top_n <= 0:
        raise ValueError(
            "top_n must be greater than zero."
        )

    selected_coefficients = (
        coefficient_table
        .head(top_n)
        .sort_values(
            by="Coefficient",
            ascending=True
        )
    )

    plt.figure(figsize=(11, 9))

    plt.barh(
        selected_coefficients["Feature"],
        selected_coefficients["Coefficient"]
    )

    plt.axvline(
        x=0,
        linewidth=1
    )

    plt.xlabel("Linear Regression Coefficient")
    plt.ylabel("Feature")
    plt.title(
        f"Top {top_n} Features by Absolute Coefficient"
    )

    plt.grid(
        axis="x",
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        FEATURE_COEFFICIENTS_PATH,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


def generate_evaluation_visualizations(
    validation_results: pd.DataFrame,
    coefficient_table: pd.DataFrame
) -> None:
    """
    Generate and save all evaluation plots.
    """

    plot_actual_vs_predicted(
        validation_results
    )

    plot_residuals(
        validation_results
    )

    plot_residual_distribution(
        validation_results
    )

    plot_feature_coefficients(
        coefficient_table
    )

    logger.info(
        "Evaluation visualizations generated successfully."
    )


# ============================================================
# OUTPUT SAVING
# ============================================================

def save_model(
    model: LinearRegression,
    feature_names: list[str]
) -> None:
    """
    Save the trained model together with metadata.
    """

    model_artifact = {
        "model": model,
        "feature_names": feature_names,
        "target_column": TARGET_COLUMN,
        "target_transformation": "log1p",
        "inverse_transformation": "expm1",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE
    }

    save_joblib_artifact(
        model_artifact,
        LINEAR_MODEL_PATH
    )

    logger.info(
        "Linear Regression model saved: %s",
        LINEAR_MODEL_PATH
    )


def save_validation_predictions(
    validation_results: pd.DataFrame
) -> None:
    """
    Save row-level validation predictions.
    """

    save_dataframe(
        validation_results,
        VALIDATION_PREDICTIONS_PATH
    )

    logger.info(
        "Validation predictions saved: %s",
        VALIDATION_PREDICTIONS_PATH
    )


def save_metrics_report(
    metrics: dict[str, float],
    model: LinearRegression,
    dataset_shape: tuple[int, int],
    training_shape: tuple[int, int],
    validation_shape: tuple[int, int],
    training_duration: float,
    prediction_duration: float,
    coefficient_table: pd.DataFrame
) -> None:
    """
    Save a human-readable model evaluation report.
    """

    top_positive_coefficients = (
        coefficient_table
        .sort_values(
            by="Coefficient",
            ascending=False
        )
        .head(10)
    )

    top_negative_coefficients = (
        coefficient_table
        .sort_values(
            by="Coefficient",
            ascending=True
        )
        .head(10)
    )

    positive_text = "\n".join(
        f"{row.Feature}: {row.Coefficient:.6f}"
        for row in top_positive_coefficients.itertuples()
    )

    negative_text = "\n".join(
        f"{row.Feature}: {row.Coefficient:.6f}"
        for row in top_negative_coefficients.itertuples()
    )

    report = f"""
LINEAR REGRESSION EVALUATION REPORT
===================================

MODEL INFORMATION
-----------------
Model                  : LinearRegression
Target                 : {TARGET_COLUMN}
Target transformation  : log1p
Inverse transformation : expm1
Random state           : {RANDOM_STATE}
Validation ratio       : {TEST_SIZE:.0%}

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

MODEL PARAMETERS
----------------
Intercept         : {model.intercept_:.6f}
Coefficient count : {len(model.coef_)}

TOP POSITIVE COEFFICIENTS
-------------------------
{positive_text}

TOP NEGATIVE COEFFICIENTS
-------------------------
{negative_text}

GENERATED FILES
---------------
Model:
{LINEAR_MODEL_PATH}

Validation predictions:
{VALIDATION_PREDICTIONS_PATH}
""".strip()

    save_text(
        report,
        LINEAR_METRICS_PATH
    )

    print("\n" + report)

    logger.info(
        "Metrics report saved: %s",
        LINEAR_METRICS_PATH
    )


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def run_linear_regression_training() -> None:
    """
    Execute the complete Linear Regression pipeline.
    """

    logger.info(
        "Linear Regression training pipeline started."
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

    model, training_duration = train_linear_regression(
        training_features,
        training_target
    )

    logger.info(
        "Linear Regression trained in %.6f seconds.",
        training_duration
    )

    (
        validation_predictions,
        prediction_duration
    ) = generate_predictions(
        model,
        validation_features
    )

    metrics = calculate_metrics(
        validation_target,
        validation_predictions
    )

    validation_results = create_validation_results(
        validation_target,
        validation_predictions
    )

    coefficient_table = create_coefficient_table(
        model,
        features.columns.tolist()
    )

    generate_evaluation_visualizations(
        validation_results,
        coefficient_table
    )

    save_model(
        model,
        features.columns.tolist()
    )

    save_validation_predictions(
        validation_results
    )

    save_metrics_report(
        metrics=metrics,
        model=model,
        dataset_shape=features.shape,
        training_shape=training_features.shape,
        validation_shape=validation_features.shape,
        training_duration=training_duration,
        prediction_duration=prediction_duration,
        coefficient_table=coefficient_table
    )

    logger.info(
        "Linear Regression pipeline completed successfully."
    )


if __name__ == "__main__":
    run_linear_regression_training()