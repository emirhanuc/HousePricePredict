"""
House Price Prediction - Linear Regression Training

This module:

1. Loads the preprocessed training dataset.
2. Separates model features and the logarithmic target.
3. Creates reproducible training and validation subsets.
4. Trains a Linear Regression baseline model.
5. Calculates regression evaluation metrics.
6. Converts predictions back to the original price scale.
7. Saves validation predictions.
8. Saves evaluation metrics.
9. Saves the trained model.
"""

from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGES_DIR = PROJECT_ROOT / "images"

PROCESSED_TRAIN_PATH = (
    PROCESSED_DATA_DIR / "processed_train.csv"
)

MODEL_PATH = (
    MODELS_DIR / "linear_regression_model.joblib"
)

METRICS_PATH = (
    RESULTS_DIR / "linear_regression_metrics.txt"
)

VALIDATION_PREDICTIONS_PATH = (
    RESULTS_DIR / "validation_predictions.csv"
)

ACTUAL_VS_PREDICTED_PATH = (
    IMAGES_DIR / "actual_vs_predicted.png"
)

RESIDUAL_PLOT_PATH = (
    IMAGES_DIR / "residual_plot.png"
)

RESIDUAL_DISTRIBUTION_PATH = (
    IMAGES_DIR / "residual_distribution.png"
)

FEATURE_COEFFICIENTS_PATH = (
    IMAGES_DIR / "feature_coefficients.png"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

TARGET_COLUMN = "SalePrice_log"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

def create_output_directories() -> None:
    """Create model, result, and image directories."""

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# DATA LOADING
# ============================================================

def load_processed_training_data() -> pd.DataFrame:
    """
    Load the processed training dataset.

    Returns
    -------
    pd.DataFrame
        Preprocessed training data containing model features and
        the SalePrice_log target.
    """

    if not PROCESSED_TRAIN_PATH.exists():
        raise FileNotFoundError(
            "Processed training data was not found.\n"
            f"Expected path: {PROCESSED_TRAIN_PATH}\n"
            "Run the preprocessing script first:\n"
            "python src/preprocessing.py"
        )

    dataframe = pd.read_csv(
        PROCESSED_TRAIN_PATH
    )

    if dataframe.empty:
        raise ValueError(
            "The processed training dataset is empty."
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' was not found "
            "in the processed training dataset."
        )

    return dataframe


# ============================================================
# FEATURE-TARGET SEPARATION
# ============================================================

def separate_features_and_target(
    dataframe: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate model inputs from the logarithmic target variable.

    Parameters
    ----------
    dataframe:
        Processed training dataset.

    Returns
    -------
    features:
        Model input variables.

    target:
        Logarithmically transformed SalePrice.
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

    if features.shape[1] == 0:
        raise ValueError(
            "No model features were found."
        )

    if features.isna().any().any():
        missing_count = int(
            features.isna().sum().sum()
        )

        raise ValueError(
            "Missing values were found in model features: "
            f"{missing_count}"
        )

    if target.isna().any():
        missing_count = int(
            target.isna().sum()
        )

        raise ValueError(
            "Missing values were found in the target: "
            f"{missing_count}"
        )

    non_numeric_columns = (
        features
        .select_dtypes(exclude=["number"])
        .columns
        .tolist()
    )

    if non_numeric_columns:
        raise TypeError(
            "All processed model features must be numeric. "
            "Non-numeric columns found: "
            f"{non_numeric_columns}"
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
    Create reproducible training and validation datasets.

    The validation subset is not used during model fitting.
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
    Train the baseline Linear Regression model.

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
    Generate logarithmic validation predictions.

    Returns
    -------
    predictions:
        Predicted SalePrice_log values.

    prediction_duration:
        Total prediction time in seconds.
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
# METRIC CALCULATION
# ============================================================

def calculate_metrics(
    actual_log_values: pd.Series,
    predicted_log_values: np.ndarray
) -> dict[str, float]:
    """
    Calculate metrics on both logarithmic and original scales.

    Log-scale RMSE corresponds closely to the metric commonly used
    for the Kaggle House Prices problem.
    """

    # --------------------------------------------------------
    # Log-scale metrics
    # --------------------------------------------------------

    log_mae = mean_absolute_error(
        actual_log_values,
        predicted_log_values
    )

    log_mse = mean_squared_error(
        actual_log_values,
        predicted_log_values
    )

    log_rmse = np.sqrt(log_mse)

    log_r2 = r2_score(
        actual_log_values,
        predicted_log_values
    )

    # --------------------------------------------------------
    # Convert predictions back to original price scale
    # --------------------------------------------------------

    actual_prices = np.expm1(
        actual_log_values.to_numpy()
    )

    predicted_prices = np.expm1(
        predicted_log_values
    )

    # Negative house-price predictions are not meaningful.
    # This is a safety measure after inverse transformation.
    predicted_prices = np.maximum(
        predicted_prices,
        0
    )

    # --------------------------------------------------------
    # Original-scale metrics
    # --------------------------------------------------------

    price_mae = mean_absolute_error(
        actual_prices,
        predicted_prices
    )

    price_mse = mean_squared_error(
        actual_prices,
        predicted_prices
    )

    price_rmse = np.sqrt(price_mse)

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


# ============================================================
# MODEL COEFFICIENT ANALYSIS
# ============================================================

def create_coefficient_table(
    model: LinearRegression,
    feature_names: list[str]
) -> pd.DataFrame:
    """
    Create a table containing model coefficients.

    Absolute coefficients are included only for ranking purposes.
    Coefficient magnitude must be interpreted carefully when
    correlated features are present.
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

    coefficient_table = (
        coefficient_table
        .sort_values(
            by="AbsoluteCoefficient",
            ascending=False
        )
        .reset_index(drop=True)
    )

    return coefficient_table


# ============================================================
# VALIDATION OUTPUT
# ============================================================

def create_validation_results(
    actual_log_values: pd.Series,
    predicted_log_values: np.ndarray
) -> pd.DataFrame:
    """
    Create row-level validation prediction results.
    """

    actual_log_array = (
        actual_log_values.to_numpy()
    )

    actual_prices = np.expm1(
        actual_log_array
    )

    predicted_prices = np.expm1(
        predicted_log_values
    )

    predicted_prices = np.maximum(
        predicted_prices,
        0
    )

    residual_log = (
        actual_log_array
        - predicted_log_values
    )

    price_difference = (
        predicted_prices
        - actual_prices
    )

    absolute_price_error = np.abs(
        price_difference
    )

    percentage_error = np.divide(
        absolute_price_error,
        actual_prices,
        out=np.zeros_like(
            absolute_price_error,
            dtype=float
        ),
        where=actual_prices != 0
    ) * 100

    validation_results = pd.DataFrame(
        {
            "Actual_Log_Price": actual_log_array,
            "Predicted_Log_Price": predicted_log_values,
            "Log_Residual": residual_log,
            "Actual_Price": actual_prices,
            "Predicted_Price": predicted_prices,
            "Price_Difference": price_difference,
            "Absolute_Price_Error": absolute_price_error,
            "Absolute_Percentage_Error": percentage_error
        }
    )

    return validation_results
def plot_actual_vs_predicted(
    validation_results: pd.DataFrame
) -> None:
    """
    Plot actual house prices against predicted house prices.

    Predictions closer to the diagonal line represent more
    accurate estimates.
    """

    actual_prices = validation_results["Actual_Price"]
    predicted_prices = validation_results["Predicted_Price"]

    minimum_value = min(
        actual_prices.min(),
        predicted_prices.min()
    )

    maximum_value = max(
        actual_prices.max(),
        predicted_prices.max()
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
        label="Perfect Prediction"
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
    Plot predicted logarithmic prices against residual errors.

    A desirable residual plot has points randomly distributed
    around zero without a visible systematic pattern.
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
        label="Zero Residual"
    )

    plt.xlabel("Predicted Log Sale Price")
    plt.ylabel("Residual")
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
    Plot the distribution of logarithmic residual errors.
    """

    residuals = validation_results[
        "Log_Residual"
    ]

    plt.figure(figsize=(9, 7))

    plt.hist(
        residuals,
        bins=30,
        alpha=0.8,
        edgecolor="black"
    )

    plt.axvline(
        x=0,
        linestyle="--",
        linewidth=2,
        label="Zero Residual"
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
    Plot the model features having the largest absolute
    Linear Regression coefficients.

    Coefficient size should not automatically be interpreted as
    causal importance, especially when correlated features exist.
    """

    top_coefficients = (
        coefficient_table
        .head(top_n)
        .sort_values(
            by="Coefficient",
            ascending=True
        )
    )

    plt.figure(figsize=(11, 9))

    plt.barh(
        top_coefficients["Feature"],
        top_coefficients["Coefficient"]
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
    Generate and save all Linear Regression evaluation plots.
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

    print("\nEvaluation visualizations saved:")

    print(
        f"- {ACTUAL_VS_PREDICTED_PATH}"
    )

    print(
        f"- {RESIDUAL_PLOT_PATH}"
    )

    print(
        f"- {RESIDUAL_DISTRIBUTION_PATH}"
    )

    print(
        f"- {FEATURE_COEFFICIENTS_PATH}"
    )



# ============================================================
# OUTPUT SAVING
# ============================================================

def save_model(
    model: LinearRegression,
    feature_names: list[str]
) -> None:
    """
    Save the trained estimator together with required metadata.
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

    joblib.dump(
        model_artifact,
        MODEL_PATH
    )


def save_validation_predictions(
    validation_results: pd.DataFrame
) -> None:
    """Save row-level validation predictions."""

    validation_results.to_csv(
        VALIDATION_PREDICTIONS_PATH,
        index=False
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
Target                  : {TARGET_COLUMN}
Target transformation   : log1p
Inverse transformation  : expm1
Random state            : {RANDOM_STATE}
Validation ratio         : {TEST_SIZE:.0%}

DATASET INFORMATION
-------------------
Complete dataset shape  : {dataset_shape}
Training subset shape   : {training_shape}
Validation subset shape : {validation_shape}
Feature count           : {dataset_shape[1]}

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
Saved model:
{MODEL_PATH}

Validation predictions:
{VALIDATION_PREDICTIONS_PATH}
""".strip()

    METRICS_PATH.write_text(
        report,
        encoding="utf-8"
    )

    print("\n" + report)


# ============================================================
# COMPLETE TRAINING PIPELINE
# ============================================================

def run_linear_regression_training() -> None:
    """
    Execute the complete Linear Regression training pipeline.
    """

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

    model, training_duration = train_linear_regression(
        training_features,
        training_target
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

    print(
        "\nLinear Regression training completed successfully."
    )


if __name__ == "__main__":
    run_linear_regression_training()