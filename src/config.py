"""
Central configuration for the House Price Prediction project.

All project paths and shared experiment settings are defined here.
"""

from pathlib import Path


# ============================================================
# PROJECT DIRECTORIES
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
IMAGES_DIR = PROJECT_ROOT / "images"


# ============================================================
# DATA FILES
# ============================================================

TRAIN_DATA_PATH = RAW_DATA_DIR / "train.csv"
TEST_DATA_PATH = RAW_DATA_DIR / "test.csv"

PROCESSED_TRAIN_PATH = (
    PROCESSED_DATA_DIR / "processed_train.csv"
)

PROCESSED_TEST_PATH = (
    PROCESSED_DATA_DIR / "processed_test.csv"
)


# ============================================================
# PREPROCESSING ARTIFACTS
# ============================================================

PREPROCESSOR_PATH = (
    MODELS_DIR / "preprocessor.joblib"
)

PREPROCESSING_SUMMARY_PATH = (
    RESULTS_DIR / "preprocessing_summary.txt"
)


# ============================================================
# LINEAR REGRESSION ARTIFACTS
# ============================================================

LINEAR_MODEL_PATH = (
    MODELS_DIR / "linear_regression_model.joblib"
)

LINEAR_METRICS_PATH = (
    RESULTS_DIR / "linear_regression_metrics.txt"
)

VALIDATION_PREDICTIONS_PATH = (
    RESULTS_DIR / "validation_predictions.csv"
)

# ============================================================
# RIDGE REGRESSION ARTIFACTS
# ============================================================

RIDGE_MODEL_PATH = (
    MODELS_DIR / "ridge_regression_model.joblib"
)

RIDGE_METRICS_PATH = (
    RESULTS_DIR / "ridge_regression_metrics.txt"
)

RIDGE_VALIDATION_PREDICTIONS_PATH = (
    RESULTS_DIR / "ridge_validation_predictions.csv"
)

RIDGE_ACTUAL_VS_PREDICTED_PATH = (
    IMAGES_DIR / "ridge_actual_vs_predicted.png"
)

RIDGE_RESIDUAL_PLOT_PATH = (
    IMAGES_DIR / "ridge_residual_plot.png"
)

RIDGE_COEFFICIENTS_PATH = (
    IMAGES_DIR / "ridge_feature_coefficients.png"
)


# ============================================================
# VISUALIZATION FILES
# ============================================================

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
# SHARED MODEL SETTINGS
# ============================================================

TARGET_COLUMN = "SalePrice_log"

TEST_SIZE = 0.20
RANDOM_STATE = 42

REMOVE_OUTLIERS = True
SCALE_NUMERIC_FEATURES = True

RIDGE_ALPHA = 10.0


# ============================================================
# DIRECTORY COLLECTION
# ============================================================

OUTPUT_DIRECTORIES = (
    PROCESSED_DATA_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    IMAGES_DIR
)