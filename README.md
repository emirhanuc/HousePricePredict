# House Price Prediction

A machine learning project that predicts residential property prices using the Ames Housing dataset.

The project includes a complete workflow covering data preprocessing, missing-value handling, feature engineering, categorical encoding, feature scaling, Linear Regression training, model evaluation, visualization, and prediction export.

---

## Project Overview

House price prediction is a supervised regression problem.

The objective of this project is to train a machine learning model that estimates the sale price of a house based on characteristics such as:

- Overall construction quality
- Living area
- Neighborhood
- Number of rooms and bathrooms
- Garage capacity
- Basement area
- Construction and renovation year
- Exterior and interior condition
- Lot characteristics

The target variable is:

```text
SalePrice
```

Since `SalePrice` has a right-skewed distribution, the target is transformed using:

```python
SalePrice_log = np.log1p(SalePrice)
```

Predictions are converted back to the original price scale using:

```python
predicted_price = np.expm1(prediction_log)
```

---

## Dataset

The project uses the Ames Housing dataset commonly associated with the Kaggle House Prices competition.

Raw datasets are stored in:

```text
data/raw/train.csv
data/raw/test.csv
```

The training dataset initially contains:

```text
1460 rows
81 columns
```

The test dataset initially contains:

```text
1459 rows
80 columns
```

The training dataset includes the target variable `SalePrice`, while the test dataset does not.

---

## Project Structure

```text
HousePricePredict/
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   │
│   └── processed/
│       ├── processed_train.csv
│       └── processed_test.csv
│
├── images/
│   ├── actual_vs_predicted.png
│   ├── residual_distribution.png
│   ├── residual_plot.png
│   └── feature_coefficients.png
│
├── models/
│   └── linear_regression_model.joblib
│
├── results/
│   ├── metrics.txt
│   ├── validation_predictions.csv
│   └── submission.csv
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── linear_regression.py
│   └── utils.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

Some generated files shown above will be created after running the preprocessing and model-training scripts.

---

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- PyCharm
- Git
- GitHub

---

## Data Preprocessing Pipeline

The preprocessing pipeline is implemented in:

```text
src/preprocessing.py
```

### 1. Identifier Removal

The `Id` column is preserved for submission generation but removed from model features because it is only a record identifier.

### 2. Outlier Removal

Two extreme observations are removed using the following condition:

```python
(train["GrLivArea"] > 4000) &
(train["SalePrice"] < 300000)
```

After outlier removal, the training dataset contains:

```text
1458 observations
```

### 3. Target Transformation

The target variable is transformed with `log1p`:

```python
y = np.log1p(train["SalePrice"])
```

This transformation:

- Reduces right skewness
- Decreases the influence of extreme prices
- Improves compatibility with Linear Regression
- Makes relative prediction errors more meaningful

### 4. Missing-Value Handling

Missing values are handled according to their semantic meaning.

For features where missing means that the property does not contain that component, categorical values are replaced with:

```text
None
```

Examples include:

- PoolQC
- GarageType
- GarageFinish
- BsmtQual
- BsmtExposure
- FireplaceQu
- Fence
- Alley

Related numerical values are replaced with:

```text
0
```

Examples include:

- GarageArea
- GarageCars
- GarageYrBlt
- TotalBsmtSF
- BsmtFinSF1
- BsmtFinSF2
- MasVnrArea

`LotFrontage` is filled using the median value of the corresponding neighborhood.

Remaining categorical missing values are filled using the mode calculated from the training dataset.

Remaining numerical missing values are filled using the median calculated from the training dataset.

Using training statistics prevents data leakage from the test dataset.

### 5. Data-Type Correction

Some numerical-looking variables represent categories rather than measurable quantities.

These features are converted to categorical variables:

```text
MSSubClass
MoSold
```

For example, an `MSSubClass` value of 60 is not mathematically twice as large as 30.

### 6. Ordinal Encoding

Ordered categorical features are converted into meaningful numerical rankings.

Example quality mapping:

```python
{
    "None": 0,
    "Po": 1,
    "Fa": 2,
    "TA": 3,
    "Gd": 4,
    "Ex": 5
}
```

This mapping is applied to features such as:

- ExterQual
- ExterCond
- HeatingQC
- KitchenQual
- FireplaceQu
- GarageQual
- GarageCond
- BsmtQual
- BsmtCond
- PoolQC

Other ordinal variables use feature-specific mappings, including:

- BsmtExposure
- BsmtFinType1
- BsmtFinType2
- GarageFinish
- PavedDrive
- LotShape
- Utilities
- LandSlope
- Electrical
- Functional
- Fence

### 7. Feature Engineering

New features are generated from existing variables.

#### Total Area

```python
TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
```

#### Total Bathrooms

```python
TotalBathrooms = (
    FullBath
    + 0.5 * HalfBath
    + BsmtFullBath
    + 0.5 * BsmtHalfBath
)
```

#### Total Porch Area

```python
TotalPorchSF = (
    OpenPorchSF
    + EnclosedPorch
    + 3SsnPorch
    + ScreenPorch
    + WoodDeckSF
)
```

#### House Age

```python
HouseAge = YrSold - YearBuilt
```

#### Renovation Age

```python
RemodelAge = YrSold - YearRemodAdd
```

#### Garage Age

```python
GarageAge = YrSold - GarageYrBlt
```

Binary indicator features are also created:

```text
HasPool
HasGarage
HasBsmt
HasFireplace
WasRemodeled
```

### 8. One-Hot Encoding

Nominal categorical features are transformed using `OneHotEncoder`.

Examples include:

- Neighborhood
- MSZoning
- RoofStyle
- Exterior1st
- Exterior2nd
- Foundation
- SaleType
- SaleCondition

The encoder uses:

```python
handle_unknown="ignore"
```

This prevents errors when the test dataset contains a category that was not observed during training.

### 9. Feature Scaling

Numerical features are standardized using:

```python
StandardScaler
```

The transformation follows:

```text
z = (x - mean) / standard deviation
```

Feature scaling helps Linear Regression and regularized linear models operate on comparable numerical ranges.

### 10. Final Preprocessing Output

After preprocessing:

```text
Training rows: 1458
Test rows: 1459
Model features: 249
Missing training values: 0
Missing test values: 0
```

The processed datasets are saved as:

```text
data/processed/processed_train.csv
data/processed/processed_test.csv
```

---

## Linear Regression

The regression model is implemented in:

```text
src/linear_regression.py
```

Linear Regression estimates the target using:

```text
y = b0 + b1x1 + b2x2 + ... + bnxn
```

Where:

- `y` is the predicted logarithmic sale price
- `b0` is the intercept
- `bi` values are learned coefficients
- `xi` values are processed house features

The dataset is divided into training and validation subsets:

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

The baseline model is:

```python
LinearRegression()
```

---

## Evaluation Metrics

The model is evaluated with the following regression metrics.

### Mean Absolute Error

```text
MAE = average absolute difference between actual and predicted values
```

MAE is relatively easy to interpret and is less sensitive to very large errors than MSE.

### Mean Squared Error

```text
MSE = average squared prediction error
```

MSE penalizes large errors more heavily.

### Root Mean Squared Error

```text
RMSE = square root of MSE
```

For this project, log-scale RMSE is especially important because the target variable is logarithmically transformed.

### Coefficient of Determination

```text
R²
```

R² represents the proportion of target variance explained by the model.

A value closer to `1.0` indicates better explanatory performance.

---

## Current Results

The model evaluation results will be added after the final Linear Regression training run.

| Metric | Validation Result |
|---|---:|
| MAE — Log Scale | To be calculated |
| MSE — Log Scale | To be calculated |
| RMSE — Log Scale | To be calculated |
| R² Score | To be calculated |
| MAE — Original Price Scale | To be calculated |

The final metrics will also be saved in:

```text
results/metrics.txt
```

---

## Visualizations

The project will generate the following model-evaluation visualizations.

### Actual vs. Predicted Prices

This plot compares actual property prices with model predictions.

```text
images/actual_vs_predicted.png
```

Points close to the diagonal reference line indicate accurate predictions.

### Residual Distribution

This plot shows the distribution of prediction errors.

```text
images/residual_distribution.png
```

A distribution centered near zero is desirable.

### Residual Plot

This plot compares predicted values with their residual errors.

```text
images/residual_plot.png
```

A strong model should produce residuals without an obvious systematic pattern.

### Feature Coefficients

This plot displays features with the largest positive and negative Linear Regression coefficients.

```text
images/feature_coefficients.png
```

Coefficient interpretation should be performed carefully because one-hot encoded variables and correlated features can affect individual coefficient values.

---

## Installation

### 1. Clone the Repository

```bash
git clone REPOSITORY_URL
cd HousePricePredict
```

Replace `REPOSITORY_URL` with the GitHub repository URL.

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run all commands from the project root directory.

### 1. Run Preprocessing

```bash
python src/preprocessing.py
```

This creates:

```text
data/processed/processed_train.csv
data/processed/processed_test.csv
```

### 2. Train and Evaluate Linear Regression

```bash
python src/linear_regression.py
```

This stage will:

- Load processed data
- Split training and validation samples
- Train Linear Regression
- Generate predictions
- Calculate evaluation metrics
- Save the trained model
- Export validation predictions
- Produce evaluation visualizations

---

## Generated Outputs

After running the complete pipeline, the project will generate:

```text
data/processed/processed_train.csv
data/processed/processed_test.csv

models/linear_regression_model.joblib

results/metrics.txt
results/validation_predictions.csv
results/submission.csv

images/actual_vs_predicted.png
images/residual_distribution.png
images/residual_plot.png
images/feature_coefficients.png
```

Generated model files may be excluded from Git tracking according to `.gitignore`.

---

## Reproducibility

The validation split uses:

```python
random_state=42
```

This ensures that repeated runs use the same training and validation observations.

All missing-value statistics, category mappings, scaling parameters, and encoding rules are learned from the training dataset wherever applicable.

This design reduces data leakage and makes model evaluation more reliable.

---

## Future Improvements

Planned improvements include:

- Ridge Regression
- Lasso Regression
- Elastic Net Regression
- Random Forest Regression
- Gradient Boosting Regression
- XGBoost
- LightGBM
- Hyperparameter optimization
- Cross-validation
- Feature selection
- Multicollinearity analysis
- Automated model comparison
- Scikit-learn Pipeline integration
- Experiment tracking
- Unit tests
- Command-line execution
- Model deployment through an API

The first objective is to establish Linear Regression as an interpretable baseline model. More advanced models can then be compared against this baseline using the same validation strategy.

---

## Important Notes

- The target variable is trained on the logarithmic scale.
- Use `np.expm1()` to convert predictions back to the original price scale.
- The `Id` variable is not used as a model feature.
- Test data must never be used to calculate preprocessing statistics.
- Model performance must be evaluated on data not used during training.
- Linear Regression coefficients do not always represent causal effects.
- Highly correlated features may cause unstable coefficients.

---

## License

This project is licensed under the MIT License.

See the `LICENSE` file for details.

---

## Author

Developed as a machine learning portfolio project focused on:

- Data preprocessing
- Feature engineering
- Regression modeling
- Model evaluation
- Reproducible project organization