import pandas as pd
import numpy as np

from src.data import load_data, clean_data

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. LOAD DATA
# ============================================================

df = load_data()

print("Original shape:", df.shape)


# ============================================================
# 2. CLEAN DATA
# ============================================================

df = clean_data(df)

# Check remaining missing values
missing = df.isnull().sum()

print("\nRemaining missing values:")

if missing[missing > 0].empty:
    print("None")
else:
    print(missing[missing > 0].to_string())


# ============================================================
# 3. CREATE FEATURES (X) AND TARGET (y)
# ============================================================

X = df.drop(["SalePrice", "Id"], axis=1)
y = df["SalePrice"]

print("\nX shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 4. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# ============================================================
# 5. IDENTIFY COLUMN TYPES
# ============================================================

categorical_columns = X_train.select_dtypes(
    include=["object", "string"]
).columns

numerical_columns = X_train.select_dtypes(
    exclude=["object", "string"]
).columns

print("\nCategorical columns:", len(categorical_columns))
print("Numerical columns:", len(numerical_columns))


# ============================================================
# 6. CREATE IMPUTERS
# ============================================================

categorical_imputer = SimpleImputer(
    strategy="most_frequent"
)

numerical_imputer = SimpleImputer(
    strategy="median"
)


# ============================================================
# 7. CREATE ENCODER
# ============================================================

encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


# ============================================================
# 8. CATEGORICAL PIPELINE
# ============================================================

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", categorical_imputer),
        ("encoder", encoder)
    ]
)


# ============================================================
# 9. NUMERICAL PIPELINE
# ============================================================

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", numerical_imputer)
    ]
)


# ============================================================
# 10. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            categorical_pipeline,
            categorical_columns
        ),
        (
            "num",
            numerical_pipeline,
            numerical_columns
        )
    ]
)


# ============================================================
# 11. TRANSFORM DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)

print(
    "\nProcessed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape:",
    X_test_processed.shape
)


# ============================================================
# 12. CREATE MODELS
# ============================================================

models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        random_state=42
    ),

    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 13. TRAIN + EVALUATE MODELS
# ============================================================

results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Train
    model.fit(
        X_train_processed,
        y_train
    )

    # Predict
    predictions = model.predict(
        X_test_processed
    )

    # Metrics
    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    # Store results
    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print(
    "\n================ MODEL COMPARISON ================"
)

print(
    results_df.to_string(index=False)
)