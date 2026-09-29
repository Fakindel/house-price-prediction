import pandas as pd
<<<<<<< HEAD
=======
import joblib
import numpy as np
from src.train import build_model_pipeline
from sklearn.model_selection import cross_val_score
>>>>>>> 3c2439e (done)

from src.data import load_data, clean_data
from src.train import models, train_all_models
from src.evaluate import evaluate_models

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

<<<<<<< HEAD
=======
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)
from sklearn.metrics import root_mean_squared_error

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
    
)

>>>>>>> 3c2439e (done)

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
# 12. TRAIN ALL MODELS
# ============================================================

trained_models = train_all_models(
    models,
    X_train_processed,
    y_train
)


# ============================================================
# 13. EVALUATE ALL MODELS
# ============================================================

result = evaluate_models(
    trained_models,
    X_test_processed,
    y_test
)


# ============================================================
# 14. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    result,
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2"
    ]
)

print(
    "\n================ MODEL COMPARISON ================"
)

print(
    results_df.to_string(index=False)
)

#from sklearn.model_selection import cross_val_score


# ============================================================
# 15. CROSS-VALIDATION
# ============================================================

print("\n================ CROSS-VALIDATION ================")

for name, model in models.items():

    model_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # MAE
    mae_scores = cross_val_score(
        model_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="neg_mean_absolute_error"
    )

    # RMSE
    rmse_scores = cross_val_score(
        model_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="neg_root_mean_squared_error"
    )

    # R2
    r2_scores = cross_val_score(
        model_pipeline,
        X_train,
        y_train,
        cv=5,
        scoring="r2"
    )

    # Convert negative error scores to positive
    mean_mae = -mae_scores.mean()
    mean_rmse = -rmse_scores.mean()
    mean_r2 = r2_scores.mean()

    print(f"\n{name}")
    print(f"Average MAE:  {mean_mae:.2f}")
    print(f"Average RMSE: {mean_rmse:.2f}")
    print(f"Average R²:   {mean_r2:.4f}")

final_model = build_model_pipeline(preprocessor, GradientBoostingRegressor( n_estimators=300,random_state=42))
final_model.fit(X_train, y_train)
final_prediction = final_model.predict(X_test)
final_mae = mean_absolute_error(y_test, final_prediction)
final_rmse = root_mean_squared_error(y_test, final_prediction)
final_r2 = r2_score(y_test, final_prediction)
print("\n================ FINAL MODEL ================")
print(f"MAE:  {final_mae:.2f}")
print(f"RMSE: {final_rmse:.2f}")
print(f"R²:   {final_r2:.4f}")
joblib.dump(final_model, "final_model.pkl")
print("\nFinal model saved as final_model.pkl")
