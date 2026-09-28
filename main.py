import pandas as pd

from src.data import load_data, clean_data
from src.train import models, train_all_models
from src.evaluate import evaluate_models

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


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