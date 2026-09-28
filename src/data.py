import pandas as pd
def load_data():
    df = pd.read_csv("data/train.csv")
    return df
def clean_data(df):
    none_columns = [
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "Alley",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature"
]
    for column in none_columns:
        df[column] = df[column].fillna("NA")
    df["MasVnrType"] = df["MasVnrType"].fillna("None")
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)
    return df