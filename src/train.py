
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)

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
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def train_all_models(models, X_train, y_train):
    result = []
    for name, model in models.items():
        trained_model = train_model(models, X_train, y_train)

    result.append((name, trained_model))
    return result