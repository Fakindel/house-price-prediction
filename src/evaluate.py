from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)
def evaluate_models(trained_models, X_test, y_test):
    result = []
    for name, model in trained_models:
        
        prediction = model.predict(X_test)
        mae = mean_absolute_error(y_test, prediction)
        rmse = root_mean_squared_error(y_test, prediction)
        r2 = r2_score(y_test, prediction)
        result.append((name, mae, rmse, r2))
    return result
