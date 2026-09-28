
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
def train_all_models(models, X_train, y_train):
    result = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        result.append((name, model))
    return result