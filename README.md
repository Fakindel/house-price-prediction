# House Price Prediction

A machine learning project that predicts house sale prices using the Ames Housing dataset and several regression models built with Python and scikit-learn.

## Project Overview

The goal of this project is to build a regression model that can predict the sale price of a house from its features, such as living area, neighborhood, number of rooms, and other property characteristics.

The project compares multiple machine learning models, evaluates them using cross-validation, and saves the final model so it can be used to make predictions on new houses.

## Models Used

The following regression models were trained and compared:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- Extra Trees Regressor

## Data Preprocessing

The dataset contains both numerical and categorical features.

The preprocessing pipeline includes:

- Missing-value imputation
- Median imputation for numerical features
- Most-frequent imputation for categorical features
- One-hot encoding for categorical features
- Ignoring unseen categories during prediction

The preprocessing is included inside the scikit-learn pipeline so that it is fitted separately within each cross-validation training fold, helping prevent data leakage.

## Model Evaluation

Five-fold cross-validation produced the following average results:

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | 19,733.76 | 38,533.81 | 0.7413 |
| Random Forest | 18,152.24 | 30,631.27 | 0.8386 |
| Gradient Boosting | 16,230.95 | 29,307.29 | 0.8513 |
| Extra Trees | 19,155.47 | 32,521.74 | 0.8204 |

The final model used in the project is Gradient Boosting Regressor.

## Project Structure

```text
house-price-prediction/
│
├── data/
│   └── train.csv
│
├── src/
│   ├── data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
├── main.py
├── final_model.pkl
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/Fakindel/house-price-prediction.git
cd house-price-prediction
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Training the Models

Run:

```bash
python main.py
```

This loads and cleans the data, creates the preprocessing pipeline, trains the models, performs cross-validation, evaluates the final model, and saves the trained pipeline as:

```text
final_model.pkl
```

## Making a Prediction

The saved model can be loaded with `predict.py`:

```bash
python src/predict.py
```

The prediction pipeline loads the trained model and produces a predicted house price.

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- joblib
- Git and GitHub

## Dataset

This project uses the Ames Housing dataset, commonly distributed through the Kaggle House Prices competition.

The dataset contains information about residential properties and their sale prices.

## Future Improvements

Possible improvements include:

- Hyperparameter tuning
- Feature engineering
- Trying additional regression models
- Building a web interface for predictions
- Adding automated tests
- Deploying the model as an API or web application