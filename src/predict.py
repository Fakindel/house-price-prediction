import joblib
import pandas as pd

model =  joblib.load("final_model.pkl")
new_house = pd.DataFrame