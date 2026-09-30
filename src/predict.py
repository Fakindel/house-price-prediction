import joblib
import pandas as pd

model =  joblib.load("final_model.pkl")
df = pd.read_csv("data/train.csv")

new_house = df.drop(["SalePrice", "Id"], axis=1).iloc[[0]]
prediction = model.predict(new_house)
print(f"Predicted house price: ${prediction[0]:,.2f}")