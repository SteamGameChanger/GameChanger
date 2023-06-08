import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

os.chdir('C:\\Users\\JongbeenSong\\Desktop\\23-1\\빅데이터 처리\\BDP-main\\price_recommender')

class PredictPrice:
    def __init__(self):
        self.vectorizer = joblib.load('tfidf_vectorizer.pkl')
        self.model = joblib.load('xgb_model.pkl')

    def predict(self, description):
        # Transform the description into a format suitable for the model
        transformed_description = self.vectorizer.transform([description])

        # Predict and return the price
        predicted_price = self.model.predict(transformed_description)

        return predicted_price[0]

game_description = ""

if __name__ == "__main__":
    pp = PredictPrice()
    # description = input("Enter game description: ")
    description = game_description
    predicted_price = pp.predict(description)
    predicted_price = round(predicted_price, 2)

    print(f"The predicted price is: ${predicted_price:.2f}")
