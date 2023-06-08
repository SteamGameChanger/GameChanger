import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os


# os.chdir('C:\Users\장경빈\PycharmProjects\price_recommender_test')

class PredictPrice:
    def __init__(self):
        self.vectorizer = joblib.load('/content/drive/MyDrive/BigData/tfidf_vectorizer.pkl')
        self.model = joblib.load('/content/drive/MyDrive/BigData/xgb_model.pkl')

    def predict(self, description):
        # Transform the description into a format suitable for the model
        transformed_description = self.vectorizer.transform([description])

        # Predict and return the price
        predicted_price = self.model.predict(transformed_description)

        return predicted_price[0]


# game_description = "Galactic Bowling is an exaggerated and stylized bowling game with an intergalactic twist. Players will engage in fast-paced single and multi-player competition while being submerged in a unique new universe filled with over-the-top humor, wild characters, unique levels, and addictive game play. The title is aimed at players of all ages and skill sets. Through accessible and intuitive controls and game-play, Galactic Bowling allows you to jump right into the action. A single-player campaign and online play allow you to work your way up the ranks of the Galactic Bowling League! Whether you have hours to play or only a few minutes, Galactic Bowling is a fast paced and entertaining experience that will leave you wanting more!                     Full Single-player story campaign including 11 Characters and Environments.                     2 Single-player play modes including Regular and Battle Modes.                     Head to Head Online Multiplayer play Modes.                     Super Powers, Special Balls, and Whammies.                     Unlockable Characters, Environments, and Minigames.                     Unlock all 30 Steam Achievements!"

if __name__ == "__main__":
    df = pd.read_csv('/content/drive/MyDrive/BigData/combined_data_for_similar_game_finder_review_and_description.csv',
                     encoding='cp949')
    df2 = pd.read_csv('/content/drive/MyDrive/BigData/kaggle_data.csv', encoding='utf-8')
    pp = PredictPrice()
    taget_loss = 10
    # description = input("Enter game description: ")
    df2['predict price'] = 0.0
    print(df['Game Description'].isnull().sum())

    for i in range(len(df[0:100])):
        if pd.isna(df['Game Description'][i]):
            continue
        description = df['Game Description'][i]
        predicted_price = pp.predict(description)
        predicted_price = round(predicted_price, 2)
        df2['predict price'][i] = predicted_price

    df2['predict loss'] = abs(df2['Price'] - df2['predict price'])
    df2.to_csv('/content/drive/MyDrive/BigData/price_predict_data.csv')
    df_result = df2.loc[(df2['predict loss'] < taget_loss) & (df2['predict price'] > 0)]
    # print(len(df_result.columns))
    df_result.sort_values(by=df_result.columns[25], ascending=True, inplace=True, na_position='last')
    print(df_result['Name'][:20])

    # print(f"The predicted price is: ${predicted_price:.2f}")