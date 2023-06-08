import pandas as pd
import os

os.chdir('C:\\Users\\JongbeenSong\\Desktop\\23-1\\빅데이터 처리\\BDP-main\\price_recommender')

df_kaggle = pd.read_csv('data\\kaggle_data.csv', encoding='utf-8')
df_combined_data = pd.read_csv('data\\combined_data_for_price_recommender.csv', encoding='cp949')

df_kaggle_extracted = df_kaggle[['AppID', 'Name', 'Price']]
df_combined_data_extracted = df_combined_data[['Currently popular', 'Game Description']]

df_concat = pd.concat([df_kaggle_extracted, df_combined_data_extracted], axis=1)

df_concat['Price'] = df_concat['Price'].astype(float)
df_concat = df_concat.dropna()

df_concat.to_csv('data\\combined_data_for_price_recommender_merged.csv', index=False)
