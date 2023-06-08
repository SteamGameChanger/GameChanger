import pandas as pd
import statistics

CSV_FILE = 'similar_game_finder\\data\\combined_data_for_similar_game_finder.csv'
data = pd.read_csv(CSV_FILE, encoding='cp949')

# finding mean and median of the number of supported languages except games only supporting English
len_language_list = []

for idx, language in enumerate(data['Supported languages']):
    language = language.replace('[', '').replace(']', '').replace('\'', '')
    temp = language.split(', ')
    if len(temp) == 1:
        continue
    len_language_list.append(len(temp))

print(len_language_list)
print(statistics.mean(len_language_list)) # 7.305493316850807
print(statistics.median(len_language_list)) # 5.0

# vectorizeing supported languages
vectorized_language_list = []

for idx, language in enumerate(data['Supported languages']):
    language = language.replace('[', '').replace(']', '').replace('\'', '')
    temp = language.split(', ')
    if len(temp) == 1:
        vectorized_language_list.append(0)
        continue
    elif len(temp) < 7.305493316850807:
        vectorized_language_list.append(1)
    else:
        vectorized_language_list.append(2)

print(vectorized_language_list)

# save vectorized supported languages
data['Supported languages'] = vectorized_language_list
data.to_csv(CSV_FILE, encoding='cp949', index=False)
