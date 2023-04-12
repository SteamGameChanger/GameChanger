import pandas as pd

merged_df = pd.DataFrame()

print("Merge In Progress...")
# range 범위는 끝 범위만 데이터 수에 맞게 조절
for n in range(0, 100):
    filename = f'steamGameNo{n}.csv'
    dframe = pd.read_csv('../dataset/' + filename)
    merged_df = pd.concat([merged_df, dframe], axis=0, ignore_index=False)
  
merged_df.to_csv('../dataset/steam_game.csv', encoding='UTF-8', mode='w', index=True)
print("Merge Completed!")

    