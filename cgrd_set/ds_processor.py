import pandas as pd
import os
current_path = os.path.dirname(__file__)


#note_1 = pd.read_csv('14653_出院病摘_1.csv')
# print(note_1.columns.values)
#selected_cols = ['入院診斷', '出院診斷', '主訴', '病史', '手術日期、方法及發現', '住院治療經過']
#note_1 = note_1[selected_cols]
#note_1 = note_1[0:100]

note_1 = pd.read_csv('ds.csv')
print(note_1.head())

print('done')