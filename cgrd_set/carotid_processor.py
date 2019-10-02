import spacy
import re
import pandas as pd
import os
from collections import Counter
current_path = os.path.dirname(__file__)

selected_cols = ['CONTENT', 'RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA',
                 'LCCA', 'LEICA', 'LIICA', 'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA', 'over_2weeks']

note_all = pd.read_csv('carotid_101318_all.csv')
note_all.dropna(axis=0, subset=['CONTENT'], inplace=True)
print(note_all.shape)
for inx, row in note_all.iterrows():
    corpus = row['CONTENT']
    rcca = row['RCCA']
    if '<BASE64>' in corpus:
        note_all.drop(index=inx, inplace=True)
    if rcca == 9:
        note_all.drop(index=inx, inplace=True)
note_all.to_csv('aa.csv')
print(note_all.shape)
note_all = note_all[selected_cols]
note_bert = note_all[note_all.over_2weeks == 1]
print(note_bert.shape)
note_down_task = note_all[note_all.over_2weeks == 0]
print(note_down_task.shape)

