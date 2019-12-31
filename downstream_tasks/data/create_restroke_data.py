import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import numpy as np
import re


def clean_text(parg):
    if parg == parg:
        # remove special characters
        parg = re.sub(r'(\s\.)|(\.{2,})|(:{2,})|(^;)|(;;;)|(=+>)|(={2})|(-+>)|(={2})|(\*{3})', '', parg)
        # fix bullet points
        parg = re.sub(r'(\d+[.{1}][\D])', r'. \1', parg)
        # remove multi-spaces
        parg = re.sub(r'[\s\t]+', ' ', parg)
        # print(parg)
        # remove Chinese
        parg = re.sub(r'[\u4e00-\u9fff]+', '', parg)
        # remove date
        parg = re.sub(r'\d{2,4}[/]\d{1,2}[/]\d{1,2}', '', parg)
    return parg


df = pd.read_csv('recurrent_stroke_ds.csv')
df.dropna(axis=0, subset=['主訴', '病史', '住院治療經過'], inplace=True)

df['processed_content'] = df['主訴']+" "+df['病史']+" "+df['住院治療經過']
df['processed_content'] = df['processed_content'].apply(clean_text)
data = df[['歸戶代號', 'processed_content', 'label']]
# sample balance
resampled = resample(data[data.label == 0],
                     replace=False,
                     n_samples=data[data.label == 1].shape[0],
                     random_state=123)
data = pd.concat([data[data.label == 1], resampled])
#
for i in range(10):
    training_data, testing_data = train_test_split(data, test_size=0.2, random_state=i)
# data.to_csv('see.csv', index=False, encoding='utf-8-sig')
print('done')