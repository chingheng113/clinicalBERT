import pandas as pd
import numpy as np
import re


def clean_text(parg):
    if parg == parg:
        # remove special characters
        parg = re.sub(r'(^;)|(;;;)|(=+>)|(={2})|(-+>)|(={2})|(\*{3})', '', parg)
        # fix bullet points
        parg = re.sub(r'(\d+[.{1}][\D])', r'. \1', parg)
        # remove multi-spaces
        parg = re.sub(r' +', ' ', parg)
        # print(parg)
        # remove Chinese
        parg = re.sub(r'[\u4e00-\u9fff]+', '', parg)
        # remove date
        parg = re.sub(r'\d{2,4}[/]\d{1,2}[/]\d{1,2}', '', parg)
    return parg


df = pd.read_csv('recurrent_stroke_ds.csv')
df.dropna(axis=0, subset=['主訴', '病史', '住院治療經過'], inplace=True)
df['主訴'] = df['主訴'].apply(clean_text)
df['病史'] = df['病史'].apply(clean_text)
df['住院治療經過'] = df['住院治療經過'].apply(clean_text)
print('done')