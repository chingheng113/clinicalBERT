import pandas as pd
import numpy as np
import os
import pickle
import re
from sklearn.model_selection import train_test_split
current_path = os.path.dirname(__file__)

def clean_text(parg):
    if parg == parg:
        a = re.search('findings:', parg, re.IGNORECASE)
        parg = re.sub(r'(:nil)', '', parg)
        # remove special characters
        parg = re.sub(r'(\s\.)|(\.{2,})|(:{2,})|(^;)|(;;;)|(=+>)|(={2})|(-+>)|(={2})|(\*{3})', '', parg)
        # fix bullet points
        parg = re.sub(r'(\d+[.{1}][\D])', r'. \1', parg)
        # remove Chinese
        parg = re.sub(r'[\u4e00-\u9fff]+', ' ', parg)
        # remove multi-spaces
        parg = re.sub(r'[\s\t]+', ' ', parg)
        # remove date
        parg = re.sub(r'\d{2,4}[/]\d{1,2}[/]\d{1,2}', '', parg)
    return parg


def save_variable(val, val_name):
    save_path = os.path.join(current_path, val_name)
    with open(save_path, 'wb') as file_pi:
        pickle.dump(val, file_pi)

# read data
data = pd.read_csv('carotid2.csv')
data.rename(columns={'IDCode': 'ID', 'RTRESTXT': 'processed_content'}, inplace=True)
data.dropna(subset=['processed_content'], axis=0, inplace=True)
data['processed_content'] = data['processed_content'].apply(clean_text)

data.to_csv(os.path.join('carotid2', 'testing.csv'), index=False)
save_variable(data, os.path.join('carotid2', 'test_bert.pickle'))


print('done')