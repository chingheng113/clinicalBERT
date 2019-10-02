import pandas as pd
import numpy as np
import os
import pickle
import re
from sklearn.model_selection import train_test_split
current_path = os.path.dirname(__file__)


def save_variable(val, val_name):
    save_path = os.path.join(current_path, val_name)
    with open(save_path, 'wb') as file_pi:
        pickle.dump(val, file_pi)

# read data
data = pd.read_csv('carotid_101518_modified.csv')
data.dropna(subset=['CONTENT'], axis=0, inplace=True)
# Preprocessing
text_arr = []
label_arr = []
for index, row in data.iterrows():
    label = row[['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA',
                 'LCCA', 'LEICA', 'LIICA', 'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']].values
    corpus = row['CONTENT']
    if '<BASE64>' not in corpus:
        sentences = corpus.split('\n')
        processed_sentence = ''
        for sentence in sentences:
            if len(re.findall(r'[\u4e00-\u9fff]+', sentence)) == 0:
                # no chinese sentence
                if re.search('(>\s*\d+|<\s*\d+)', sentence):
                    sentence = sentence.replace('>', ' greater ')
                    sentence = sentence.replace('<', ' less ')
                sentence = sentence.replace('%', ' percent')
                processed_sentence += sentence+' '
        processed_sentence = re.sub(' +', ' ', processed_sentence)
        text_arr.append(processed_sentence)
        # multi-babel
        label_arr.append(label)
        # binary
        # if sum(label) == 0:
        #     label_arr.append('0')
        # else:
        #     label_arr.append('1')
text_arr = np.array(text_arr)
label_arr = np.array(label_arr)
# Train, Test split
x_train, x_test, Y_train, Y_test = train_test_split(text_arr, label_arr, test_size=0.2, random_state=42)

save_variable([x_train, Y_train], os.path.join(current_path, 'downstream_tasks', 'training_bert.pickle'))
save_variable([x_test, Y_test], os.path.join(current_path, 'downstream_tasks', 'test_bert.pickle'))

print('done')