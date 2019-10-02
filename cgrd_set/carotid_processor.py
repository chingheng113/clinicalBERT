import spacy
import re
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
current_path = os.path.dirname(__file__)


def save_variable(val, val_name):
    save_path = os.path.join(current_path, val_name)
    with open(save_path, 'wb') as file_pi:
        pickle.dump(val, file_pi)

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
note_all = note_all[selected_cols]
print(note_all.shape)
note_bert = note_all[note_all.over_2weeks == 1]
print(note_bert.shape)
note_down_task = note_all[note_all.over_2weeks == 0]
print(note_down_task.shape)

nlp = spacy.load('en_core_sci_md', disable=['tagger','ner'])

# BERT training
note_bert = note_bert[0:100]
with open('../data/carotid.txt', 'w', encoding="utf-8") as f:
    for inx, row in note_bert.iterrows():
        # print(selected_cols[i])
        parg = str(row['CONTENT'])
        # remove special characters
        parg = re.sub(r'(^;)|(;;;)|(=+>)|(={2})|(-+>)|(={2})|(\*{3})', '', parg)
        # remove multi-spaces
        parg = re.sub(r' +', ' ', parg)
        # remove Chinese
        parg = re.sub(r'[\u4e00-\u9fff]+', '', parg)
        sentences = nlp(parg)
        processed_sentence = ''
        for sentence in sentences.sents:
            # trim white space at the head and the end
            see_sentence = sentence.text.strip()
            # if the sentence is very short or doesn't contain any word or start with non-character, remove it!
            if not ((len(see_sentence.split()) < 3) | (see_sentence == 'nil') | (
                    len(re.findall(r'[a-zA-Z_]{2}', see_sentence)) < 3)):
                if not re.search(r'\W', see_sentence).start() == 0:
                    see_sentence = re.sub(r'[\n]+', '\n', see_sentence)
                    processed_sentence += see_sentence
        f.write(processed_sentence)
        f.write('\n\n')

# Down stream task dataset
# note_down_task = note_down_task[0:100]
text_arr = []
label_arr = []
for inx, row in note_down_task.iterrows():
    label = row[['RCCA', 'REICA', 'RIICA', 'RACA', 'RMCA', 'RPCA', 'REVA', 'RIVA', 'BA',
                 'LCCA', 'LEICA', 'LIICA', 'LACA', 'LMCA', 'LPCA', 'LEVA', 'LIVA']].values
    parg = str(row['CONTENT'])
    # remove special characters
    parg = re.sub(r'(^;)|(;;;)|(=+>)|(={2})|(-+>)|(={2})|(\*{3})', '', parg)
    # remove multi-spaces
    parg = re.sub(r' +', ' ', parg)
    # print(parg)
    # remove Chinese
    parg = re.sub(r'[\u4e00-\u9fff]+', '', parg)
    sentences = nlp(parg)
    processed_sentence = ''
    for sentence in sentences.sents:
        # trim white space at the head and the end
        see_sentence = sentence.text.strip()
        # if the sentence is very short or doesn't contain any word or start with non-character, remove it!
        if not ((len(see_sentence.split()) < 3) | (see_sentence == 'nil') | (
                len(re.findall(r'[a-zA-Z_]{2}', see_sentence)) < 3)):
            if not re.search(r'\W', see_sentence).start() == 0:
                processed_sentence += see_sentence+'\n'
    text_arr.append(processed_sentence)
    label_arr.append(label)

text_arr = np.array(text_arr)
label_arr = np.array(label_arr)

# Train, Test split
x_train, x_test, Y_train, Y_test = train_test_split(text_arr, label_arr, test_size=0.2, random_state=42)

save_variable([x_train, Y_train], os.path.join('..', 'downstream_tasks', 'data', 'training_bert.pickle'))
save_variable([x_test, Y_test], os.path.join('..', 'downstream_tasks', 'data', 'test_bert.pickle'))

print('done')