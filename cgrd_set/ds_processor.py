import spacy
import re
import pandas as pd
import os
from collections import Counter
current_path = os.path.dirname(__file__)


note_1 = pd.read_csv('14653_出院病摘_1.csv')
# print(note_1.columns.values)
selected_cols = ['入院診斷', '出院診斷', '主訴', '病史', '手術日期、方法及發現', '住院治療經過']
note_1 = note_1[selected_cols]
note_1 = note_1[0:100]

# note_1 = pd.read_csv('ds.csv')
# print(note_1.head())
nlp = spacy.load('en_core_sci_md', disable=['tagger','ner'])
with open('ds.txt', 'w', encoding="utf-8") as f:
    for inx, row in note_1.iterrows():
        for i in range(len(selected_cols)):
            print(selected_cols[i])
            parg = str(row[selected_cols[i]])
            # remove special characters
            parg = re.sub(r'(^;)|(;;;)|(=+>)|(={2})|(-+>)|(={2})|(\*{3})', '', parg)
            # fix bullet points
            parg = re.sub(r'(\d+[.{1}][\D])', r'. \1', parg)
            # remove multi-spaces
            parg = re.sub(' +', ' ', parg)
            print(parg)
            # remove Chinese
            parg = re.sub(r'[\u4e00-\u9fff]+', '', parg)
            # remove date
            parg = re.sub(r'\d{2,4}[/]\d{1,2}[/]\d{1,2}', '', parg)
            sentences = nlp(parg)
            for sentence in sentences.sents:
                # remove bullet point number
                see_sentence = re.sub(r'(^\d+\.)|(^\s+)', '', sentence.text)
                # trim white space at the head and the end
                see_sentence = see_sentence.strip()
                # if sentence is very short or doesn't contain any word, remove it!
                if len(see_sentence.split()) > 2:
                    print(see_sentence)
                    # f.write(see_sentence)
                    # f.write('\n')
        # f.write('\n')

print('done')
