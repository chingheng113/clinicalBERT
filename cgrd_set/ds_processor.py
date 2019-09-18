import spacy
import re
import pandas as pd
import os
current_path = os.path.dirname(__file__)


#note_1 = pd.read_csv('14653_出院病摘_1.csv')
# print(note_1.columns.values)
selected_cols = ['入院診斷', '出院診斷', '主訴', '病史', '手術日期、方法及發現', '住院治療經過']
#note_1 = note_1[selected_cols]
#note_1 = note_1[0:100]

note_1 = pd.read_csv('ds.csv')
# print(note_1.head())
nlp = spacy.load('en_core_sci_md', disable=['tagger','ner'])
regex = re.compile('(^;)|(;;;)|(=+>)|(={2})|(\*{3})')
for inx, row in note_1.iterrows():
    for i in range(len(selected_cols)):
        print(selected_cols[i])
        parg = str(row[selected_cols[i]])
        parg= re.sub(regex, '', parg)
        print(parg)

        print('---')
print('done')



# ;          ;;;, ;;;, ******,  ==> 2007/07/13