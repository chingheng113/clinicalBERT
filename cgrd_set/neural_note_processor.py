import spacy
import re
import pandas as pd
import os
current_path = os.path.dirname(__file__)


note_1 = pd.read_csv('14653_腦神經內科報告_1.csv')
# print(note_1.columns.values)

selected_cols = ['檢查項目', '報告01']
note_1 = note_1[selected_cols]
# remove M29-079: carotid report...
note_1 = note_1[note_1['檢查項目'] != 'M29-079']
note_1 = note_1[0:100]

delimiters = ['Conclusion:', 'Comments:', 'Interpretation:', 'INTERPRETATION:', 'Doppler Findings:', 'COMMENTS:']
nlp = spacy.load('en_core_sci_md', disable=['tagger', 'ner'])
with open('../data/nu_note.txt', 'w', encoding="utf-8") as f:
    for inx, row in note_1.iterrows():
        parg = str(row[selected_cols[1]])
        for delimiter in delimiters:
            print(parg.find(delimiter))
        print('--')
        # parg = parg.lower()
        # remove special characters
        # parg = re.sub(r'(^;)|(;;;)|(=+>)|(_{2})|(={2})|(-{2})|(={2})|(\*{3})', '', parg)
        # remove Chinese
        # parg = re.sub(r'[\u4e00-\u9fff]+', '', parg)

print('done')
