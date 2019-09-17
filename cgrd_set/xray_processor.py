import pandas as pd
import os
current_path = os.path.dirname(__file__)

# 1: MRI

x_note_1 = pd.read_csv('14653_出院病摘_1.csv')
# x_note_2 = pd.read_csv('14653_X光科報告_2.csv')

print('done')