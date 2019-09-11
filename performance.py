import pandas as pd
from sklearn.metrics import roc_curve, auc
import os
import pickle

read_path = os.path.join('downstream_tasks', 'output', 'test_prediction.pickle')

with open(read_path, 'rb') as f:
    data = pickle.load(f)
    all_logits = data['all_logits']
    all_labels = data['all_labels']

    print('done')