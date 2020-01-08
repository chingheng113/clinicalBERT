import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
import os
import pickle

aucs = []
for i in range(10):
    model_name = 'sb_all_'+str(i)
    print(model_name)
    read_path = os.path.join('restroke', model_name, 'test_prediction.pickle')
    with open(read_path, 'rb') as f:
        data = pickle.load(f)
        all_logits = data['all_logits']
        all_labels = data['all_labels']
        fpr, tpr, _ = roc_curve(all_labels, all_logits[:, 1])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # print(roc_auc)
        p_label_b = (all_logits[:, 1] > 0.5).astype(int)
        print(classification_report(all_labels, p_label_b))
        print(confusion_matrix(all_labels, p_label_b))
print(np.mean(aucs), np.std(aucs))
print('done')