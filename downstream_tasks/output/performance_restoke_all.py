import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
import os
import pickle

aucs = []
precisions =[]
recalls =[]
fscores =[]
for i in range(10):
    model_name = 'c_all_'+str(i)
    # print(model_name)
    read_path = os.path.join('restroke_all', model_name, 'test_prediction.pickle')
    with open(read_path, 'rb') as f:
        data = pickle.load(f)
        all_logits = data['all_logits']
        all_labels = data['all_labels']
        fpr, tpr, _ = roc_curve(all_labels, all_logits[:, 1])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # print(roc_auc)
        p_label_b = (all_logits[:, 1] > 0.5).astype(int)
        precision, recall, fscore, support = score(all_labels, p_label_b, average='macro')
        precisions.append(precision)
        recalls.append(recall)
        fscores.append(fscore)
        # print(classification_report(all_labels, p_label_b))
        # print(confusion_matrix(all_labels, p_label_b))

# print(round(np.mean(precisions),3), round(np.std(precisions),3))
# print(round(np.mean(recalls),3), round(np.std(recalls),3))
# print(round(np.mean(fscores),3), round(np.std(fscores),3))
print(round(np.mean(aucs),3), round(np.std(aucs),3))
print('done')