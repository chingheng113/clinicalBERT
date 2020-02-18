import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
import os
import pickle
import matplotlib.pyplot as plt
from scipy import interp

for model in ['sb', 'c']:
    aucs = []
    fprs = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 300)
    precisions =[]
    recalls =[]
    fscores =[]
    for i in range(10):
        model_name = model+'_all_'+str(i)
        # print(model_name)
        read_path = os.path.join('restroke_all', model_name, 'test_prediction.pickle')
        with open(read_path, 'rb') as f:
            data = pickle.load(f)
            all_logits = data['all_logits']
            all_labels = data['all_labels']
            fpr, tpr, _ = roc_curve(all_labels, all_logits[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
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
    # https: // scikit - learn.org / stable / auto_examples / model_selection / plot_roc_crossval.html
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    plt.plot(mean_fpr, mean_tpr,
             label=model+' (AUC = %0.3f $\pm$ %0.3f)' % (np.mean(aucs), np.std(aucs))
             )
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.2,
                     label= r'$\pm$ 1 std. dev.')

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black', label='Luck', alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('title')
plt.legend(loc="lower right")
plt.show()