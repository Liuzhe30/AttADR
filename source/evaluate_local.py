import numpy as np
import  pandas as pd

pred = np.load("y_pred.npy")
true = np.load("y_true.npy")
y_pred = []
for item in pred:
    if(item >= 0.5):
        y_pred.append(1)
    else:
        y_pred.append(0)
pred = np.array(y_pred)
print(true)
#print(pred)

from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

print("acc = " + str(accuracy_score(true, pred)))
print("precision = " + str(precision_score(true, pred)))
print("recall = " + str(recall_score(true, pred)))
print("f1_score = " + str(f1_score(true, pred)))


f,ax=plt.subplots()
ax.set_title('confusion matrix',fontsize=12)
cm = confusion_matrix(true, pred,labels=[0, 1])
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm,annot=True,ax=ax,cmap='Blues')
plt.show()


fpr,tpr,threshold = roc_curve(true, pred) 
roc_auc = auc(fpr,tpr)
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc) 
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()