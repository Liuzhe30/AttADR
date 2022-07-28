import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

inputn = np.load("input.npy") # (500, 3246, 2)
att_out = np.load("att_out.npy") # (500, 3246, 2)
att_out2 = np.load("att_out2.npy") # (500, 1623, 2)
att_out3 = np.load("att_out3.npy") # (500, 811, 2)
label = np.load("label_test500.npy") # (500,)
print(label.shape)

select_f = att_out3

fig = plt.figure()
tsne = TSNE(n_components=2, init='pca', random_state=0)
stack = np.concatenate((select_f[:,:,0], select_f[:,:,1]), axis=1)
print(stack.shape)
result = tsne.fit_transform(stack)
print(result.shape)
#fig = plot_embedding(result, label,'t-SNE embedding of the digits')
x_min, x_max = np.min(result, 0), np.max(result, 0)
result = (result - x_min) / (x_max - x_min)

color = ["#B0E0E6","#EE6363"]
#color = ["#B0E0E6","#EE00EE"]

ax = plt.subplot(111)
for i in range(result.shape[0]):
    if(label[i] == 0):
        s1 = plt.scatter(result[i, 0], result[i, 1],s=20,color=color[label[i]])
for i in range(result.shape[0]):
    if(label[i] == 1):
        s2 = plt.scatter(result[i, 0], result[i, 1],s=20,color=color[label[i]])
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.title('t-SNE embedding of the 2nd MHA layer')
plt.legend((s1,s2),('0','1') ,loc = 'best')
plt.show()