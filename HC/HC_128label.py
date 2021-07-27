#! /usr/bin/env python
# -*- coding:utf-8 -*-
# SciPy Hierarchical Clusterin by Zhe Liu on 7/21/2021

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
import sys   
sys.setrecursionlimit(123000)
import numpy as np
#np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=5, suppress=True)
np.random.seed(1029)

# set paramaters
class_num = 128  # type of ddis
N = 3037  # number of drugs
edge = 122999  # number of ddis
a = 167  # smile 
b = 2314  # target
c = 336  # enzyme
d = 398  # pathway
e = 27  # transporter

# more beautiful tree: showing the distance between clusters makes the tree graph more intuitive
def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('sample index or (cluster size)')
        plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5), textcoords='offset points', va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata


'''
X = np.array([[1,3,5,2,[0,1,0,0]],
              [3,7,8,3,[0,1,0,0]],
              [6,4,3,3,[0,1,0,0]],
              [4,6,2,4,[0,0,1,0]],
              [5,7,3,4,[0,0,1,0]],
              [8,6,5,4,[0,0,0,1]]])
              '''
'''
X = np.array([[1,3,0,1,2,1],
              [3,7,0,0,3,1],
              [6,4,1,0,3,1],
              [4,6,0,0,4,2],
              [5,7,0,0,4,2],
              [8,6,0,0,4,0]])
              '''

def data_generator():
    drug_profile = np.load('../data/feature.npy') # N * (a + b + c + d + e)
    #similarity_mat = np.loadtxt('data/similarity_mat.npy') # 5N * N
    y = np.load('../data/label.npy')  # edge * 3
    drug_id_dict = {}
    for i in range(0, len(drug_profile)):
        drug_id_dict[drug_profile[i][-1]] = i
    x_slide = []
    for i in range(0, edge):
    #for i in range(0, 1000):
        print(i)
        con_temp = np.r_[drug_profile[drug_id_dict[y[i][0]]][:-1], drug_profile[drug_id_dict[y[i][1]]][:-1]].transpose()
        con = np.r_[con_temp, np.array(y[i][2])]
        cony = con.astype(np.float)
        new = cony.astype(np.int)
        #print(new)
        x_slide.append(new)
    X = np.array(x_slide)
    
    return X

X = data_generator()
print(X)

#con = np.concatenate(feature, label)
#print(con[0])

#plt.figure(figsize=(25, 10))
#plt.scatter(X[:, 0], X[:, 1])
#plt.show()

def mydist(p1, p2):
    if(p1[-1] == p2[-1]):
        diff = 0
    else:
        diff = p1[:-1] - p2[:-1]
    return np.vdot(diff, diff) ** 0.5

#Z = linkage(X, 'ward')
Z = linkage(X, method='single', metric=mydist, optimal_ordering=False)
print(Z)
np.savetxt("HC_process.txt", Z, fmt='%d %d %.4f %d')
print('linkage finished!')

c, coph_dists = cophenet(Z, pdist(X))
print(c)

'''
# Scale down the tree: dendrogram
plt.figure(figsize=(50, 10))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(Z, leaf_rotation=90., leaf_font_size=8.)
#plt.show()
plt.savefig('Hierarchical Clustering Dendrogram.png')

plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=90., leaf_font_size=12., show_contracted=True)
#plt.show()
plt.savefig('Hierarchical Clustering Dendrogram (truncated).png')

fancy_dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=90., leaf_font_size=12., show_contracted=True, annotate_above=10)
#plt.show()
plt.savefig('Hierarchical Clustering Dendrogram (truncated).png')

# Select the critical distance to determine the number of clusters
max_d = 7
fancy_dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=90., leaf_font_size=12., show_contracted=True, annotate_above=10, max_d=max_d)
#plt.show()
plt.savefig('Hierarchical Clustering Dendrogram (fancy d = 7).png')

max_d = 6
fancy_dendrogram(Z, truncate_mode='lastp', p=12, leaf_rotation=90., leaf_font_size=12., show_contracted=True, annotate_above=10, max_d=max_d)
#plt.show()
plt.savefig('Hierarchical Clustering Dendrogram (fancy d = 6.png')
'''

# We can use the FCLUSTER equation to get the cluster information:
# If we already know the maximum threshold from the tree, 
# we can get the cluster subscript for each experimental sample by using the following code  
max_d = 6
clusters = fcluster(Z, max_d, criterion='distance')
print(clusters)

'''
# If we already know that we will end up with two clusters, we can get the cluster index like this:  
k = 2
fcluster(Z, k, criterion='maxclust')
'''

'''
# If the number of features in your experimental sample is small, you can visualize your cluster results: 
# have more samples: using T-SNE
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='prism')
plt.savefig('cluster results.png')
#plt.show()
'''