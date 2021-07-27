import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
#import palettable

#! /usr/bin/env python
# -*- coding:utf-8 -*-
# SciPy Hierarchical Clusterin by Zhe Liu on 7/21/2022 (feature avg)

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

X_avg = np.loadtxt("HC_data_avg.txt")
print(X_avg.shape)
X_sort = X_avg[np.argsort(X_avg[:,-1])]
X_cut = X_sort[:,:-1]
print(X_cut)
print(X_cut.shape)

#plt.figure(dpi=300)
#sns.heatmap(X_cut)
#plt.show()

sns.clustermap(data=X_cut,
               method ='ward',
               cmap="mako",
               col_cluster=False,
               metric='euclidean'
              )
#plt.show()
plt.savefig('HC_avg_clustermap.png')



