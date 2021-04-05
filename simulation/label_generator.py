import os
import numpy as np

# set paramaters
N = 50
edge = 200
a = 881
b = 1162
c = 202
d = 957
e = 500

label_list = []
for i in range(0, edge + 20):
    pair = np.random.randint(N, high = None, size = 2, dtype = 'l')
    pair.sort()
    label_list.append(pair)

corr = np.array(list(set([tuple(t) for t in label_list])))
print(corr.shape)
cla = np.random.randint(4, high = None, size = (edge, 1), dtype = 'l')

label_mat = np.c_[corr, cla]
np.savetxt('data/label_mat.npy', label_mat)
