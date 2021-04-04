import os
import numpy as np

# set paramaters
N = 500
a = 881
b = 1162
c = 202
d = 957
e = 500

# drug profile
substructure_vec = np.zeros((2, a), int)
for i in range(0, 2):
    for j in range(0, a):
        substructure_vec[i][j] = np.random.randint(2, high = None, size= 1, dtype = 'l')
np.savetxt('data/substructure_vec.npy', substructure_vec, fmt = "%d")

target_vec = np.zeros((2, b), int)
for i in range(0, 2):
    for j in range(0, b):
        target_vec[i][j] = np.random.randint(2, high = None, size = 1, dtype = 'l')
np.savetxt('data/target_vec.npy', target_vec, fmt = "%d")

enzyme_vec = np.zeros((2, c), int)
for i in range(0, 2):
    for j in range(0, c):
        enzyme_vec[i][j] = np.random.randint(2, high = None, size = 1, dtype = 'l')
np.savetxt('data/enzyme_vec.npy', enzyme_vec, fmt = "%d")

pathway_vec = np.zeros((2, d), int)
for i in range(0, 2):
    for j in range(0, d):
        pathway_vec[i][j] = np.random.randint(2, high = None, size = 1, dtype = 'l')
np.savetxt('data/pathway_vec.npy', pathway_vec, fmt = "%d")

transporter_vec = np.zeros((2, e), int)
for i in range(0, 2):
    for j in range(0, e):
        transporter_vec[i][j] = np.random.randint(2, high = None, size = 1, dtype = 'l')
np.savetxt('data/transporter_vec.npy', transporter_vec, fmt = "%d")

# drug similarity
substructure_mat = np.random.uniform(0, 1, (N, N))
for i in range(0, N):
    substructure_mat[i][i] = 1
np.savetxt('data/substructure_mat.npy', substructure_mat)

target_mat = np.random.uniform(0, 1, (N, N))
for i in range(0, N):
    target_mat[i][i] = 1
np.savetxt('data/target_mat.npy', target_mat)

pathway_mat = np.random.uniform(0, 1, (N, N))
for i in range(0, N):
    pathway_mat[i][i] = 1
np.savetxt('data/pathway_mat.npy', pathway_mat)

transporter_mat = np.random.uniform(0, 1, (N, N))
for i in range(0, N):
    transporter_mat[i][i] = 1
np.savetxt('data/transporter_mat.npy', transporter_mat)