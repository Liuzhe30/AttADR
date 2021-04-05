import os
import numpy as np

# set paramaters
N = 50
a = 881
b = 1162
c = 202
d = 957
e = 500

# drug profile
drug_feature = []
for idx in range(0, N):
    drug_feature_vec = np.random.randint(2, high = None, size = a + b + c + d + e, dtype = 'l')
    drug_feature.append(drug_feature_vec)
drug_profile = np.array(drug_feature)
#print(drug_profile.shape)
np.savetxt('data/drug_profile.npy', drug_profile, fmt = "%d")

# drug similarity
substructure_mat = np.random.uniform(0, 1, (N, N))
for i in range(0, N):
    substructure_mat[i][i] = 1

target_mat = np.random.uniform(0, 1, (N, N))
for i in range(0, N):
    target_mat[i][i] = 1

enzyme_mat = np.random.uniform(0, 1, (N, N))
for i in range(0, N):
    enzyme_mat[i][i] = 1

pathway_mat = np.random.uniform(0, 1, (N, N))
for i in range(0, N):
    pathway_mat[i][i] = 1

transporter_mat = np.random.uniform(0, 1, (N, N))
for i in range(0, N):
    transporter_mat[i][i] = 1
    
similarity_mat = np.r_[np.r_[np.r_[np.r_[substructure_mat, target_mat], enzyme_mat], pathway_mat], transporter_mat]
#print(similarity_mat.shape)
np.savetxt('data/similarity_mat.npy', similarity_mat)
