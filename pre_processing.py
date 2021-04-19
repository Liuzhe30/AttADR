import numpy as np
import csv
np.set_printoptions(threshold=np.inf)
#np.genfromtxt('col.txt',dtype='str')

smile_npy = np.load('data/feature_numpy/smile.npy')
print(smile_npy.shape)
target_npy = np.load('data/feature_numpy/target.npy')
print(target_npy.shape)
enzyme_npy = np.load('data/feature_numpy/enzyme.npy')
print(enzyme_npy.shape)
pathway_npy = np.load('data/feature_numpy/pathway.npy')
print(pathway_npy.shape)
transporter_npy = np.load('data/feature_numpy/transporter.npy')
print(transporter_npy.shape)

a = 167  # smile 
b = 2314  # target
c = 336  # enzyme
d = 398  # pathway
e = 27  # transporter

# feature concatenation
generated = np.c_[np.c_[np.c_[np.c_[smile_npy, target_npy], enzyme_npy], pathway_npy], transporter_npy]
print(generated.shape)
drug_list = np.genfromtxt('data/feature_list/TotalDrugID.txt',dtype='str')
feature = np.c_[generated, drug_list]
np.save('data/feature.npy', feature)
print(feature.shape)

# label recreate
label_list = []
idx = 0
with open('data/label_ddi/ddi_label.csv', 'r') as file:
    line = file.readline()
    line = file.readline()
    while line:
        label_list.append([line.split(',')[3], line.split(',')[4], line.split(',')[0]])
        #print([line.split(',')[3], line.split(',')[4], line.split(',')[0]])
        #idx += 1
        #print(idx)
        line = file.readline()
label = np.array(label_list)
np.save('data/label.npy', label)
print(label.shape)
    
