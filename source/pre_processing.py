import numpy as np
import csv
np.set_printoptions(threshold=np.inf)
import json
from sklearn.utils import shuffle
#np.genfromtxt('col.txt',dtype='str')

smile_npy = np.load('../data/feature_numpy/smile.npy')
print(smile_npy.shape)
target_npy = np.load('../data/feature_numpy/target.npy')
print(target_npy.shape)
enzyme_npy = np.load('../data/feature_numpy/enzyme.npy')
print(enzyme_npy.shape)
pathway_npy = np.load('../data/feature_numpy/pathway.npy')
print(pathway_npy.shape)
transporter_npy = np.load('../data/feature_numpy/transporter.npy')
print(transporter_npy.shape)

a = 167  # smile 
b = 2314  # target
c = 336  # enzyme
d = 398  # pathway
e = 27  # transporter

# feature concatenation
feature_dict = {}
generated = np.c_[np.c_[np.c_[np.c_[smile_npy, target_npy], enzyme_npy], pathway_npy], transporter_npy]
print(generated.shape)
drug_list = np.genfromtxt('../data/feature_list/TotalDrugID.txt',dtype='str')
feature = np.c_[generated, drug_list]
#np.save('../data/feature.npy', feature)
print(feature.shape) # (3037, 3243)
for i in range(feature.shape[0]):
    feature_dict[feature[i][-1]] = feature[i][0:-1].tolist()
#print(feature_dict)
json_str = json.dumps(feature_dict)
with open('../data/feature.json', 'w') as json_file:
    json_file.write(json_str)


# label recreate
label_list = []
idx = 0
with open('../data/label_ddi/psy_positive_pairs.csv', 'r') as file:
    line = file.readline()
    line = file.readline()
    while line:
        label_list.append([line.split(',')[0], line.split(',')[1], line.split(',')[-1]])
        line = file.readline()
with open('../data/label_ddi/negtive_pairs.csv', 'r') as file:
    line = file.readline()
    line = file.readline()
    while line:
        label_list.append([line.split(',')[0], line.split(',')[1], line.split(',')[-1]])
        line = file.readline()        
        
label = np.array(label_list)
label_shuffle = shuffle(label)
label_shuffle2 = shuffle(label_shuffle)
np.save('../data/pair_label.npy', label_shuffle2)
print(label.shape) # (380480, 3)
    
