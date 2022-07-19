import os
import numpy as np
from sklearn.utils import shuffle

drug_full_list = []
with open('../data/label_ddi/psy_positive_pairs.csv', 'r') as file:
    line = file.readline()
    line = file.readline()
    while line:
        if(line.split(',')[0] not in drug_full_list):
            drug_full_list.append(line.split(',')[0])
        if(line.split(',')[1] not in drug_full_list):
            drug_full_list.append(line.split(',')[1])
        line = file.readline()
with open('../data/label_ddi/negtive_pairs.csv', 'r') as file:
    line = file.readline()
    line = file.readline()
    while line:
        if(line.split(',')[0] not in drug_full_list):
            drug_full_list.append(line.split(',')[0])
        if(line.split(',')[1] not in drug_full_list):
            drug_full_list.append(line.split(',')[1])
        line = file.readline()

print(len(drug_full_list)) # 3037
drug_full = np.array(drug_full_list)
drug_full_shuffle = shuffle(drug_full)

psy_drug_list = []   
with open('../data/label_ddi/psychiatric.csv', 'r') as file:
    line = file.readline()
    line = file.readline()
    while line:
        if(line.split(',')[1] not in psy_drug_list):
            psy_drug_list.append(line.split(',')[1])
        line = file.readline()    
        
print(len(psy_drug_list)) # 130
psy_drug = np.array(psy_drug_list)
psy_drug_shuffle = shuffle(psy_drug)

dataX_list = []
dataY_list = []
for i in range(10):
    dataY_list.append(psy_drug_shuffle[i])
for i in range(100):
    dataY_list.append(drug_full_shuffle[i])
    
for item in drug_full_shuffle:
    if(item not in dataY_list):
        dataX_list.append(item)
for item in psy_drug_shuffle:
    if(item not in dataY_list):
        dataX_list.append(item)
        
print(len(dataX_list)) # 3046
print(len(dataY_list)) # 110

with open("../data/drugsetX.txt","w+") as file:
    for item in dataX_list:
        file.write(item + '\n')
with open("../data/drugsetY.txt","w+") as file:
    for item in dataY_list:
        file.write(item + '\n')

pair_task1 = []
pair_test_task2 = []
pair_test_task3 = []
pair_label = np.load("../data/pair_label.npy")
for item in pair_label:
    if(item[0] in dataY_list and item[1] in dataY_list):
        pair_test_task3.append(item)
    elif(item[0] in dataX_list and item[1] in dataY_list):
        pair_test_task2.append(item)
    elif(item[0] in dataY_list and item[1] in dataX_list):
        pair_test_task2.append(item)   
    elif(item[0] in dataX_list and item[1] in dataX_list):
        pair_task1.append(item)    
        
pair_task1_array = np.array(pair_task1)
pair_test_task2_array = np.array(pair_test_task2)
pair_test_task3_array = np.array(pair_test_task3)
print(pair_task1_array.shape)# (332862, 3)
print(pair_test_task2_array.shape)# (46388, 3)
print(pair_test_task3_array.shape)# (1230, 3)

pair_task1_array_shuffle = shuffle(pair_task1_array)
np.save("../data/pair_label_task1.npy", pair_task1_array_shuffle)
np.save("../data/pair_label_test_task2.npy", pair_test_task2_array)
np.save("../data/pair_label_test_task3.npy", pair_test_task3_array)