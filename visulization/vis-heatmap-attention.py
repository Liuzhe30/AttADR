import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def rerange(x):
    maxnum = max(max(abs(x)))
    print(maxnum)
    for i in range(x.shape[0]):
        x[i] = x[i] / maxnum
    return x

def rerange2(x):
    maxnum = max(max(abs(x)))
    print(maxnum)
    for i in range(x.shape[0]):
        x[i] = x[i] / (maxnum * 2)
    return x
        

inputn = np.load("input.npy") # (500, 3246, 2)
GH0 = np.load("weights_global_attention0.npy") # (3246, 3246)
GH1 = np.load("weights_global_attention1.npy") # (3246)
att_out2 = np.load("att_out2.npy") # (500, 1623, 2) MHA1
att_out3 = np.load("att_out3.npy") # (500, 811, 2) MHA2
label = np.load("label_test500.npy") # (500,)
print(label.shape)

plt.figure(figsize=(35, 2))
x_ticks = np.linspace(0, 3246, 10)  
plt.xticks(x_ticks)  

'''
# global attention
GH1 = GH1.reshape((1, GH1.shape[0])) # (1, 3246)
GH1 = rerange(GH1)
sns.heatmap(GH1, cmap = 'bwr', center=0, vmin=-1, vmax=1)
plt.show()
'''

'''
att_out2 = np.concatenate((att_out2[:,:,0], att_out2[:,:,1]), axis=0) # (1000, 1623)
att_out2 = np.sum(att_out2, axis=0) # (1623,)
new_att2_list = []
for item in att_out2:
    new_att2_list.append(item)
    new_att2_list.append(item)
att_out2 = np.array(new_att2_list)
att_out2 = att_out2.reshape((1, att_out2.shape[0])) # (1, 3246)
att_out2 = rerange(att_out2)
sns.heatmap(att_out2, cmap = 'bwr', center=0, vmin=-1, vmax=1)
plt.show()
'''

att_out3 = np.concatenate((att_out3[:,:,0], att_out3[:,:,1]), axis=0) # (1000, 811)
#att_out3 = att_out3[:,:,1] # (1000, 811)
att_out3 = np.sum(att_out3, axis=0) # (811,)
new_att3_list = []
for item in att_out3:
    new_att3_list.append(item)
    new_att3_list.append(item)
new_att3_list.append(0)
att_out3 = np.array(new_att3_list) # (1, 1623)

new_att3_list = []
for item in att_out3:
    new_att3_list.append(item)
    new_att3_list.append(item)
att_out3 = np.array(new_att3_list)
att_out3 = att_out3.reshape((1, att_out3.shape[0])) # (1, 3246)
print(att_out3.shape)
att_out3 = rerange(att_out3)
sns.heatmap(att_out3, cmap = 'bwr', center=0.5, vmin=0, vmax=1)
plt.show()