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
        

inputn = np.load("input.npy") # (500, 929, 2)
GH0 = np.load("weights_global_attention0.npy") # (929, 929)
GH1 = np.load("weights_global_attention1.npy") # (929)
att_out2 = np.load("att_out2.npy") # (500, 929, 2) MHA1
att_out3 = np.load("att_out3.npy") # (500, 929, 2) MHA2
label = np.load("label_test500.npy") # (500,)
print(label.shape)

plt.figure(figsize=(35, 2))  


# global attention
GH1 = GH1.reshape((1, GH1.shape[0])) # (1, 929)
GH1 = rerange(GH1)
sns.heatmap(GH1, cmap = 'bwr', center=0, vmin=-1, vmax=1)
plt.show()


'''
att_out2 = np.concatenate((att_out2[:,:,0], att_out2[:,:,1]), axis=0) # (1000, 929)
att_out2 = np.sum(att_out2, axis=0) # (929,)
att_out2 = att_out2.reshape((1, att_out2.shape[0])) # (1, 929)
att_out2 = rerange(att_out2)
np.save('MHA1-weights-out.npy',att_out2)
sns.heatmap(att_out2, cmap = 'bwr', center=0.45, vmin=0, vmax=1.5)
plt.show()
'''
'''
att_out3 = np.concatenate((att_out3[:,:,0], att_out3[:,:,1]), axis=0) # (1000, 929)
#att_out3 = att_out3[:,:,1] # (1000, 929)
att_out3 = np.sum(att_out3, axis=0) # (929,)
att_out3 = att_out3.reshape((1, att_out3.shape[0])) # (1, 929)
print(att_out3.shape)
att_out3 = rerange(att_out3)
np.save('MHA2-weights-out.npy',att_out3)
sns.heatmap(att_out3, cmap = 'bwr', center=-0.93, vmin=-1, vmax=-0.6)
plt.show()
'''