#! /usr/bin/env python
# -*- coding:utf-8 -*-

import tensorflow as tf
import keras
from keras import layers, models, optimizers
from keras.utils import to_categorical
from keras.layers import *
from keras.models import *
from keras.callbacks import Callback
from keras.initializers import Constant
from keras import backend as K
K.clear_session()
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from keras import callbacks
from keras import backend as K 
from keras.utils.np_utils import to_categorical
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from collections import Counter
K.set_image_data_format('channels_last')
np.random.seed(0)

# set paramaters
class_num = 128  # type of ddis
N = 3037  # number of drugs
edge = 122999  # number of ddis
a = 167  # smile 
b = 2314  # target
c = 336  # enzyme
d = 398  # pathway
e = 27  # transporter

def stack_layer(AE1, AE2, AE3, AE4, AE5):
    return Lambda(K.stack)([AE1, AE2, AE3, AE4, AE5])

def xdata_generator():
    drug_profile = np.load('data/feature.npy') # N * (a + b + c + d + e)
    #similarity_mat = np.loadtxt('data/similarity_mat.npy') # 5N * N
    y = np.load('data/label.npy')  # edge * 3
    drug_id_dict = {}
    for i in range(0, len(drug_profile)):
        drug_id_dict[drug_profile[i][-1]] = i
    x_slide = []
    for i in range(0, edge):
        #print(np.r_[drug_profile[int(y[i][0])], drug_profile[int(y[i][1])]])
        x_slide.append(np.c_[drug_profile[drug_id_dict[y[i][0]]], drug_profile[drug_id_dict[y[i][1]]]].transpose())
    x = np.array(x_slide)
    return x
   
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="attddi stucture.")
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.05, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")            
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    parser.add_argument('--save_dir', default='attddi_train')
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")    
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    # load data
    x = xdata_generator()
    print("x shape: " + str(x.shape)) # edge * 2 * (a + b + c + d + e) ,  (200, 2, 3702)
    y = np.load('data/label.npy')[:,2]
    y_onehot = to_categorical(y, class_num)
    print("y_onehot shape: " + str(y_onehot.shape)) # edge * 4 , (200, 4)
    
    # split dataset
    x_train = x[0:120000, :, :]
    x_valid = x[120000:121499, :, :]
    x_test = x[121499:1222999, :, :]
    y_train = y_onehot[0:120000, :]
    y_valid = y_onehot[120000:121499, :]
    y_test = y_onehot[121499:1222999, :]
    
    # build model
    ## drug profile
    input1 = Input(shape=(2, a),name = 'drug_profile_a')
    input2 = Input(shape=(2, b),name = 'drug_profile_b')
    input3 = Input(shape=(2, c),name = 'drug_profile_c')
    input4 = Input(shape=(2, d),name = 'drug_profile_d')
    input5 = Input(shape=(2, e),name = 'drug_profile_e')
    
    AE1 = Flatten()(input1)
    AE1 = Dense(N, activation='relu', name = 'AE_1')(AE1)
    AE2 = Flatten()(input2)
    AE2 = Dense(N, activation='relu', name = 'AE_2')(AE2)    
    AE3 = Flatten()(input3)
    AE3 = Dense(N, activation='relu', name = 'AE_3')(AE3)
    AE4 = Flatten()(input4)
    AE4 = Dense(N, activation='relu', name = 'AE_4')(AE4)
    AE5 = Flatten()(input5)
    AE5 = Dense(N, activation='relu', name = 'AE_5')(AE5)    
    print('AE1.get_shape()', AE1.get_shape())
    
    stack = Lambda(lambda x: K.stack(x, axis=1) )
    profile_merged = stack([AE1, AE2, AE3, AE4, AE5])
    print('profile_merged.get_shape()', profile_merged.get_shape()) # 5 * N  ,(?, 5, 100)
    
    ## drug similarity
    input6 = Input(shape=(N, N),name = 'drug_similarity_a')
    input7 = Input(shape=(N, N),name = 'drug_similarity_b')
    input8 = Input(shape=(N, N),name = 'drug_similarity_c')
    input9 = Input(shape=(N, N),name = 'drug_similarity_d')
    input10 = Input(shape=(N, N),name = 'drug_similarity_e')
    
    similarity_merged = concatenate([input6, input7, input8, input9, input10], axis=1)
    print('similarity_merged.get_shape()', similarity_merged.get_shape())  # 5N * N , (?, 500, 100)
    
    final_merged = concatenate([profile_merged, similarity_merged], axis=1) 
    print('final_merged.get_shape()', final_merged.get_shape())  # 5(N + 1) * N  , (?, 505, 100)

    final_merged_trans = Permute((2, 1))(final_merged)
    print('final_merged_trans.get_shape()', final_merged_trans.get_shape())  # N * 5(N + 1) , (?, 100, 505)
    
    # MultiHeadAttention
    from keras_multi_head import MultiHeadAttention  # pip install keras_multi_head
    att1 = MultiHeadAttention(head_num = N, name = 'Multi-Head-1',)(final_merged)
    print('att1.get_shape()', att1.get_shape())
    att2 = MultiHeadAttention(head_num= 5 * (N + 1), name = 'Multi-Head-2',)(final_merged_trans)
    print('att2.get_shape()', att2.get_shape())
    
    # Output
    att2 = Permute((2, 1))(att2)
    print('att2.get_shape()', att2.get_shape())
    att_merged = concatenate([att1, att2], axis=0)
    print('att_merged.get_shape()', att_merged.get_shape())
    gap = GlobalAveragePooling1D()(att_merged)
    outputs = Dense(class_num, activation='softmax', name = 'softmax_output')(gap)
    print('outputs.get_shape()', outputs.get_shape())
    
    # register model
    model = Model(inputs=[input1, input2, input3, input4, input5, input6, input7, input8, input9, input10], outputs=outputs)  
    model.summary()