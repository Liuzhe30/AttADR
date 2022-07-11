import numpy as np
import  pandas as pd
pd.set_option('display.max_columns', None)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow import keras
import argparse
import copy
import json
from sklearn.utils import shuffle
from utils.Transformer import MultiHeadSelfAttention
from utils.Transformer import TransformerBlock
from utils.Transformer import TokenAndPositionEmbedding
from keras.utils.np_utils import *

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import argparse
import copy
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
tf.compat.v1.disable_eager_execution()

# set paramaters
class_num = 2  # type of ddis
N = 3037  # number of drugs
edge = 380480  # number of ddis
a = 167  # smile 
b = 2314  # target
c = 336  # enzyme
d = 398  # pathway
e = 27  # transporter
sum_feature = 3242 # dimension of 5 features
maxlen = 3250 # 3242 + 8

# -----set transformer parameters-----
vocab_size = 5
embed_dim = 64
num_heads = 4
ff_dim = 64
pos_embed_dim = 64
seq_embed_dim_bet = 62

def generate_valid_test(feature_dict, psy_list, pair_label):

    dataX_batch, dataY_batch = [], []
    for i in range(len(pair_label)):
        drugA = pair_label[i][0]
        drugB = pair_label[i][1]
        label = int(pair_label[i][2].strip())
        
        # check psydrug 0/1
        if(drugA in psy_list):
            flag_A = 1
        else:
            flag_A = 0
        if(drugB in psy_list):
            flag_B = 1
        else:
            flag_B = 0
        
        drugA_feature = np.array(feature_dict[drugA] + [flag_A] + [0, 0, 0, 0, 0, 0, 0])
        drugB_feature = np.array(feature_dict[drugB] + [flag_B] + [0, 0, 0, 0, 0, 0, 0])
        
        dataX_batch.append(np.c_[drugA_feature, drugB_feature])
        dataY_batch.append(to_categorical(label,2))

    x = np.array(dataX_batch)
    y = np.array(dataY_batch)
    mask = np.c_[np.ones((len(pair_label), sum_feature + 1)), np.zeros((len(pair_label), 7))]
    #print(mask.shape)

    return [x, mask], y


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    print(seq.shape)
    # add extra dimensions to add the padding
    # to the attention logits.
    return  seq[:, tf.newaxis, tf.newaxis, :]# (batch_size, 1, 1, seq_len)

class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        val_targ = self.validation_data[1]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_acc = accuracy_score(val_targ, val_predict)
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')
        
        logs['val_acc'] = _val_acc
        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" - val_acc: %f - val_f1: %f - val_precision: %f - val_recall: %f" % (_val_acc, _val_f1, _val_precision, _val_recall))
        #print(" - val_f1: %f - val_precision: %f - val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return
    
# -----prepare datasets-----
with open("../data/feature.json",'r') as load_f:
    feature_dict = json.load(load_f) # 3037 keys with each key a 3243-d list 

# psydrug list
psy_list = []
with open('../data/label_ddi/psychiatric.csv', 'r') as file:
    line = file.readline()
    line = file.readline()
    while line:
        psy_list.append(line.split(',')[1])
        line = file.readline()

pair_label = np.load("../data/pair_label.npy") # (380480, 3) each ddi (drug ID1, drug ID2, label)
valid_set = pair_label[edge-1000:edge-500, :] # 500
test_set = pair_label[edge-500:edge, :] # 500
valid_data = generate_valid_test(feature_dict, psy_list, valid_set)
test_data = generate_valid_test(feature_dict, psy_list, test_set)
#print(valid_data)
#print(test_data)

[x_test, mask_test], y_test = test_data

# ---------------------------------------build bigbird model--------------------------------------------------------
input1 = tf.keras.layers.Input(shape=(maxlen, 2), name = 'input-feature')
input2 = tf.keras.layers.Input(shape=(maxlen, ), name = 'input-mask')
mask = create_padding_mask(input2)

embedding_layer_bet = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, pos_embed_dim, seq_embed_dim_bet)
trans_block_bet1 = TransformerBlock(embed_dim, num_heads, ff_dim)
trans_block_bet2 = TransformerBlock(embed_dim, num_heads, ff_dim)
trans_block_bet3 = TransformerBlock(embed_dim, num_heads, ff_dim)

bet = embedding_layer_bet([input2,input1])
print('embedding_layer_bet.get_shape()', bet.get_shape()) # 

bet = trans_block_bet1(bet, mask)
bet = trans_block_bet2(bet, mask)
#bet = trans_block_bet3(bet, mask)

print('trans_layer1.get_shape()', bet.get_shape()) 
bet = tf.keras.layers.Conv1D(32 ,4, kernel_initializer='he_uniform')(bet)
bet = tf.keras.layers.Conv1D(4, 1, kernel_initializer='he_uniform')(bet)
bet = tf.keras.layers.GlobalAveragePooling1D()(bet)
output = tf.keras.layers.Dense(2, activation = 'softmax', name = 'output_softmax')(bet)

model = tf.keras.models.Model(inputs=[input1, input2], outputs=output)
model.load_weights("../models/weights-01.h5")
model.summary()

pred = model.predict({"input-feature":x_test, "input-mask":mask_test}, batch_size = 16)
print(pred)
