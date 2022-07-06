# train model with generator

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
from utils import bigbird_utils
from utils import bigbird_attention
from utils.bigbird_attention import MultiHeadedAttentionLayer
from utils.Transformer import TokenAndPositionEmbedding

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

# set paramaters
class_num = 2  # type of ddis
N = 3037  # number of drugs
edge = 122999  # number of ddis
a = 167  # smile 
b = 2314  # target
c = 336  # enzyme
d = 398  # pathway
e = 27  # transporter
sum_feature = 3242 # dimension of 5 features
maxlen = 3250 # 3242 + 8
block_size = 50
num_heads = 4
embed_dim = 64
ff_dim = 256

class TransformerBlock(layers.Layer):
    def __init__(self, seq_length, block_size, size_per_head,  
                 embed_dim, num_heads, ff_dim, initializer_range=0.02,rate=0.1,use_bias=True):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadedAttentionLayer(attention_type='block_sparse',
                                             #attention_type='original_full',
                                             num_attention_heads=num_heads,
                                             size_per_head=size_per_head,
                                             num_rand_blocks=3,
                                             from_seq_length=seq_length,
                                             to_seq_length=seq_length,
                                             from_block_size=block_size,
                                             to_block_size=block_size,                                             
                                             name="self")
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(size_per_head*num_heads),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
        self.projection_layer = bigbird_utils.Dense3dProjLayer(
                      num_heads, size_per_head,
                      bigbird_utils.create_initializer(initializer_range), None,
                      "dense", use_bias)        

    def call(self, inputs, attention_mask=None, band_mask=None, from_mask=None, to_mask=None, input_blocked_mask=None, training=True): # transformer encoder
        # masks:[attention_mask, band_mask, from_mask, to_mask, input_blocked_mask]
        attn_output = self.att(inputs, inputs, [
            attention_mask, band_mask, from_mask, to_mask, input_blocked_mask, input_blocked_mask], training=training)
        attn_output = self.dropout1(attn_output, training=training)
        attn_output = self.projection_layer(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config().copy()
        config.update({
              'embed_dim': embed_dim,
              'num_heads': num_heads,
              'ff_dim': ff_dim,
        })
        return config

def generate_batch(feature_dict, pair_label, batch_size):
    while 1:
        for i in range(len(pair_label) - batch_size): # 122999
            dataX_batch, dataY_batch, weights = [], [], []  # (batch_size, 3242, 2) , (batch_size, 1)
            
            for j in range(i, i + batch_size):
                
                drugA = pair_label[j][0]
                drugB = pair_label[j][1]
                label = pair_label[j][2]
                
                drugA_feature = np.array(feature_dict[drugA] + [0, 0, 0, 0, 0, 0, 0, 0])
                drugB_feature = np.array(feature_dict[drugB] + [0, 0, 0, 0, 0, 0, 0, 0])
                
                dataX_batch.append(np.c_[drugA_feature, drugB_feature])
                dataY_batch.append(label)
                if(int(label) == 0):
                    weights.append(1)
                elif(int(label) == 1):
                    weights.append(3)
            
            i += batch_size
            x = np.array(dataX_batch)
            y = np.array(dataY_batch)
            mask = np.c_[np.ones(batch_size, sum_feature), np.zeros(batch_size, 8)]
            sample_weights = np.array(weights)

            yeild([x, mask], y, sample_weights)

def create_bbmask(ori_mask):
    # attention_mask:[batch_size,seq_length, seq_length]
    # from_mask:[batch_size, 1, seq_length, 1]
    # to_mask:[batch_size, 1, 1, seq_length]
    q_mask = tf.expand_dims(ori_mask, axis=-1)  # [seq_len, 1]
    k_mask = layers.Reshape((1, -1))(q_mask) # [1, seq_len] 
    attention_mask = tf.matmul(q_mask, k_mask)   # [seq_len, seq_len] 
    from_mask = ori_mask[:, tf.newaxis, :, tf.newaxis]
    print(from_mask.get_shape())
    to_mask = ori_mask[:, tf.newaxis, tf.newaxis, :]
    block_mask = layers.Reshape((maxlen//block_size, block_size))(from_mask)
    band_mask = bigbird_attention.create_band_mask_from_inputs(block_mask,block_mask)
    return [attention_mask, from_mask, to_mask, block_mask, band_mask]

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
        
# set args
parser = argparse.ArgumentParser(description="DDI stucture.")
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--lr', default=0.001, type=float,
                        help="Initial learning rate")
parser.add_argument('--lr_decay', default=0.05, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")            
parser.add_argument('--save_dir', default='../models/')
parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
parser.add_argument('-t', '--testing', action='store_true',
                        help="Test the trained model on testing dataset")  
args = parser.parse_args()
print(args)

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    
# -----prepare datasets-----
with open("../data/feature.json",'r') as load_f:
    feature_dict = json.load(load_f) # 3037 keys with each key a 3243-d list  
pair_label = np.load("../data/pair_label.npy") # (122999, 3) each ddi (drug ID1, drug ID2, label)

# ---------------------------------------build bigbird model--------------------------------------------------------
input1 = tf.keras.layers.Input(shape=(maxlen, 2), name = 'input-feature')
input2 = tf.keras.layers.Input(shape=(maxlen, ), name = 'input-mask')

[attention_mask, from_mask, to_mask, block_mask, band_mask] = create_bbmask(input2)
print('attention_mask.get_shape()', attention_mask.get_shape())


trans_block1 = TransformerBlock(maxlen, block_size, 
                                    input1.get_shape()[-1]//num_heads, 
                                    embed_dim, num_heads, ff_dim)

trans_layer1 = trans_block1(input1, 
                       attention_mask=attention_mask, 
                       band_mask=band_mask, 
                       from_mask=from_mask, 
                       to_mask=to_mask, 
                       input_blocked_mask=block_mask)
print('trans_layer1.get_shape()', trans_layer1.get_shape())
mut_bew = tf.keras.layers.Conv1D(4, 1, kernel_initializer='he_uniform')(trans_layer1)
mut_bew = tf.keras.layers.GlobalAveragePooling1D()(mut_bew)
output = tf.keras.layers.Dense(2, activation = 'softmax', name = 'output_softmax')(mut_bew)

model = tf.keras.models.Model(inputs=[input1, input2], outputs=output)
model.summary()



