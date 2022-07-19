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

from tensorflow.keras import backend as K
K.set_learning_phase(1)
is_training = K.learning_phase()

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

def generate_batch(feature_dict, psy_list, pair_label, batch_size):
    while 1:
        for i in range(len(pair_label) - batch_size): # 380480
            dataX_batch, dataY_batch, weights = [], [], []  # (batch_size, 3242, 2) , (batch_size, 2)
            
            for j in range(i, i + batch_size):
                
                drugA = pair_label[j][0]
                drugB = pair_label[j][1]
                label = int(pair_label[j][2].strip())
                
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
                #dataY_batch.append(to_categorical(label,2))
                dataY_batch.append(label)
            
            i += batch_size
            x = np.array(dataX_batch)
            y = np.array(dataY_batch)
            mask = np.c_[np.ones((batch_size, sum_feature + 1)), np.zeros((batch_size, 7))]
            sample_weights = np.array(weights)
            #print(mask.shape) #(batch_size, 3250)
            #print(x.shape) # (batch_size, 2)
            #print(y.shape) # (batch_size, 2)

            #yield([x, mask], y, sample_weights)
            yield([x, mask], y)

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
        #dataY_batch.append(to_categorical(label,2))
        dataY_batch.append(label)

    x = np.array(dataX_batch)
    y = np.array(dataY_batch)
    mask = np.c_[np.ones((len(pair_label), sum_feature + 1)), np.zeros((len(pair_label), 7))]

    return [x, mask], y


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    print(seq.shape)
    # add extra dimensions to add the padding
    # to the attention logits.
    return  seq[:, tf.newaxis, tf.newaxis, :]# (batch_size, 1, 1, seq_len)
        
# set args
parser = argparse.ArgumentParser(description="DDI stucture.")
parser.add_argument('--epochs', default=15, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--lr', default=0.002, type=float,
                        help="Initial learning rate")
parser.add_argument('--lr_decay', default=0.05, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs")            
parser.add_argument('--save_dir', default='../models/newstru-classweight-1-2-lr0.002/')
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

# psydrug list
psy_list = []
with open('../data/label_ddi/psychiatric.csv', 'r') as file:
    line = file.readline()
    line = file.readline()
    while line:
        psy_list.append(line.split(',')[1])
        line = file.readline()

pair_label = np.load("../data/pair_label.npy") # (380480, 3) each ddi (drug ID1, drug ID2, label)
train_set = pair_label[0:edge-1000, :]

# data augmentation
train_set_verse = train_set[:, [1, 0, 2]]
train_set_aug = np.vstack((train_set, train_set_verse))

valid_set = pair_label[edge-1000:edge-500, :] # 500
test_set = pair_label[edge-500:edge, :] # 500
valid_data = generate_valid_test(feature_dict, psy_list, valid_set)
test_data = generate_valid_test(feature_dict, psy_list, test_set)
#print(valid_data)
#print(test_data)

# ---------------------------------------build bigbird model--------------------------------------------------------
input1 = tf.keras.layers.Input(shape=(maxlen, 2), name = 'input-feature')
input2 = tf.keras.layers.Input(shape=(maxlen, ), name = 'input-mask')
mask = create_padding_mask(input2)

embedding_layer_bet = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, pos_embed_dim, seq_embed_dim_bet)
trans_block_bet1 = TransformerBlock(embed_dim, num_heads, ff_dim)

# self-attention 1
bet = embedding_layer_bet([input2,input1])
print('embedding_layer_bet.get_shape()', bet.get_shape()) # (3250, 64)

bet = trans_block_bet1(bet, mask)
print('trans_layer1.get_shape()', bet.get_shape()) # (3250, 64)
bet = tf.keras.layers.BatchNormalization()(bet, training = True)

# cnn block 1
bet = tf.keras.layers.Conv1D(64, 3, kernel_initializer='he_uniform', activation='relu')(bet)
bet = tf.keras.layers.BatchNormalization()(bet, training = True)
bet = tf.keras.layers.Conv1D(128, 5, kernel_initializer='he_uniform', activation='relu')(bet)
bet = tf.keras.layers.BatchNormalization()(bet, training = True)
bet = tf.keras.layers.MaxPooling1D(2, strides=2)(bet)

# self-attention 2
attention2 = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=16)
bet = attention2(bet, bet)

# cnn block 2
bet = tf.keras.layers.Conv1D(4, 1, kernel_initializer='he_uniform')(bet)
bet = tf.keras.layers.GlobalAveragePooling1D()(bet)
#output = tf.keras.layers.Dense(2, activation = 'softmax', name = 'output_softmax')(bet)
output = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'output_softmax')(bet)

model = tf.keras.models.Model(inputs=[input1, input2], outputs=output)
model.summary()

# compile and fit
# callbacks
log = tf.keras.callbacks.CSVLogger(args.save_dir + '/log.csv')
#tensorboard = tf.keras.callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs', histogram_freq=int(args.debug))
#EarlyStopping = callbacks.EarlyStopping(monitor='val_cc2', min_delta=0.01, patience=5, verbose=0, mode='max', baseline=None, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', 
                                        monitor='val_acc', 
                                        mode='max', 
                                        #val_categorical_accuracy val_acc
                                        save_best_only=True, 
                                        save_weights_only=True, verbose=1)        
lr_decay = tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

class_weights = {0:1, 1:2}
# Train the model and save it
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

history = model.fit(generate_batch(feature_dict, psy_list, train_set, args.batch_size), # Tf2 new feature
          steps_per_epoch = len(train_set)/args.batch_size,
          epochs = args.epochs, verbose=1,
          validation_data = generate_valid_test(feature_dict, psy_list, valid_set),
          validation_steps = len(valid_set),
          #callbacks = [log, tensorboard, checkpoint, lr_decay],
          callbacks = [log, checkpoint, lr_decay],
          shuffle = True,
          #batch_size=args.batch_size,
          workers = 1,
          class_weight = class_weights).history

model.save_weights(args.save_dir + '/trained_weights.h5')
model.save(args.save_dir + '/trained_model.h5')
print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)



