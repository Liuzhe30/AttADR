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

# -----set transformer parameters-----
vocab_size = 5
embed_dim = 64
num_heads = 4
ff_dim = 64
pos_embed_dim = 64
seq_embed_dim_bet = 62

def generate_batch(feature_dict, pair_label, batch_size):
    while 1:
        for i in range(len(pair_label) - batch_size): # 122999
            dataX_batch, dataY_batch, weights = [], [], []  # (batch_size, 3242, 2) , (batch_size, 2)
            
            for j in range(i, i + batch_size):
                
                drugA = pair_label[j][0]
                drugB = pair_label[j][1]
                label = pair_label[j][2]
                
                drugA_feature = np.array(feature_dict[drugA] + [0, 0, 0, 0, 0, 0, 0, 0])
                drugB_feature = np.array(feature_dict[drugB] + [0, 0, 0, 0, 0, 0, 0, 0])
                
                dataX_batch.append(np.c_[drugA_feature, drugB_feature])
                dataY_batch.append(to_categorical(label,2))
                if(int(label) == 0):
                    weights.append(1)
                elif(int(label) == 1):
                    weights.append(3)
            
            i += batch_size
            x = np.array(dataX_batch)
            y = np.array(dataY_batch)
            mask = np.c_[np.ones((batch_size, sum_feature)), np.zeros((batch_size, 8))]
            sample_weights = np.array(weights)
            #print(mask.shape) #(batch_size, 3250)
            #print(x.shape) # (batch_size, 2)
            #print(y.shape) # (batch_size, 2)

            yield([x, mask], y, sample_weights)

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
        
# set args
parser = argparse.ArgumentParser(description="DDI stucture.")
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--batch_size', default=32, type=int)
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
train_set = pair_label[0:120000, :]
valid_set = pair_label[120000:121499, :]
test_set = pair_label[121499:122999, :]

# ---------------------------------------build bigbird model--------------------------------------------------------
input1 = tf.keras.layers.Input(shape=(maxlen, 2), name = 'input-feature')
input2 = tf.keras.layers.Input(shape=(maxlen, ), name = 'input-mask')
mask = create_padding_mask(input2)

embedding_layer_bet = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, pos_embed_dim, seq_embed_dim_bet)
trans_block_bet1 = TransformerBlock(embed_dim, num_heads, ff_dim)

bet = embedding_layer_bet([input2,input1])
print('embedding_layer_bet.get_shape()', bet.get_shape()) # 

trans_layer1 = trans_block_bet1(bet, mask)


print('trans_layer1.get_shape()', trans_layer1.get_shape()) 
mut_bew = tf.keras.layers.Conv1D(4, 1, kernel_initializer='he_uniform')(trans_layer1)
mut_bew = tf.keras.layers.GlobalAveragePooling1D()(mut_bew)
output = tf.keras.layers.Dense(2, activation = 'softmax', name = 'output_softmax')(mut_bew)

model = tf.keras.models.Model(inputs=[input1, input2], outputs=output)
model.summary()


# compile and fit
# callbacks
log = tf.keras.callbacks.CSVLogger(args.save_dir + '/log.csv')
#tensorboard = tf.keras.callbacks.TensorBoard(log_dir=args.save_dir + '/tensorboard-logs', histogram_freq=int(args.debug))
#EarlyStopping = callbacks.EarlyStopping(monitor='val_cc2', min_delta=0.01, patience=5, verbose=0, mode='max', baseline=None, restore_best_weights=True)
checkpoint = tf.keras.callbacks.ModelCheckpoint(args.save_dir + '/weights-{epoch:02d}.h5', monitor='val_acc', mode='max', #val_categorical_accuracy val_acc
                                       save_best_only=True, save_weights_only=True, verbose=1)        
lr_decay = tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

# Train the model and save it
model.compile(loss='mse', optimizer='adam', metrics=['acc'])

history = model.fit(generate_batch(feature_dict, train_set, args.batch_size), # Tf2 new feature
          steps_per_epoch = len(train_set)/args.batch_size,
          epochs = args.epochs, verbose=1,
          validation_data = generate_batch(feature_dict, valid_set, args.batch_size),
          validation_steps = len(valid_set),
          #callbacks = [log, tensorboard, checkpoint, lr_decay],
          callbacks = [Metrics(valid_data=(generate_batch(feature_dict, valid_set, args.batch_size))),
          log, checkpoint, lr_decay],
          shuffle = True,
          #batch_size=args.batch_size,
          workers = 1).history

model.save_weights(args.save_dir + '/trained_weights.h5')
model.save(args.save_dir + '/trained_model.h5')
print('Trained model saved to \'%s/trained_model.h5\'' % args.save_dir)



