# -*- coding:utf-8 -*-
"""
Author: Yihang Bao
Created: Apr 25, 2021
Last modified: Apr 26, 2021
"""

import keras
import numpy as np

class data_gener(keras.utils.Sequence):

    def __init__(self, labels, batch_size, edge, dim,
                 n_classes, shuffle=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.drug_profile = np.load('data/feature.npy')
        self.labels = labels
        self.drug_id_dict = {}
        self.edge = edge
        for i in range(0, len(self.drug_profile)):
            self.drug_id_dict[self.drug_profile[i][-1]] = i
        self.list_IDs = []
        for i in range(0, edge):
            # print(np.r_[drug_profile[int(y[i][0])], drug_profile[int(y[i][1])]])
            self.list_IDs.append([self.labels[i][0], self.labels[i][1]])
        self.indexes = np.arange(edge)
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        base_index = index * self.batch_size

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, base_index)

        return X, y

    def __data_generation(self, list_IDs_temp, base_index):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty(self.batch_size, dtype=int)


        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.c_[self.drug_profile[self.drug_id_dict[ID[0]]], self.drug_profile[self.drug_id_dict[ID[1]]]].transpose()[:, :-1]
            # Store class
            y[i] = self.labels[base_index+i][2]
        print(X.shape)
        print(keras.utils.to_categorical(y, num_classes=self.n_classes).shape)
        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)

