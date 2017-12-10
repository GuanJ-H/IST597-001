
# coding: utf-8

# In[1]:



import datetime
import pandas as pd
import random

import numpy as np
import pickle

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import csv
from keras import optimizers
from keras.models import Sequential
from keras.layers import core
from keras.layers.recurrent import LSTM,GRU,SimpleRNN

from keras.layers import BatchNormalization



def create_model(outputunits = 1,
                loss = 'mse'):
    
    # RNN model
    model = Sequential()
    model.add(LSTM(
            input_shape = (None, 1),
            units=64,
            return_sequences = True))
    model.add(LSTM(128, return_sequences = False))
    model.add(core.Dropout(0.2))
    # model.add(LSTM(400, return_sequences = False))
    model.add((BatchNormalization()))
    model.add(core.Dense(units = 256, activation='relu'))
    model.add(core.Dense(units = 64, activation='relu'))
    model.add(core.Dropout(0.2))
    model.add(core.Dense(units = outputunits, activation='linear'))

    model.compile(loss = loss, optimizer = optimizers.RMSprop(lr = 0.00005,decay = 0.000001))
    model.summary()
    return model


# In[40]:


def load_obj(name,path):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# In[59]:


def pretrain(PREPROCESSED_DATA_NAME, PATH, 
             predictday = 1,
             nbatch = None, 
             validationpct = 0.1,
             shuffle = True,
             nepoch = 30):
    
    stock_obj = load_obj(name = PREPROCESSED_DATA_NAME, path = PATH)
    # print(stock_obj)
    X_trainData, Y_trainData = np.array(stock_obj["X_trainData"]), np.array(stock_obj["Y_trainData"])
    # print(X_trainData.shape)
    # print(X_trainData)
    X_trainData = np.reshape(X_trainData, (X_trainData.shape[0], X_trainData.shape[1], 1))

    # print(X_trainData)
    model = create_model(predictday)
    print("start fitting")
    hist_obj = model.fit(X_trainData, Y_trainData,
            batch_size       = nbatch,
            epochs         = nepoch,
            validation_split = validationpct,
            shuffle = True,
            verbose = 2)
    model.save("/gpfs/scratch/gzh8/stock/model/model-lstm-sub_simp.h5")
    print("Model Saved")
# Test Unit -------------------------------------------------------------
PATH = '/gpfs/scratch/gzh8/stock/data/'
PREPROCESSED_DATA_NAME = 'preprocessed_stock_data_OnlyTrain100_Day5_TarAAPL'
# PREPROCESSED_DATA_NAME = "preprocessed_stock_data_10"
# PATH = "../stock_dataset/"
pretrain(PREPROCESSED_DATA_NAME, PATH, predictday = 5)



# In[30]:


# # # Plot loss functiom
# loss = hist_obj.history
# traintloss = loss['loss']
# testloss = loss['val_loss']
# plt.figure
# plt.plot(testloss, label="Test loss")
# plt.plot(traintloss, label="Traning loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.title(r'Company Ticker = {}'.format(tkIdx[0])+"\n"+r'cell = {}, dropout = {}, activation = {}, loss = {}, optimizer = {}, epoch = {}'.format('LSTM', 0.2, 'linear', 'mse', 'rmsprop', nepoch))
# plt.legend(loc='best')
# plt.savefig('loss-model-lstm-sub_simp')
# # plt.show()

