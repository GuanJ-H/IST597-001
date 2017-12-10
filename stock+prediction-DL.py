
# coding: utf-8

# In[33]:


import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
import datetime
import pandas as pd
import random

import numpy as np
import pickle
import pydot
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import csv
from keras.utils import plot_model
from keras import optimizers
from keras.models import Sequential
from keras.layers import core, Flatten
from keras.layers.recurrent import LSTM,GRU,SimpleRNN
from keras.layers import BatchNormalization,Conv1D
# from keras import backend as K
# K.set_image_dim_ordering('tf')


# In[34]:


import quandl
quandl.ApiConfig.api_key = 'ZhXAeHP_M4TuzbXSznR6'




# In[35]:


def get_cummulative_return(data):
    cumret  = (data / data[0]) - 1

    return cumret


# In[36]:


def windowData(data, windowsize = 1, step = 1):
    # Note: this func will discard the tail of the data if they cannot fit the window fully.
    return np.array([data[i: i + windowsize] for i in range(0,(len(data)-windowsize+1), step)])

# Unit test ---------------------------------------------------------------------------------------
# data = np.array(range(20))
# print(data)
# slicedData = window(data, windowsize=4, step=2)
# print(slicedData)


# In[37]:


def normalize_in_window(data):
    # Normalize data in each window
    data
    return np.array([get_cummulative_return(split) for split in data])

# # Unit test ---------------------------------------------------------------------------------------
# data = np.array(range(20))
# print(data)
# slicedData = windowData(data, windowsize=4, step=2)
# print(slicedData)
# normailzeData = normalize_in_window(slicedData)
# print(normailzeData)
    




def load_obj(name,path):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# In[42]:


def splitSingleData(data,
          window_size    = 100,
          trainpct     = 0.60,
          shift     = 1,
          normalize = False,
          predictdays = 1):
    

    if trainpct < 1:
        size   = len(data)
        split  = int(np.rint(trainpct * size))

        train  = data[:split+1]
        test   = data[split+1:]    
    else:
        train = data
    

    offset = shift - 1

    window_train = windowData(train, windowsize = window_size)
    if trainpct < 1:
        window_test = windowData(test, windowsize = window_size, step = predictdays)

    if normalize:
        window_train = normalize_in_window(window_train)
        if trainpct < 1:
            window_test = normalize_in_window(window_test)

    
    
    Xtrain = window_train[:-window_size,:-1]
    ytrain = window_train[:, -1]
    
    if trainpct < 1:
        Xtest = window_test[::predictdays,:-1]
        ytest = window_test[:, -1] 
        temp = len(Xtest)-len(list(range(0,ytest.shape[0] - window_size, predictdays)))
        Xtest = Xtest[:-temp]

    # Chunk
    ytrain = np.array([ytrain[i:i+predictdays] for i in range(0,ytrain.shape[0] - window_size)])
    if trainpct < 1:
        ytest = np.array([ytest[i:i+predictdays] for i in range(0,ytest.shape[0] - window_size, predictdays)])

    if trainpct == 1:
        return (Xtrain, ytrain) 
    else:
        return (Xtrain, Xtest, ytrain, ytest) 

# Unit test --------------------------------------------------------------------------
# share = quandl.get("WIKI/AAPL")
# data = share["Close"]
# Xtrain,  ytrain = splitSingleData(data, trainpct = 1, normalize = False,predictdays = 3)
# print(Xtrain)
# print(Xtrain.shape)
# print(ytrain.shape)
# print(Xtest.shape)
# print(ytest.shape)
# Xtrain, Xtest, ytrain, ytest = split(share, 'Close', normalize = False)
# print(Xtrain)
# print(Xtrain.shape)
# print(ytrain.shape)







# In[44]:


def create_LSTM_1(outputunits):
    
    # RNN model
    model = Sequential()
    model.add(LSTM(
            input_shape = (None, 1),
            units=128,
            return_sequences = False))
#     model.add(LSTM(200, return_sequences = True))
#     model.add(core.Dropout(0.2))
#     model.add(LSTM(100, return_sequences = False))
#     model.add(core.Dropout(0.2))
    model.add(core.Dense(units = 64, activation = 'relu'))
    model.add(core.Dense(units = outputunits, activation = 'linear')) 

    model.compile(loss = 'mse', optimizer = optimizers.RMSprop())
    plot_model(model,to_file='model22.png', show_shapes= True, show_layer_names=False)
    model.summary()
    return model

# Test Unit----------------
create_LSTM_1(5)


# In[45]:


def create_LSTM_2(outputunits):
    
    # RNN model
    model = Sequential()
    model.add(LSTM(
            input_shape = (None, 1),
            units=128,
            return_sequences = True))
    model.add(LSTM(64, return_sequences = True))
#     model.add(core.Dropout(0.2))
#     model.add(LSTM(64, return_sequences = False))
#     model.add(core.Dropout(0.2))

    model.add(core.Dense(units = 64, activation = 'relu'))
    model.add(core.Dense(units = outputunits, activation = 'linear')) 
    
    model.compile(loss = 'mse', optimizer = optimizers.RMSprop())
    plot_model(model,to_file='lstm.png', show_shapes= True, show_layer_names=False)
    model.summary()
    return model
# Test Unit----------------
create_LSTM_2(5)


# In[46]:


def create_LSTM_3(outputunits):
    
    # RNN model
    model = Sequential()
    model.add(LSTM(
            input_shape = (None, 1),
            units=128,
            return_sequences = True))
    model.add(LSTM(128, return_sequences = True))
#     model.add(core.Dropout(0.2))
    model.add(LSTM(64, return_sequences = False))
#     model.add(core.Dropout(0.2))

    model.add(core.Dense(units = 64, activation = 'relu'))
    model.add(core.Dense(units = outputunits, activation = 'linear')) 
    
    model.compile(loss = 'mse', optimizer = optimizers.RMSprop())
    model.summary()
    return model
    
# Test Unit ------------
create_LSTM_3(5)


# In[47]:


def create_SimpleRNN_1(outputunits):
    
    # RNN model
    model = Sequential()
    model.add(SimpleRNN(
            input_shape = (None, 1),
            units=128,
            return_sequences = False))
#     model.add(SimpleRNN(500, return_sequences = True))
# #     model.add(core.Dropout(0.2))
#     model.add(SimpleRNN(600, return_sequences = False))
#     model.add((BatchNormalization()))
#     model.add(core.Dropout(0.2))

    model.add(core.Dense(units = 64, activation = 'relu'))
    model.add(core.Dense(units = outputunits, activation = 'linear')) 

    model.compile(loss = 'mse', optimizer = optimizers.RMSprop())
    model.summary()
    
    return model
# Test Unit----------------
create_SimpleRNN_1(5)


# In[48]:


def create_SimpleRNN_2(outputunits):
    
    # RNN model
    model = Sequential()
    model.add(SimpleRNN(
            input_shape = (None, 1),
            units=128,
            return_sequences = True))
    model.add(SimpleRNN(64, return_sequences = False))
# #     model.add(core.Dropout(0.2))
#     model.add(SimpleRNN(600, return_sequences = False))
#     model.add((BatchNormalization()))
#     model.add(core.Dropout(0.2))

    model.add(core.Dense(units = 64, activation = 'relu'))
    model.add(core.Dense(units = outputunits, activation = 'linear')) 

    model.compile(loss = 'mse', optimizer = optimizers.RMSprop())
    plot_model(model,to_file='model22.png', show_shapes= True, show_layer_names=False)
#     model.summary()
    return model

# Test Unit----------------
create_SimpleRNN_2(5)


# In[49]:


def create_SimpleRNN_3(outputunits):
    
    # RNN model
    model = Sequential()
    model.add(SimpleRNN(
            input_shape = (None, 1),
            units=128,
            return_sequences = True))
    model.add(SimpleRNN(128, return_sequences = True))
    model.add(SimpleRNN(64, return_sequences = False))
# #     model.add(core.Dropout(0.2))
#     model.add(SimpleRNN(600, return_sequences = False))
#     model.add((BatchNormalization()))
#     model.add(core.Dropout(0.2))

    model.add(core.Dense(units = 64, activation = 'relu'))
    model.add(core.Dense(units = outputunits, activation = 'linear')) 

    model.compile(loss = 'mse', optimizer = optimizers.RMSprop())
    model.summary()
    plot_model(model,to_file='model1.png', show_shapes= True, show_layer_names=False)
    return model
# Test Unit----------------
create_SimpleRNN_3(5)


# In[50]:


def create_GRU_1(outputunits):
    
    # RNN model
    model = Sequential()
    model.add(GRU(
            input_shape = (None, 1),
            units=128,
            return_sequences = False))
#     model.add(SimpleRNN(128, return_sequences = True))
#     model.add(SimpleRNN(64, return_sequences = False))
# #     model.add(core.Dropout(0.2))
#     model.add(SimpleRNN(600, return_sequences = False))
#     model.add((BatchNormalization()))
#     model.add(core.Dropout(0.2))

    model.add(core.Dense(units = 64, activation = 'relu'))
    model.add(core.Dense(units = outputunits, activation = 'linear')) 

    model.compile(loss = 'mse', optimizer = optimizers.RMSprop(decay = 0.00005))
    model.summary()
    return model


# In[51]:


def create_GRU_2(outputunits):
    
    # RNN model
    model = Sequential()
    model.add(GRU(
            input_shape = (None, 1),
            units=128,
            return_sequences = True))
    model.add(GRU(64, return_sequences = False))
#     model.add(SimpleRNN(64, return_sequences = False))
# #     model.add(core.Dropout(0.2))
#     model.add(SimpleRNN(600, return_sequences = False))
#     model.add((BatchNormalization()))
#     model.add(core.Dropout(0.2))

    model.add(core.Dense(units = 64, activation = 'relu'))
    model.add(core.Dense(units = outputunits, activation = 'linear')) 

    model.compile(loss = 'mse', optimizer = optimizers.RMSprop())
    model.summary()
    return model


# In[52]:


def create_GRU_3(outputunits):
    
    # RNN model
    model = Sequential()
    model.add(GRU(
            input_shape = (None, 1),
            units=128,
            return_sequences = True))
    model.add(GRU(128, return_sequences = True))
    model.add(GRU(64, return_sequences = False))
# #     model.add(core.Dropout(0.2))
#     model.add(SimpleRNN(600, return_sequences = False))
#     model.add((BatchNormalization()))
#     model.add(core.Dropout(0.2))

    model.add(core.Dense(units = 64, activation = 'relu'))
    model.add(core.Dense(units = outputunits, activation = 'linear')) 

    model.compile(loss = 'mse', optimizer = optimizers.RMSprop())
    plot_model(model,to_file='GRU.png', show_shapes= True, show_layer_names=False)
    model.summary()
    return model

# Test Unit----------------
create_GRU_3(5)


# In[66]:


def stock_predict():

    share = quandl.get("WIKI/AAPL")
    data = share["Close"]

    MSE = []
    predictday = 1

    nbatch      = 128  
    nepoch      = 40
    nvalidation = 0.05

    learning_rate = 0.0005
    dropout    = 0.2
    activation = 'linear'
    loss = 'mse'

        
    # Xtrain, Xtest, ytrain, ytest = splitShare(share, 'Close', normalize = True,predictdays = predictday)
    Xtrain, Xtest, ytrain, ytest = splitSingleData(data, window_size = 100,  normalize = True, predictdays = predictday)

    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
    Xtest  = np.reshape( Xtest, ( Xtest.shape[0],  Xtest.shape[1], 1))


    # Choose one model
    model = create_SimpleRNN_1(predictday)


    # Training
#     validation = (Xtest, ytest)
    hist_obj = model.fit(Xtrain, ytrain,
            batch_size       = nbatch,
            epochs         = nepoch,
            validation_split = 0.1)

    # Testing
    p = model.predict(Xtest)
    mse = mean_squared_error(ytest, p)
    MSE.append(mse)
    
    return p, ytest, MSE, hist_obj

# Test Unit ----------------------------------------------------------------------
p, ytest, MSE, hist_obj = stock_predict()


# In[62]:


print("Loss of prediction is ", MSE, r"(cell = {}, dropout = {}, activation = {}, loss = {}, optimizer = {}, epoch = {})".format('LSTM', 0.2, 'linear', 'mse', 'rmsprop', 15))


# In[58]:


print(plt.style.available)


# In[70]:


# # Plot predction figure
plt.style.use('ggplot')
plt.figure
plt.plot(ytest.flatten(), label="Ground Truth")
plt.plot(p.flatten(), label="Prediction")
plt.xlabel("Days")
plt.ylabel("Close Price")
plt.title(r'Company Ticker = {}'.format('AAPL')+"\n"+r'cell = {},  loss = {}, optimizer = {}, epoch = {}'.format('SimpleRNN-1','mse', 'rmsprop', 40))
plt.legend(loc='best')
# plt.savefig('predict')
plt.show()


# In[68]:


# # Plot loss functiom
loss = hist_obj.history
traintloss = loss['loss']
testloss = loss['val_loss']
plt.figure
plt.plot(testloss, label="Validation loss")
plt.plot(traintloss, label="Traning loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(r'Company Ticker = {}'.format('AAPL')+"\n"+r'cell = {},  loss = {}, optimizer = {}, epoch = {}'.format('SimpleRNN-1','mse', 'rmsprop', 40))
plt.legend(loc='best')
# plt.savefig('loss')
plt.show()




