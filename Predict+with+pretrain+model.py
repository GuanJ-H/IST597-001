
# coding: utf-8

# In[36]:


from keras.models import load_model
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from keras import optimizers
from keras.models import Sequential
from keras.layers import core
from keras.layers.recurrent import LSTM,GRU,SimpleRNN
from keras.layers import BatchNormalization

import quandl
quandl.ApiConfig.api_key = 'ZhXAeHP_M4TuzbXSznR6'


# In[20]:


def get_cummulative_return(data):
    cumret  = (data / data[0]) - 1

    return cumret


# In[21]:


def windowData(data, windowsize = 1, step = 1):
    # Note: this func will discard the tail of the data if they cannot fit the window fully.
    return np.array([data[i: i + windowsize] for i in range(0,(len(data)-windowsize+1), step)])

# Unit test ---------------------------------------------------------------------------------------
# data = np.array(range(20))
# print(data)
# slicedData = window(data, windowsize=4, step=2)
# print(slicedData)


# In[22]:


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


# In[16]:


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


# In[35]:


# model = load_model('model-lstm-sub_simp.h5')
# model.summary()
def create_model(outputunits = 1,
                loss = 'mse'):
    
    # RNN model
    model = Sequential()
    model.add(LSTM(
            input_shape = (None, 1),
            units=60,
            return_sequences = True))
    model.add(LSTM(100, return_sequences = False))
    # model.add(core.Dropout(0.3))
    # model.add(LSTM(400, return_sequences = False))
    model.add((BatchNormalization()))
    model.add(core.Dense(units = 100))
    model.add(core.Dropout(0.3))
    model.add(core.Dense(units = 40))
    model.add(core.Dropout(0.2))
    model.add(core.Dense(units = outputunits))
    model.add(core.Activation('linear')) 

    model.compile(loss = loss, optimizer = optimizers.RMSprop())
    model.summary()
    return model


# In[54]:


def predict(PRETRAIN_MODEL, targetCode='WIKI/AAPL'):
    share = quandl.get(targetCode)
    data = share["Close"]
    
#     model = create_model(outputunits = 5)

    nbatch      = 128  
    nepoch      = 100
    predictday = 5
    
    Xtrain, Xtest, ytrain, ytest = splitSingleData(data, window_size = 100,  normalize = True, predictdays = predictday)
    
    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], 1))
    Xtest  = np.reshape( Xtest, ( Xtest.shape[0],  Xtest.shape[1], 1))
    
    model = load_model(PRETRAIN_MODEL)
    model.compile(loss = 'mse', optimizer = optimizers.RMSprop(lr = 0.00005,decay = 0.000001))
#     validation = (Xtest, ytest)
    hist_obj = model.fit(Xtrain, ytrain,
            batch_size       = nbatch,
            epochs         = nepoch,
            validation_split = 0.1)
    
    p = model.predict(Xtest)
    mse = mean_squared_error(ytest, p)
    
    return p, ytest,mse, hist_obj
# Test Unit ---------------------------------------------------------
targetCode = 'WIKI/AAPL'
PRETRAIN_MODEL = 'model-lstm-sub_simp.h5'
p, ytest,mse, hist_obj=predict(PRETRAIN_MODEL, targetCode=targetCode) 
print(mse)


# In[55]:


# Plot predction figure
plt.figure
plt.plot(ytest.flatten(), label="Ground Truth")
plt.plot(p.flatten(), label="Prediction")
plt.xlabel("Days")
plt.ylabel("Close Price")
plt.title(r'Company Ticker = {}'.format('AAPL')+"\n"+r'cell = {}, dropout = {}, activation = {}, loss = {}, optimizer = {}, epoch = {}'.format('LSTM', 0.2, 'linear', 'mse', 'rmsprop', 50))
plt.legend(loc='best')
# plt.savefig('predict')
plt.show()


# In[51]:


# # Plot loss functiom
loss = hist_obj.history
traintloss = loss['loss']
valloss = loss['val_loss']
plt.figure
# plt.plot(valloss, label="Validation loss")
plt.plot(traintloss, label="Traning loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(r'Company Ticker = {}'.format('AAPL')+"\n"+r'cell = {},  loss = {}, optimizer = {}, epoch = {}'.format('LSTM-1','mse', 'rmsprop', 15))
plt.legend(loc='best')
# plt.savefig('loss')
plt.show()

