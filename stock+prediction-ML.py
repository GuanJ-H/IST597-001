
# coding: utf-8

# In[37]:


import datetime
import pandas as pd
import random
import numpy as np

from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


import csv

from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Perceptron
# from tempfile import TemporaryFile
import quandl
quandl.ApiConfig.api_key = 'ZhXAeHP_M4TuzbXSznR6'


# In[2]:


def get_cummulative_return(data):
    
    cumret  = (data / data[0]) - 1

    return cumret


# In[ ]:





# In[25]:


def splitShare(share,
          attrs     = 'Close',
          window    = 0.01,
          train     = 0.60,
          shift     = 1,
          normalize = False,
          predictdays = 1):
    
    Data   = share[attrs]
    
    data = []
    for d in Data:
        data.append(d)
    
    size   = len(data)
    split  = int(np.rint(train * size))
    
    train  = data[:split+1]
    test   = data[split+1:]    
    
#     while (len(test)%predictdays != 0):
#         test   = test[:-1]
    
#     while (len(train)%predictdays != 0):
#         train   = train[:-1]    
    
    length = len(share)

    window = int(np.rint(length * window))
    window = 100
    offset = shift - 1

    splits_train = np.array([train[i if i is 0 else i + offset: i + window] for i in range(0,len(train) - window)])

    splits_test = np.array([test[i if i is 0 else i + offset: i + window] for i in range(0,len(test) - window, predictdays)])

    if normalize:
        splits_train = np.array([get_cummulative_return(split) for split in splits_train])
        splits_test = np.array([get_cummulative_return(split) for split in splits_test])

    
    
    Xtrain, Xtest = splits_train[:-window,:-1], splits_test[::predictdays,:-1]
    ytrain, ytest = splits_train[:, -1], splits_test[:, -1]
    temp = len(Xtest)-len(list(range(0,ytest.shape[0] - window, predictdays)))
    Xtest = Xtest[:-temp]

    # Chunk
    ytrain = np.array([ytrain[i:i+predictdays] for i in range(0,ytrain.shape[0] - window)])
    ytest = np.array([ytest[i:i+predictdays] for i in range(0,ytest.shape[0] - window, predictdays)])


    return (Xtrain, Xtest, ytrain, ytest)   

# Unit test --------------------------------------------------------------------------
# share = bb.Share('WIKI','FB')
# Xtrain, Xtest, ytrain, ytest = splitShare(share, 'Close', normalize = False,predictdays = 3)
# print(Xtrain)
# print(Xtrain.shape)
# print(ytrain.shape)
# print(Xtest.shape)
# print(ytest.shape)
# Xtrain, Xtest, ytrain, ytest = split(share, 'Close', normalize = False)
# print(Xtrain)
# print(Xtrain.shape)
# print(ytrain.shape)


# In[26]:


def windowData(data, windowsize = 1, step = 1):
    # Note: this func will discard the tail of the data if they cannot fit the window fully.
    return np.array([data[i: i + windowsize] for i in range(0,(len(data)-windowsize+1), step)])

# Unit test ---------------------------------------------------------------------------------------
# data = np.array(range(20))
# print(data)
# slicedData = window(data, windowsize=4, step=2)
# print(slicedData)






MSE = []
tkIdx = ['WIKI/AAPL']
layers      = [1, 100, 100, 1]# number of neurons in each layer
nbatch      = 512  
epochs      = 6
nvalidation = 0.05
predictday = 5

for i in range(len(tkIdx)):
    try:
        share = quandl.get(tkIdx[i])
        Data   = share['Close']
        dataset = []
        for data in Data:
            dataset.append(data)

    except Exception:
        numSample = numSample - 1
        pass
        print("skip")
        continue

    
    Xtrain, Xtest, ytrain, ytest = splitShare(share, normalize = True, predictdays = predictday)


    Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1]))
    Xtest  = np.reshape( Xtest, ( Xtest.shape[0],  Xtest.shape[1]))

    # Decision Tree Regressor
#     regressor  = DecisionTreeRegressor()
#     regressor = regressor.fit(Xtrain,ytrain)
#     p = regressor.predict(Xtest)

    # linear regression
    clf = LinearRegression(fit_intercept=False)
    clf.fit(Xtrain,ytrain)
    p = clf.predict(Xtest)


    mse = mean_squared_error(ytest, p)
    MSE.append(mse)
    print("done")



# In[48]:


print("Loss of prediction is ", MSE)





# In[49]:


plt.figure
plt.style.use('ggplot')
plt.plot(range(0,len(Xtest[0])),Xtest[0],label="Training Data")
plt.plot(range(len(Xtest[0]),len(Xtest[0])+len(ytest[0])),ytest[0], label="Ground Truth")
plt.plot(range(len(Xtest[0]),len(Xtest[0])+len(p[0])),p[0], label="Prediction")
plt.xlabel("Days")
plt.ylabel("Close Price")
plt.title(r'Company Ticker = {}, Predict Days = {}'.format('AAPL', predictday)+"\n"+"Method = Decision Tree Regression")
plt.legend(loc='best')
plt.savefig('predict_ML_5D_part')
plt.show()


# In[35]:


# Plot predction figure
plt.figure
plt.style.use('ggplot')
plt.plot(ytest.flatten(), label="Ground Truth")
plt.plot(p.flatten(), label="Prediction")
plt.xlabel("Days")
plt.ylabel("Close Price")
plt.title(r'Company Ticker = {}, Predict Days = {}'.format(tkIdx[0], predictday)+"\n"+"Method = Decision Tree Regression")
plt.legend(loc='best')
plt.savefig('predict_ML_5D')
plt.show()


# In[243]:


# Plot loss function
loss = hist_obj.history
testloss = loss['loss']
trainingloss = loss['val_loss']
plt.figure
plt.plot(testloss, label="Test loss")
plt.plot(trainingloss, label="Traning loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(r'Company Ticker = {}'.format(tkIdx[0])+"\n"+r'cell = {}, dropout = {}, activation = {}, loss = {}, optimizer = {}, epoch = {}'.format('LSTM', 0.2, 'linear', 'mse', 'rmsprop', epochs))
plt.legend(loc='best')
plt.savefig('loss')
# plt.show()


# In[29]:


plt.plot(share.data['Close'])
plt.show()


# In[25]:




