
# coding: utf-8

# In[8]:


import pickle
import numpy as np


# In[1]:


def get_cummulative_return(data):
    cumret  = (data / data[0]) - 1
    # print("data:", data)
    return cumret


# In[2]:


def load_obj(name,path):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# In[3]:


def save_obj(obj, name, path):
    with open(path+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# In[4]:


def windowData(data, windowsize = 1, step = 1):
    # Note: this func will discard the tail of the data if they cannot fit the window fully.
    return np.array([data[i: i + windowsize] for i in range(0,(len(data)-windowsize+1), step)])

# Unit test ---------------------------------------------------------------------------------------
# data = np.array(range(20))
# print(data)
# slicedData = window(data, windowsize=4, step=2)
# print(slicedData)


# In[5]:


def normalize_in_window(data):
    # Normalize data in each window
    
    return np.array([get_cummulative_return(split) for split in data])

# Unit test ---------------------------------------------------------------------------------------
# data = np.array(range(20))
# print(data)
# slicedData = windowData(data, windowsize=4, step=2)
# print(slicedData)
# normailzeData = normalize_in_window(slicedData)
# print(normailzeData)
    


# In[6]:


def splitSingleData(data,
          window_size    = 100,
          trainpct     = 0.60,
          shift     = 1,
          normalize = False,
          predictdays = 1):
    
#     Data   = share[attrs]
    
#     data = []
#     for d in Data:
#         data.append(d)
    if trainpct < 1:
        size   = len(data)
        split  = int(np.rint(trainpct * size))

        train  = data[:split+1]
        test   = data[split+1:]    
    else:
        train = data
      

    length = len(data)

    window_size = int(window_size)
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


# In[18]:


def splitAllData(DataDict, 
                 targetCode,
                 window_size    = 100,
                 trainpct  = 0.60,
                 shift     = 1,
                 normalize = True,
                 predictdays = 1):
    
    
    targetData = DataDict[targetCode]
    n = len(DataDict)
    DataDict.pop(targetCode,0)
    
    X_trainData = []
    Y_trainData = []
    
    for key, val in DataDict.items():
        if len(val) <= 130: 
            print("prune for less than 130")
            n = n-1
            continue 
        X_data,Y_data = splitSingleData(val,
              window_size    = 100,
              trainpct = 1.0,
              shift = shift,
              normalize = normalize,
              predictdays = predictdays)

        for i in range(len(Y_data)):
            X_trainData.append(X_data[i])
            Y_trainData.append(Y_data[i])
        
    # X_train_targetData, X_test_targetData, Y_train_targetData, Y_test_targetData = splitSingleData(val,
    #           window_size = window_size,
    #           trainpct = trainpct,
    #           shift = shift,
    #           normalize = normalize,
    #           predictdays = predictdays)
    # print("X_trainData shape: ", np.array(X_trainData).shape)
    print("X_trainData: ", X_trainData)
    print("X_trainData shape: ", np.array(X_trainData).shape)
    pDataDict= {"X_trainData": X_trainData,
                "Y_trainData": Y_trainData}
                # "X_train_targetData":X_train_targetData,
                # "X_test_targetData": X_test_targetData,
                # "Y_train_targetData_": Y_train_targetData,
                # "Y_test_targetData":Y_test_targetData}
    
    path = "/gpfs/scratch/gzh8/stock/data/"
    save_obj(pDataDict, "preprocessed_stock_data_Train{}_Day{}_Tar_AAPL".format(n, predictdays),path)
    print("Preprocessed stock data saved: pDataDict")
    
    return pDataDict
    
# Test Unit -----------------------------------------------------------------------------
# stock_data = load_obj(name = "stock_data_10_10", path = "../stock_dataset/")
# print(stock_data.keys())
# targetCode = "WIKI/SFE"
# pDataDict = splitAllData(stock_data, targetCode)
# print(pDataDict.keys())


# In[17]:


stock_data = load_obj(name = "stock_data_3181_3180", path = "../stock_dataset/")
targetCode = 'WIKI/AAPL'

obj = splitAllData(stock_data, 
                 targetCode,
                 window_size    = 0.01,
                 trainpct  = 0.60,
                 shift     = 1,
                 normalize = True,
                 predictdays = 5)

print(obj.keys())

