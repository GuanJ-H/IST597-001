
# coding: utf-8

# In[1]:


import datetime
import pandas as pd
import random
import quandl
import numpy as np
import time
import pickle
quandl.ApiConfig.api_key = 'ZhXAeHP_M4TuzbXSznR6'


# In[2]:




def save_obj(obj, name, path):
    with open(path+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)





def load_obj(name,path):
    with open(path + name + '.pkl', 'rb') as f:
        return pickle.load(f)


# In[4]:


def stock_collect(CODE_FILE_NAME,attr, subsample = True):
    df = pd.read_csv(CODE_FILE_NAME)
    codes = df.iloc[:,0]  
    
    if subsample == True:
        # Option 1: n random company (Comment if do not choose)
        n = 3
        effective_company = n
        random.seed(1)
        idx = random.sample(range(len(df)), n)
    else:
        # Option 2: all companys in code file (Comment if do not choose)
        n = 3181
        idx = range(n)
        effective_company = n

    tkIdx = []
    for i in idx:
        tkIdx.append(codes[i])
        
    # Data = []
    DataDict = {}    
    for i in range(len(tkIdx)):
        try:
            share = quandl.get(tkIdx[i])
            data_pd = share[attr]
            data_list = data_pd.tolist()
            DataDict[tkIdx[i]] = data_list
            print(tkIdx[i])
        except Exception:
            effective_company -= 1
            print("ERROR of Quandl happened:", tkIdx[i])
            pass 
            continue
        time.sleep(1)
    print("number of effective company: ", effective_company)
    # np.save("stock_data_{}_{}.npy".format(n,effective_company), Data)
    path = "../stock_dataset/"
    save_obj(DataDict, "stock_data_{}_{}".format(n,effective_company),path)
    print("Stock data saved!")
# Unit Test-----------------------------------------------------------
CODE_FILE_NAME = "WIKI-datasets-codes.csv"
stock_collect(CODE_FILE_NAME, attr = "Close", subsample = True)


