
# coding: utf-8

# In[1]:

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# README : If data/xx.npy does not exist, change to False
imported_data=True


# In[2]:

# Read submission data
submission = pd.read_csv('submission.txt', sep='\t')

# Stock list of services to predict
ass_list=submission['ASS_ASSIGNMENT'].unique()


# In[4]:

# Import data
# GIVEN: DATE, ASS_ASSIGNMENT
# PREDICT : CSPL_CALLS
if not imported_data : 
    fields = ['DATE', 'ASS_ASSIGNMENT','CSPL_CALLS']
    df = pd.read_csv('train_2011_2012_2013.csv', sep=';', usecols=fields)


# In[5]:

if not imported_data :    
    # Convert date field entries to datetime64[ns] type
    df['DATE']=pd.to_datetime(df['DATE'])
    
    # Separating by service
    for x in ass_list : 
        # Create smaller df with just that service, cols = datetime and # of calls
        ass=df[df['ASS_ASSIGNMENT']==x][['DATE','CSPL_CALLS']]

        # Convert to nparray, export
        np.save('data/'+x,ass.values)
