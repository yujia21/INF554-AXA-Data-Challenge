
# coding: utf-8

# In[3]:

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# README : If data/xx.npy does not exist, change to False
imported_data=False


# In[2]:

# Read submission data
submission = pd.read_csv('submission.txt', sep='\t')

# Stock list of services to predict
ass_list=submission['ASS_ASSIGNMENT'].unique()


# In[5]:

# Import data
# GIVEN: DATE, ASS_ASSIGNMENT
# PREDICT : CSPL_CALLS
if not imported_data : 
    fields = ['DATE', 'ASS_ASSIGNMENT','CSPL_CALLS']
    df = pd.read_csv('train_2011_2012_2013.csv', sep=';', usecols=fields)
    df.columns = ['DATETIME', 'ASS_ASSIGNMENT','CSPL_CALLS']
    print("Original length : "+str(len(df)))
    
    # Sum over moments with multiple data for same assignment
    df=df.groupby(['DATETIME','ASS_ASSIGNMENT']).sum()
    df=df.reset_index()
    print("Compressed length : "+str(len(df)))


# In[8]:

if not imported_data :    
    # Convert date field entries to datetime64[ns] type
    df['DATETIME']=pd.to_datetime(df['DATETIME'])
    
    # Separating by service
    for x in ass_list : 
        # Create smaller df with just that service, cols = datetime and # of calls
        ass=df[df['ASS_ASSIGNMENT']==x][['DATETIME','CSPL_CALLS']]

        # Convert to nparray, export
        np.save('data/'+x,ass.values)

