# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')

# README : If data/xx.npy does not exist, change to False
imported_data=True

# Read submission data
submission = pd.read_csv('submission.txt', sep='\t')

# Stock list of services to predict
ass_list=submission['ASS_ASSIGNMENT'].unique()

if (! imported_data) : 
	# Import data
	# GIVEN: DATE, ASS_ASSIGNMENT
	# PREDICT : CSPL_CALLS

	fields = ['DATE', 'ASS_ASSIGNMENT','CSPL_CALLS']
	df = pd.read_csv('train_2011_2012_2013.csv', sep=';', usecols=fields)

	# Convert date field entries to datetime64[ns] type
	df['DATE']=pd.to_datetime(df['DATE'])


	# Separating by service
	for x in ass_list : 
	    # Create smaller df with just that service, cols = datetime and # 	of calls
	    ass=df[df['ASS_ASSIGNMENT']==x][['DATE','CSPL_CALLS']]

	    # Convert to nparray, export
    	np.save('data/'+x,ass.values)




# BEGINNING OF LEARNING CODE

# Imports data into dictionary
data = {}
for x in ass_list : 
    data[x] = np.load('data/'+x+'.npy')

