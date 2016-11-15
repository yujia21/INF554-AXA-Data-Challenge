# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
import time

#TODO : CHANGE ACCORDING TO NEEDS
imported_data=True # True if data/... already exists
evaluation=True # True if want to evaluate error of machine learning algo
submit=True # True if want to replace submission file by new values

#Import data if not already done
if (not imported_data) : 
	import ImportData

# Read submission data
submission = pd.read_csv('submission.txt', sep='\t')

# Stock list of services to predict
ass_list=submission['ASS_ASSIGNMENT'].unique()

# Imports data into dictionary
data = {}
for x in ass_list : 
    data[x] = np.load('data/'+x+'.npy')


# data is ambient dictionary in this script

# predict takes in "dates" which is a list of dates (continuous)
# predict returns a dictionary predicted where:
# keys = services, values = numpy arrays (TimeStamp, Calls Received)

# predict contains a machine learning algorithm that does : 
# 1. Preprocessing - feature reduction + engineering features if necessary
# 2. Learning - regression, random forest, neural network, etc.

def predict(dates) : 
    #Initialize and clear dictionary to return
    predicted={}
    predicted.clear()

    
    
    
    
    # EXAMPLE PREDICTION
    predicted['CMS']=np.array([[datetime.strptime('2012-12-28 00:00:00.000', "%Y-%m-%d %H:%M:%S.%f"),1], 
                           [datetime.strptime('2012-12-28 00:30:00.000', "%Y-%m-%d %H:%M:%S.%f"),1]])
    # TODO : CODE HERE




#EVALUATION
if (evaluation) : 
    # INSERT FRAPY'S EVALUATION FUNCTION HERE
    # EXAMPLE DATE
    dates = ['2011-12-28' '2011-12-29' '2011-12-30' '2011-12-31' '2012-01-01' '2012-01-02' '2012-01-03']

    # Call predict
    predict(dates)


    
    print("Error for dates : ")
    print(dates)
    #plt.plot(time,ave)
    #plt.show()


# UPDATE SUBMISSIONS.TXT
# ASS_ASSIGNMENT : because not every service appears at any time, 
# have to check values exist in submissions.txt that we received,
# we cannot construct new dataframe directly from predicted

if (submit) : 
    # List of dates in submission.txt file
    dates=submission['DATE'].unique()
    
    # Call predict function
    predict(dates)
    
    #Rewrite submissions dataframe
    for x in ass_list : 
        if (x in predicted) : # x=assignment
            for row in predicted[x] : # row=[date,calls]
                # Convert timestamp to yyyy-mm-dd hh:mm:ss.xxx
                date = row[0].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                submission.loc[(submission['DATE']==date)&(submission['ASS_ASSIGNMENT']==x), ['prediction']]=row[1]

                #better way to loop through this?
    # EXPORT to submission1.txt : 
    np.savetxt('submission1.txt',submission.values, comments='', header='DATE\tASS_ASSIGNMENT\tprediction', delimiter='\t', fmt='%s %s %d')
