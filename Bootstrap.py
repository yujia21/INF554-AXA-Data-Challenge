
# coding: utf-8

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime
import time
get_ipython().magic('matplotlib inline')

#TODO : CHANGE ACCORDING TO NEEDS
imported_data=True # True if data/... already exists
evaluation=True # True if want to evaluate error of machine learning algo
submit=True # True if want to replace submission file by new values



#Import data if not already done
if (not imported_data) : 
    import ImportData



# BEGINNING OF LEARNING CODE
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

    # TODO : CODE HERE
    
    
    
    
    return predicted




# Function dates_to_datetime creates datetime stamps at every half hour from a list of date
def dates_to_datetime(dates) : 
    datetimes=[]
    times=["00:00:00.000", "00:30:00.000", "01:00:00.000", "01:30:00.000", 
           "02:00:00.000", "02:30:00.000", "03:00:00.000", "03:30:00.000", 
           "04:00:00.000", "04:30:00.000", "05:00:00.000", "05:30:00.000", 
           "06:00:00.000", "06:30:00.000", "07:00:00.000", "07:30:00.000", 
           "08:00:00.000", "08:30:00.000", "09:00:00.000", "09:30:00.000", 
           "10:00:00.000", "10:30:00.000", "11:00:00.000", "11:30:00.000", 
           "12:00:00.000", "12:30:00.000", "13:00:00.000", "13:30:00.000", 
           "14:00:00.000", "14:30:00.000", "15:00:00.000", "15:30:00.000", 
           "16:00:00.000", "16:30:00.000", "17:00:00.000", "17:30:00.000", 
           "18:00:00.000", "18:30:00.000", "19:00:00.000", "19:30:00.000", 
           "20:00:00.000", "20:30:00.000", "21:00:00.000", "21:30:00.000", 
           "22:00:00.000", "22:30:00.000", "23:00:00.000", "23:30:00.000"]
    for date in dates : 
        for time in times : 
            datetime_entry=str(date+' '+time)
            datetimes.append(datetime.strptime(datetime_entry, "%Y-%m-%d %H:%M:%S.%f"),)
    return datetimes

# Error function as defined in project
def linex(y_true,y_pred) : 
    a=0.1
    assert(len(y_true)==len(y_pred))
    diff=np.subtract(y_true,y_pred)
    return np.exp(a*diff)-a*diff-np.ones(len(y_true))




#EVALUATION
if (evaluation) : 
    # EXAMPLE DATE
    dates = ['2011-12-28', '2011-12-29', '2011-12-30', '2011-12-31', '2012-01-01', '2012-01-02', '2012-01-03']
    fulldates=dates_to_datetime(dates) #with timestamps, for predict function
    
    # Call predict
    predicted=predict(fulldates)

    # Dictionaries for true and predicted values
    y_true={}
    y_pred={}
    y_true.clear()
    y_pred.clear()
    err={}
    err.clear()
    err_tot=0
    len_tot=0
    
    # Stocking and printing error by service where there is data
    for x in ass_list : 
        y_true[x]=[]
        y_pred[x]=[]
        print("Evaluating error for : "+x)
        for t in fulldates : # for date
            for i in range(len(data[x])) : 
                if (data[x][i,0]==t) : # if date found in original data
                    y_true[x].append(data[x][i,1])
                    for j in range(len(predicted[x])) : #find date in predicted
                        if (str(predicted[x][j,0])==str(t)) : 
                            y_pred[x].append(predicted[x][j,1])
                            break # add predicted one time
                    break # add original one time
        err[x]=linex(y_true[x],y_pred[x])
        err_tot+=np.sum(err[x])
        len_tot+=len(err[x])
        #print("vector error : ")
        #print(err[x])
        print("error : ")
        print(np.sum(err[x]))
        print()
    print("total error : ")
    print(err_tot)




# UPDATE SUBMISSIONS.TXT
# ASS_ASSIGNMENT : because not every service appears at any time, 
# have to check values exist in submissions.txt that we received,
# we cannot construct new dataframe directly from predicted

if (submit) : 
    # List of dates in submission.txt file
    dates=[]
    for date in submission['DATE'].unique() : 
        dates.append(datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f"))
    
    # Call predict function
    predicted=predict(dates)
    
    #Rewrite submissions dataframe
    for x in ass_list : 
        print("Stocking predictions for "+x)
        if (x in predicted) : # x=assignment
            for row in predicted[x] : # row=[date,calls]
                # Convert timestamp to yyyy-mm-dd hh:mm:ss.xxx
                date = row[0].strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                submission.loc[(submission['DATE']==date)&(submission['ASS_ASSIGNMENT']==x), ['prediction']]=row[1]

                #better way to loop through this?
    
    fh = open("submission1.txt","w")
    fh.write("DATE\tASS_ASSIGNMENT\tprediction")
    for index, row in submission.iterrows(): 
        fh.write("\n")
        fh.write(str(row[0])+'\t'+str(row[1])+'\t'+str(row[2]))
    fh.close()
    print("Text file written!")

