# INF554Projet

Due to space constraints, train_2011_2012_2013.csv and data/... files are not included.

# Bootstrap.py : 
This is the base file for writing the predict functions. (Line 40)

There are three options to play with on line 9 : 

  imported_data=True 
  True if data/... already exists
  If False, will run ImportData.py, which takes train_2011_2012_2013.csv and creates data/...
  
  evaluation=True 
  True if want to evaluate error of machine learning algo. Can change the dates called
  TODO: add Frapy's evaluation function
  
  submit=True 
  True if we want to create a submission file with new prediction values. 
  Dates predicted are always the dates provided in the original submission.txt file
