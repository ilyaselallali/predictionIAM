import pandas as pd

from plotstockData import *
from modelLearning import *
from dataSource import *

#---------------------------LOAD DATA------------------------------------
X,y=loadData()

#---------------------------PLOT DATA------------------------------------

#close price over time
CloseOverTime(X,y)

#frequency of close price values
CloseValueFrequency(y)

# Scatter plot for 'Close' vs 'Open'
CloseVsOpen(X,y)

Xn=X.drop('Date', axis=1)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(Xn, y, test_size=0.2, random_state=42)

print(X_train)
print(y_train)


#----------------------------TEST MODELS----------------------------

#linear regression--------------------
LinearmodelError(X_train, y_train,X_test,y_test)



#ann-------------------------

#ann(X_train, y_train)

#anntm
#yp=annTm(X_test,y_test)
decT(X_train, y_train,X_test,y_test)
#random forest--------------------
forestMod(X_train, y_train,X_test,y_test)