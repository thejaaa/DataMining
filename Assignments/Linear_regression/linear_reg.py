# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 22:59:52 2017

@author: theju
"""
#importing all the libraries.
import matplotlib.pyplot as plt
import pandas as pd
#for splitting the data.
from sklearn.cross_validation import train_test_split
# fro using linear regression.
from sklearn.linear_model import LinearRegression

# the csv file is loaded by using pandas.
set=pd.read_csv("C:\\Users\\theju\\Downloads\\Data.csv")
print(set)
# x-axis has the values in the first index.
x=set.iloc[:, :-1].values
#y-axis has the values of secpnd index.
y=set.iloc[:, 1].values
print(x)
print(y)

# splitting the datasets into training and testing sets by half.
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3,random_state=0)

print(x_train)
print(x_test)
print(y_train)
print(y_test)
#fit the linear regression in the training sets.
reg= LinearRegression()
reg.fit(x_train,y_train)

# predicting the result.
y_pred = reg.predict(x_test)

#plotting the values that reflects as a graph.
#testing datasets values as green color
plt.scatter(x_test,y_test,color='green')
plt.plot(x_test,y_pred,color='red',linewidth=3)
#training datasets values as red color
plt.scatter(x_train,y_train, color = 'blue')
plt.plot(x_test,y_pred,color = 'red',linewidth=3)
plt.xlabel('Intrest Rate(%)')
plt.ylabel('Median Home Price')
plt.title('Linear Regression Graph')
plt.show()

