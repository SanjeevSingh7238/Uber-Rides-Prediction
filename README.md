# Uber-Rides-Prediction
# Import Essential libraries
import numpy as np
import pandas as pd
# saperate dataset
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,4].values
# split dataset into training dataset and testing dataset
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
import pickle
# Test the model
model=pickle.load(open('taxi.pkl','rb'))
model.predict([[80,1770000,6000,85]])[0].round(2)
