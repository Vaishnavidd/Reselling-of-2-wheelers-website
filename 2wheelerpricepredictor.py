# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 18:30:44 2019

@author: vaishnavi
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 23:07:05 2019

@author: vaishnavi
"""

from pandas import DataFrame
from sklearn import linear_model
import statsmodels.api as sm

Stock_Market = {'Year': [2014,2016,2014,2013,2013,2013,2017,2017,2015,2017],
                'Interest_Rate': [47000,12000,42000,20000,35000,22334,13000,28566,8888,15567],
                'Unemployment_Rate': [2013,2013,2013,2014,2014,2015,2016,2017,2017,2017],
      'Stock_Index_Price': [20000,22000,21000,24000,24100,25600,29600,30000,27500,29600]        
                }
df = DataFrame(Stock_Market,columns=['Year','Interest_Rate','Unemployment_Rate','Stock_Index_Price'])


x = df[['Interest_Rate','Unemployment_Rate']] # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
y = df['Stock_Index_Price']


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x=LabelEncoder();
x=labelencoder_x.fit_transform(x)

#cannot compare country values
onehotencoder=OneHotEncoder()
categorical_features=[0]
x=onehotencoder.fit_transform(x).toarray()

labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

#Splitting data into training and testing data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#Feature scaling
from sklearn.preprocessing import StandardScaler

#to convert in format 
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)
