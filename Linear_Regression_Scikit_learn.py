import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt 
import seaborn as sns

%matplotlib inline

df = pd.read_csv('USA_Housing.csv')

df.head()

df.info()

df.describe()

df.columns

sns.pairplot(df)
#Output for this line of code can be viewed at : https://tinyurl.com/y8qrm6pl

sns.distplot(df['Price'])
#Output for this line of code can be viewed at : https://tinyurl.com/yaaeax3o

df.corr()
sns.heatmap(df.corr())
#Output for this line of code can be viewed at : https://tinyurl.com/ybr3jb2u

sns.heatmap(df.corr(), annot=True)
#Output for this line of code can be viewed at : https://tinyurl.com/y9nb5cbx


df.columns

x = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]
       
y = df['Price']     


#Split the data

from sklearn.cross_validation import train_test_split


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.4, random_state=101)

#Now we have training and testing data
#Let's proceed further

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(x_train, y_train)
#Now LinearRegression model has trained 

#Evaluate our model
#Print the intercept

print(lm.intercept_)
#-2640159.79685


#Print the coefficient 

print(lm.coef_)
#[  2.15282755e+01   1.64883282e+05   1.22368678e+05   2.23380186e+03
#   1.51504200e+01]
#>>> 

cdf = pd.DataFrame(lm.coef_, x.columns, columns=['Coeff'])
cdf


#                                      Coeff
#Avg. Area Income                  21.528276
#Avg. Area House Age           164883.282027
#Avg. Area Number of Rooms     122368.678027
#Avg. Area Number of Bedrooms    2233.801864
#Area Population                   15.150420



#Now let's try some prediction

predictions = lm.predict(x_test)
predictions
#array([ 1260960.70567626,   827588.75560352,  1742421.24254328, ...,
#         372191.40626952,  1365217.15140895,  1914519.54178824])

#y_test contains correct prices 

plt.scatter(y_test, predictions)
#Output for this line of code can be viewed at : https://tinyurl.com/y8freepx

#Let's create a histogram distribution of residuals 
sns.distplot(y_test - predictions)
#Output of this line of code can be viewed at : https://tinyurl.com/y7qlaubu

#Regression evaluation matrix 

from sklearn import metrics

metrics.mean_absolute_error(y_test, predictions)
#82288.222519149567
#>>> 

metrics.mean_squared_error(y_test, predictions)
#10460958907.209501

np.sqrt(metrics.mean_squared_error(y_test, predictions))
#102278.82922291153

#We successfully performed linear regression using scikit learn library 
