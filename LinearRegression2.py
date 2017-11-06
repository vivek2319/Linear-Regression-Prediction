#Linear Regression project 2
#We are trying to predict whether e-commerce company should go for mobile app or website sales

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv("Ecommerce Customers")

customers.head()
#>>> customers.head()
#                           Email  \
#0      mstephenson@fernandez.com   
#1              hduke@hotmail.com   
#2               pallen@yahoo.com   
#3        riverarebecca@gmail.com   
#4  mstephens@davidson-herman.com   
#
#                                             Address            Avatar  \
#0       835 Frank Tunnel\nWrightmouth, MI 82180-9605            Violet   
#1     4547 Archer Common\nDiazchester, CA 06566-8576         DarkGreen   
#2  24645 Valerie Unions Suite 582\nCobbborough, D...            Bisque   
#3   1414 David Throughway\nPort Jason, OH 22070-1220       SaddleBrown   
#4  14023 Rodriguez Passage\nPort Jacobville, PR 3...  MediumAquaMarine   
#
#   Avg. Session Length  Time on App  Time on Website  Length of Membership  \
#0            34.497268    12.655651        39.577668              4.082621   
#1            31.926272    11.109461        37.268959              2.664034   
#2            33.000915    11.330278        37.110597              4.104543   
#3            34.305557    13.717514        36.721283              3.120179   
#4            33.330673    12.795189        37.536653              4.446308   
#
#   Yearly Amount Spent  
#0           587.951054  
#1           392.204933  
#2           487.547505  
#3           581.852344  
#4           599.406092  


customers.describe()

# 	Avg. Session Length 	Time on App 	Time on Website 	Length of Membership 	Yearly Amount Spent
#count 	500.000000 	        500.000000 	     500.000000 	     500.000000 	         500.000000
#mean 	33.053194 	        12.052488 	     37.060445 	         3.533462 	             499.314038
#std 	0.992563 	        0.994216 	     1.010489 	         0.999278  	             79.314782
#min 	29.532429 	        8.508152  	     33.913847           0.269901 	             256.670582
#25% 	32.341822 	        11.388153 	     36.349257 	         2.930450 	             445.038277
#50% 	33.082008 	        11.983231 	     37.069367 	         3.533975 	             498.887875
#75% 	33.711985 	        12.753850 	     37.716432 	         4.126502 	             549.313828
#max 	36.139662 	        15.126994 	     40.005182 	         6.922689 	             765.518462


customers.info()

#Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. 
#Does the correlation make sense?


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')


# More time on site, more money spent.
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)
#Output for this line of code can be viewed at : https://tinyurl.com/ybscl8h6

sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)
#Output for this line of code can be viewed at : https://tinyurl.com/y8b789za

sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)
#Output for this line of code can viwed at : https://tinyurl.com/y7nzj9ed

#Let's explore these types of relationships across the entire data set. 
#Use pairplot to recreate the plot below.(Don't worry about the the colors)
sns.pairplot(customers)

#Output for this line of code can be viewed at : https://tinyurl.com/ycezak2r

#Create a linear model plot (using seaborn's lmplot) of 
#Yearly Amount Spent vs. Length of Membership. 


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
#Output for this line of code can be viewed at : https://tinyurl.com/y7jj9aqd

#Training and Testing Data
#Now that we've explored the data a bit, let's go ahead and split the data into 
#training and testing sets. Set a variable X equal to the numerical features of 
#the customers and a variable y equal to the "Yearly Amount Spent" column.


y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]


#Use model_selection.train_test_split from sklearn to split the data into 
#training and testing sets. Set test_size=0.3 and r

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


#Training the Model

#Now its time to train our model on our training data!

#Import LinearRegression from sklearn.linear_model

from sklearn.linear_model import LinearRegression

#Create an instance of a LinearRegression() model named lm.

lm = LinearRegression()

#Train/fit lm on the training data.
lm.fit(X_train,y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)



#Print out the coefficients of the model
# The coefficients
print('Coefficients: \n', lm.coef_)
#Coefficients: 
# [ 25.98154972  38.59015875   0.19040528  61.27909654]


#Predicting Test Data
#Now that we have fit our model, 
#let's evaluate its performance by predicting off the test values!
#Use lm.predict() to predict off the X_test set of the data.

predictions = lm.predict( X_test)

#Create a scatterplot of the real test values versus the predicted values. 

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

#Output for this line of code can be viewed at : https://tinyurl.com/y9dldgcj

#Evaluating the Model

#Let's evaluate our model performance by calculating the residual 
#sum of squares and the explained variance score (R^2).

#Calculate the Mean Absolute Error, Mean Squared Error, and the 
#Root Mean Squared Error. 

# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#MAE: 7.22814865343
#MSE: 79.813051651
#RMSE: 8.93381506698

#Residuals

#Plot a histogram of the residuals and make sure it looks normally distributed.
#Use either seaborn distplot, or just plt.hist().

sns.distplot((y_test-predictions),bins=50)
#Output for this line of code can be viewed at : https://tinyurl.com/y9rqfsjb

#Conclusion

#We still want to figure out the answer to the original question, 
#do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important. Let's see if we can interpret the coefficients at all to get an idea.

#Recreate the dataframe below.

coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients

#                      Coeffecient
#Avg. Session Length     25.981550
#Time on App             38.590159
#Time on Website          0.190405
#Length of Membership    61.279097

#How can you interpret these coefficients? 

#Interpreting the coefficients:

#Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
#Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
#Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
#Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.

