# Linear-Regression-Prediction
_Note: This repository focuses on the projects that I would be doing on "Linear Regression". Feel free to make any improvements.
The theory given here is taken from various sources over internet. Me as a owner of this project do not claim it's ownership in any whatsoevr way._


## Introduction :
Lets first know what we mean by Regression. Regression is a statistical way to establish a relationship between a dependent variable and a set of independent variable(s). e.g., if we say 

Age = 5 + Height * 10 + Weight * 13

Here we are establishing a relationship between Height & Weight of a person with his/ Her Age. This is a very basic example of Regression.
Simple Linear Regression

Least Square “Linear Regression” is a statistical method to regress the data with dependent variable having continuous values whereas independent variables can have either continuous or categorical values. In other words “Linear Regression” is a 
method to predict dependent variable (Y) based on values of independent variables (X).  It can be used for the cases where we 
want to predict some continuous quantity. 

![Linear Regression Explanation](http://ictedusrv.cumbria.ac.uk/maths/SecMaths/U4/images/pic111.gif)

Any line can be characterized by its intercept and slope. The slope is the change in y for a one-unit change in x. Also, an equivalent definition is the change in y divided by the change in x for any segment of the line. 
The equation of line is y = mx + c, where m is slope and c is intercept. By plugging different values for x into this equation 
we can find the corresponding y values that are on the line drawn. The vertical distances of the points from the line are called "residuals". The least square principle says that the best-fit line is the one with the smallest sum of squared residuals.  It is interesting to note that the sum of the residuals (not squared) is zero for the least-squares best-fit line.

The general meaning of a slope coefficient is the change in Y caused by a one-unit increase in x. It is very important to know in what units x are measured, so that the meaning of a one-unit increase can be clearly expressed. Every regression analysis should include a residual analysis as a further check on the adequacy of the chosen regression model.  Remember that there is a residual value for each data point.

A  positive  residual  indicates  a  data  point  higher  than  expected,  and  a  negative residual indicates a point lower than expected. A residual is the deviation of an outcome from the predicated mean value for all subjects with the same value for the explanatory variable. The residual vs.  fit plot can be used to detect non-linearity and / or unequal variance. A quantile normal plot of the residuals of a regression analysis can be used to detect non-Normality.

Regression is reasonably robust to the equal variance assumption.  Moderate degrees of violation, e.g., the band with the widest variation is up to twice as wide as the band with the smallest variation, tend to cause minimal problems. For more 
severe violations, the p-values are incorrect in the sense that their null hypotheses tend to be rejected more that 100 α
% of the time when the null hypothesis is true.

The confidence intervals (and the SE’s they are based on) are also incorrect.  For worrisome violations of the equal variance assumption, try transformations of the y variable (because the assumption applies at each x value,  transformation of x will be ineffective).

Regression is quite robust to the Normality assumption. You only need to worry about severe violations.  For markedly skewed or kurtotic residual distributions, we need to worry that the p-values and confidence intervals are incorrect.  In that
case try transforming they variable.  Also,  in the case of data with less than a handful of different y values or with severe truncation of the data (values piling up at the ends of a limited width scale), regression may be inappropriate due to
non-Normality.
