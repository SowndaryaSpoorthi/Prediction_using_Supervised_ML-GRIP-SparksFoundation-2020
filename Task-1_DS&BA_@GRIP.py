#!/usr/bin/env python
# coding: utf-8

# # GRIP @ THE SPARKS FOUNDATION- 2020
#  ## TASK 1: Prediction using Supervised ML
#  ###### To Predict the percentage of an student based on the no. of study hours.

# ### Author: Sowndarya Spoorthi.B

# In[85]:


#importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# ## Step1-  Reading the data from source

# In[86]:


# Reading the Data 
url= "http://bit.ly/w-data"
data = pd.read_csv('http://bit.ly/w-data')
data.head(20)


# In[87]:


print("Student Data imported successfully!")


# In[88]:


# Check if there any null value in the Dataset
data.isnull == True 


# Since there is no null value in the Dataset so, we can now visualize our Data

# ## step 2-Input data Visualization

# In[89]:


# Plotting the distribution of score
sns.set_style('darkgrid')
sns.scatterplot(y= data['Scores'], x= data['Hours'])
plt.title('Marks Vs Study Hours',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# ## Step 3 - Data Preprocessing
#     This step involved division of data into "attributes" (inputs) and "labels" (outputs).

# In[90]:


# Defining X and y from the Data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values


# ## Step4- Model Training
#  Splitting the data into training and testing sets, and training the algorithm.

# In[91]:


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
regression = LinearRegression()
regression.fit(train_X, train_y)
print("---------Model Trained---------")


# From the above scatter plot there looks to be correlation between the 'Marks Percentage' and 'Hours Studied', Lets plot a regression line to confirm the correlation.

# ## Step 5 - Plotting the Line of regression
#    Since the model is trained now, it's the time to visualize the best-fit line of regression.

# In[92]:


# Plotting the regression line
sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()
print(data.corr())


# ## step6- - Making Predictions
#     Now that this algorithm is trained, it's time to test the model by making some predictions.
# 
# For this we will use our test-set data
# 

# In[93]:


pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# ## Step7- Comparing Actual result to the Predicted Model result
# 

# In[94]:


# Comparing Actual vs Predicted
compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
compare_scores


# In[95]:


#Visually Comparing the Predicted Marks with the Actual Marks

plt.scatter(x=val_X, y=val_y, color='blue')
plt.plot(val_X, pred_y, color='green')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# In[105]:


# Plotting the Bar graph to depict the difference between the actual and predicted value
compare_scores.plot(kind='bar',figsize=(8,8))
plt.grid(which='major', linewidth='2.0', color='red')
plt.grid(which='minor', linewidth='2.0', color='blue')
plt.show()


# In[97]:


# Testing the model with our own data
hours = 9.25
test = np.array([hours])
test = test.reshape(-1, 1)
own_pred = regression.predict(test)
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ## Step8- Evaluating the model
#     
# The final step is to evaluate the performance of algorithm. This step is particularly important to compare how well different algorithms perform on a particular dataset.

# In[101]:


# Calculating the accuracy of the model
print('Mean absolute error: ',mean_absolute_error(val_y,pred_y))


# ## What will be the predicted score of a student if he/she studies for 9.25 hrs/ day?

# In[80]:


hours = [9.25]
answer = regression.predict([hours])
print("Score = {}".format(round(answer[0],6)))
print("Approximate Score = {}".format(round(answer[0],2)))


# ## From the above model it's predicted that, if a student studies for 9.25 hours a day he/she is likely to score 93.89 marks.

# ## THANK YOU
# 
