#!/usr/bin/env python
# coding: utf-8

# ## TASK-1

# In this regression task given to me, I will predict the percentage of marks that a student is expected to score based upon the hours they studied.
# 
# This is a simple linear regression task as it involves just two variable.
# 
# Data can be at http://bit.ly/w-data

# In[1]:


# Importing libraries to work on data

import pandas as pd
import numpy as np


# In[2]:


# Loading the data from the URL into the data frame

url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"

try:
    data = pd.read_csv(url)
    print("Data Imported Successfully !!")
except:
    print("Data Error!")


# In[3]:


print(data.shape)


# In[4]:


data


# In[5]:


# Getting the Stats of the numerical columns using describe()

data.describe()


# In[6]:


#Getting an overall overview of the data using the info()

data.info()


# In[7]:


print(data.isna().sum())


# In[8]:


#Plotting the distribution of scores.

import matplotlib.pyplot as plt


# In[9]:


#Plotting the graph for the required data.

data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Scorre Received')  
plt.xlabel('Hours Studied')  
plt.ylabel('Score')  
plt.show()


# In[10]:


data.corr()


# In[11]:


X = data.iloc[:, :-1].values  
Y = data.iloc[:, 1].values


# In[12]:


print(Y)


# In[13]:


print(X)


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.40, random_state = 0)

# Splits the given data into test and train data sets based on the test_size mentioned.


# In[15]:


X_train, Y_train # Prints the training data used from the given data


# In[16]:


X_test, Y_test # Prints the testing data used from the given data


# In[17]:


from sklearn.linear_model import LinearRegression  
model = LinearRegression()  
model.fit(X_train, Y_train) 


# In[18]:


line = model.coef_*X + model.intercept_


# In[19]:


#Plotting for the test data

plt.scatter(X, Y)
plt.plot(X, line, color='red');
plt.show()


# In[20]:


Y_pred = model.predict(X_test) # Predicting the scores
print(Y_pred)


# In[21]:


df = pd.DataFrame({'Actual': Y_test, 'Predicted': Y_pred})  
display(df)


# In[22]:


df.plot()


# In[23]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_pred))


# In[24]:


hours = [[9.25]] #Input the value as a 2d Array

own_pred = model.predict(hours) # Calling the required function to predict
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# ### Author: ARUNA YELLAPU
