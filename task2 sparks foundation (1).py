#!/usr/bin/env python
# coding: utf-8

# ## TASK-2

# In this regression task given to me, I have to  predict the optimum number of clusters have to represent it visually.
# 
# This is a unsupervised ML model.I used KMeans clustering algorithm
# 
# 
# Dataset : https://bit.ly/3kXTdox 

# In[26]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


# In[27]:


# read the Iris dataset using read_csv
Iris = pd.read_csv("iris data.csv")
Iris.head()


# In[28]:


# Take all rows and  first 4columns except species column and assign it to Iris1
Iris1=Iris.iloc[:,[1,2,3,4]]


# In[29]:


Iris1


# In[30]:


# user defined normalization function for normalizing the values in Iris1 dataset,It will give you the values between 0 and 1
def norm_func(i):
    x = (i-i.min()) / (i.max() - i.min())
    return (x)


# In[31]:


#Apply the normalization function to Iris1 dataset
df_norm = norm_func(Iris1)


# In[32]:


df_norm.columns


# In[33]:


#df_norm.drop('Id',inplace=True,axis=1)


# In[34]:


df_norm.head(10)


# In[35]:


from sklearn.cluster import KMeans
fig = plt.figure(figsize=(10, 8))
WCSS = []
for i in range(1, 11):
    clf = KMeans(n_clusters=i)
    clf.fit(df_norm)
    WCSS.append(clf.inertia_) # inertia is another name for WCSS
plt.plot(range(1, 11), WCSS)
plt.title('The Elbow Method')
plt.ylabel('WCSS')
plt.xlabel('Number of Clusters')
plt.show()


# In[36]:


# we cannot measure slope value after 3 or after k=3 there is no slope,within cluster similarity decreases when k value increases
clf = KMeans(n_clusters=3)
# Applying KMeans to our normalized dataset
y_kmeans = clf.fit_predict(df_norm)


# In[37]:


y_kmeans
#clf.cluster_centers_
clf.labels_


# In[38]:


md=pd.Series(y_kmeans)  # converting numpy array into pandas series object 
Iris['clust']=md # creating a  new column and assigning it to new column 
Iris


# In[39]:


Iris1.iloc[:,:-2].groupby(Iris.clust).mean()


# In[40]:


Iris1.plot(x="PetalLengthCm",y ="PetalWidthCm",c=clf.labels_,kind="scatter",s=50 ,cmap=plt.cm.coolwarm) 
plt.title('Clusters using KMeans')


# In[41]:


Iris1.plot(x="PetalLengthCm",y ="PetalWidthCm",c=y_kmeans ,kind="scatter",s=50 ,cmap=plt.cm.copper_r) 
plt.title('Clusters using KMeans')  


# ### Author: ARUNA YELLAPU

# In[ ]:




