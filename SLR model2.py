#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[5]:


data=pd.read_csv("Salary_Data.csv")
data.head()


# In[6]:


data.columns=["Experience","Salary"]


# In[7]:


data.shape


# In[8]:


data.info()


# In[11]:


import matplotlib.pyplot as plt


# In[12]:


plt.plot(data.Experience,data.Salary,"bo");
plt.xlabel("Experience");
plt.ylabel("Salary");


# In[13]:


data.corr()


# In[14]:


data.describe()


# In[15]:


import seaborn as sns


# In[16]:


sns.displot(data['Experience'])


# In[17]:


sns.displot(data["Salary"])


# In[18]:


sns.regplot(x="Experience",y="Salary",data=data);


# In[19]:


import statsmodels.formula.api as smf
model=smf.ols("Experience~Salary",data=data).fit()


# In[20]:


model.summary()


# In[21]:


model.params


# In[22]:


print(model.tvalues,'\n',model.pvalues)


# In[23]:


print(model.rsquared,model.rsquared_adj)


# In[ ]:




