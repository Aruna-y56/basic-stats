#!/usr/bin/env python
# coding: utf-8

# # predict delivery time using sorting time

# In[7]:


#dependent variable=delivery time
#independent variable=sorting time
#here we have only one dependent variable to predict so it is simple linear regression
#we all know equation of straight line Y=mx+c
#so we are going to build our regression model by importing neccessory libraries that we are work with


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[9]:


data=pd.read_csv("delivery_time.csv")
data.head()


# In[10]:


data.shape


# In[11]:


data.info()


# In[12]:


data.columns=["Delivery_time","Sorting_time"]


# 

# In[13]:


plt.plot(data.Delivery_time,data.Sorting_time,"bo");
plt.xlabel("Delivery_time");
plt.ylabel("Sorting_time")


# In[14]:


plt.hist(data.Delivery_time)


# In[15]:


plt.boxplot(data.Delivery_time)


# In[16]:


plt.hist(data.Sorting_time)


# In[17]:


plt.boxplot(data.Sorting_time)


# In[18]:


data.corr()


# In[19]:


data.describe()


# In[20]:


import seaborn as sns


# In[21]:


#now I can try to visualize my data in 2 plots
#distribution plot
#regression plot


# In[22]:


sns.displot(data["Delivery_time"])


# In[23]:


sns.regplot(x="Delivery_time",y="Sorting_time",data=data);


# In[24]:


import statsmodels.formula.api as smf


# In[25]:


model1=smf.ols("Delivery_time~Sorting_time",data=data).fit()


# In[26]:


model1.params


# In[27]:


model1.summary()


# In[28]:


(model1.rsquared,model1.rsquared_adj)


# In[29]:


pred1=model1.predict(data.iloc[:,1])


# In[30]:


pred1


# In[31]:


model1.resid
model1.resid_pearson


# In[35]:


pd.set_option("display.max_rows",None)
pred1


# In[36]:


rmse_lin = np.sqrt(np.mean((np.array(data["Delivery_time"])-np.array(pred1))**2))
rmse_lin 


# In[34]:


#though first one is not a good model going to build another model by using transformation


# In[42]:


model2=smf.ols("Delivery_time~np.log(Sorting_time)",data=data).fit()


# 

# In[43]:


model2.summary()


# In[44]:


(model2.rsquared,model2.rsquared_adj)


# In[45]:


model2.resid
model2.resid_pearson


# In[46]:


pred2=model2.predict(data.iloc[:,1])
pred2


# In[47]:


rmse_log = np.sqrt(np.mean((np.array(data["Delivery_time"])-np.array(pred2))**2))
rmse_log 


# In[48]:


pred2.corr(data.Delivery_time)


# In[52]:


model3=smf.ols('np.log(Delivery_time)~Sorting_time',data=data).fit()
model3.params
model3.summary()


# In[53]:


(model3.rsquared,model3.rsquared_adj)


# In[55]:


pred_log=model3.predict(pd.DataFrame(data["Sorting_time"]))
pred_log


# In[56]:


pred3=np.exp(pred_log)


# In[57]:


pred3


# In[58]:


rmse_exp = np.sqrt(np.mean((np.array(data["Delivery_time"])-np.array(pred3))**2))
rmse_exp


# In[ ]:


#quadratic model y=c+mx+mx2


# In[60]:


data["Sorting_sq"]=data.Sorting_time*data.Sorting_time
data


# In[61]:


model_quad=smf.ols("np.log(Delivery_time)~Sorting_time+Sorting_sq",data=data).fit()


# In[62]:


model_quad.params


# In[163]:


model_quad.summary()


# In[63]:


(model_quad.rsquared,model_quad.rsquared_adj)


# In[64]:


pred_quad=model_quad.predict(data)


# In[166]:


pred4=np.exp(pred_quad)
pred4


# In[66]:


rmse_quad = np.sqrt(np.mean((np.array(data["Delivery_time"])-np.array(pred_quad))**2))
rmse_quad


# In[68]:


pred_quad.corr(data.Delivery_time)


# In[70]:


data = {"MODEL":pd.Series(["rmse_lin","rmse_log","rmse_exp","rmse_quad"]),
        "RMSE_Values":pd.Series([rmse_lin,rmse_log,rmse_exp,rmse_quad]),
        "Rsquare":pd.Series([model1.rsquared,model2.rsquared,model3.rsquared,model_quad.rsquared])}
table=pd.DataFrame(data)
table 


# In[ ]:




