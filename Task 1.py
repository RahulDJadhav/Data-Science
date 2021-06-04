#!/usr/bin/env python
# coding: utf-8

# # Data Science and Business Analytics (GRIP June21)
# 
# 

# ## Task 1: Prediction using supervised ML

# ### Author : Rahul Jadhav
# #### Problem Statement
# #### What will be predicted score if a student studies for 9.25 hrs/ day?

# #### First we import necessary librarie

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# #### Import Dataset

# In[5]:


data=pd.read_csv('C:/Users/Devil/Downloads/student_scores.csv')
print(data)


# In[13]:


data.head()


# In[10]:


data.tail()


# In[16]:


# Complete Uderstanding of Data


data.describe()


# In[5]:


data.shape


# In[6]:


data.info()


# #### Plotting the Data

# In[7]:


data.plot(x='Hours',y='Scores',style='o',c='orange')
plt.title('Hours vs Score')
plt.xlabel('Hours studied')
plt.ylabel('Score obtained')
plt.show()


# In[8]:


data.corr()


# #### From above plot, it is concluded that there is a linear relationship between the hours and score. 

# #### Cleaning the data

# In[14]:


data.isnull()


# In[10]:


hrs=(data['Hours'].values).reshape(-1,1)
scr=data['Scores'].values


# In[11]:


print(hrs)


# In[12]:


print(scr)


# In[13]:


from sklearn.model_selection import train_test_split
hrs_train,hrs_test,scr_train,scr_test=train_test_split(hrs,scr,test_size=0.2,random_state=0)

#Splitting Done


# In[14]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(hrs_train,scr_train)
print('Training Done')


# In[15]:


print('Intercept value = ',reg.intercept_)
print('Linear coefficient = ',reg.coef_)


# ##### Plotting the regression line

# In[16]:


line = reg.coef_*hrs+reg.intercept_

plt.scatter(hrs, scr, c='orange')
plt.title('Linear regression vs trained model')
plt.xlabel('Hours studied')
plt.ylabel('Score obtained')
plt.plot(hrs, line);
plt.show()


# In[17]:


scr_pred = reg.predict(hrs_test)
scr_pred


# #### Comparing Actual value vs Predicated value

# In[18]:


d=pd.DataFrame({'Actual':scr_test,'Predicted':scr_pred})
d


# In[19]:


plt.scatter(scr_test,scr_pred,c='orange')
plt.show()


# #### Solution for given problem statement
# ##### Problem statement : What will be predicted score if student studies for 9.25 hrs a day

# In[20]:


hours=9.25
pred_score=reg.predict([[hours]])
print("Number of Hours = {}".format(hours))
print("Predicted Score = {}".format(pred_score[0]))


# In[21]:


from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(scr_test, scr_pred))


# ### Conclusion :

# #### We used a Linear Regression Model to predict the score of a  student if student study for 9.25 Hrs a day, Model predicts score is 93.6917
