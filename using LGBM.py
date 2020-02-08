#!/usr/bin/env python
# coding: utf-8

# ## DSN  : Predicitve Model Notebook [Using Light GBM]
# **Author**:🧕🏿 Hasanat Owoseni \
# **Date** : 14th October, 2019

# In[20]:


import numpy as np
import pandas as pd
import pickle
get_ipython().run_line_magic('matplotlib', 'inline')


# In[21]:


test_df = pd.read_csv('cleaned_test.csv')
train_df = pd.read_csv('cleaned_train.csv')


# In[22]:


test_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
train_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
train_df.info()
test_df.info()


# In[23]:


df_test = pd.read_csv('test.csv')
df_train = pd.read_csv('train.csv')


# In[24]:


columns = [column for column in test_df.columns if column != 'employeeno']
test_x = test_df[columns].values.astype('float')

test_empid = df_test['EmployeeNo']
#what test x should be 
s_test_x = test_df.loc[: ,'trainings_attended':].values.astype('float')


# In[25]:


columns = [column for column in train_df.columns if column != ('promoted_or_not' ) and column !=( 'employeeno')]

X = train_df[columns].values.astype('float')
s_train = train_df[columns]

y = train_df['promoted_or_not']
empid = df_train['EmployeeNo']


# In[26]:


print(test_x.shape)
print(X.shape)
print(s_test_x.shape)
s_train.info()


# In[27]:


#train test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# --------------------- 

# In[28]:


import lightgbm as lgb


# In[29]:


lgb_model = lgb.LGBMClassifier(n_estimators=1000, max_depth=3, learning_rate=0.01, random_state=1)


# In[30]:


from sklearn.metrics import roc_auc_score


# In[31]:


lgb_model.fit(X_train, y_train)
pred_lgb = lgb_model.predict(X_test)
print(roc_auc_score(pred_lgb, y_test))


# In[32]:


lgb_model.fit(X_train, y_train)


# In[33]:


with open('lgbm.pkl', 'wb') as file:
    pickle.dump(lgb_model, file)


# In[34]:


ypred2 = lgb_model.predict(test_x)
ypred2[0:5]  # showing first 5 predictions
y_test[0:5]


# In[35]:


for i in range(0,16496):
    if ypred2[i]>=.5:       # setting threshold to .5
        ypred2[i]=1.0
    else:  
        ypred2[i]=0.0

from sklearn.metrics import accuracy_score


# In[36]:


ypred2[0:5]  # showing first 5 predictions


# In[37]:


df_submission = pd.DataFrame({'employeeno':df_test['EmployeeNo'], 'promoted_or_not':ypred2.astype('int64')})
df_submission.to_csv('set.csv', index = False)


# In[38]:


from sklearn.metrics import roc_auc_score


# In[ ]:





# In[ ]:




