#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


train_df = pd.DataFrame(pd.read_csv('train.csv'))
test_df = pd.DataFrame(pd.read_csv('test.csv'))


# In[3]:


train_df.head()


# In[4]:


#number of NaN values of each cols

for i in train_df.columns:
  print(i,"\t-\t", train_df[i].isna().mean()*100)


# In[5]:


#drop cabin cause of most Nan vals

train_df = train_df.drop(["Cabin"], axis=1)


# In[6]:


train_df['Age'].fillna(train_df['Age'].median(), inplace=True)#filling Nan values of Age
train_df['Embarked'].fillna(train_df['Embarked'].mode(), inplace=True)


# In[7]:


train_df.info()


# In[8]:


train_df = train_df.drop(["PassengerId", "Fare", "Ticket", "Name"], axis = 1)


# In[9]:


train_df.info()


# In[10]:


#str to number values

from sklearn.preprocessing import LabelEncoder

cat_col= train_df.drop(train_df.select_dtypes(exclude=['object']), axis=1).columns
print(cat_col)

enc1 = LabelEncoder()
train_df[cat_col[0]] = enc1.fit_transform(train_df[cat_col[0]].astype('str'))

enc2 = LabelEncoder()
train_df[cat_col[1]] = enc2.fit_transform(train_df[cat_col[1]].astype('str'))


# In[11]:


train_df.head()


# In[12]:


train_df.info()


# ## Training

# In[13]:


X = train_df.drop(['Survived'], axis=1)
y = train_df['Survived']


# In[14]:


#now lets split data in test train pairs

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


# In[30]:


# model training 

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()  
lr = model.fit(X_train, y_train)


# ## Test

# In[16]:


y_pred = model.predict(X_test)

pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
pred_df.head()


# ## To check Accuracy

# In[33]:


from sklearn import metrics

# Measure the Accuracy Score
print("Accuracy score of the predictions: {0}".format(metrics.accuracy_score(y_pred, y_test)))


# ## Prediction

# In[18]:


test_df.head()


# In[19]:


for i in train_df.columns:
  print(i,"\t-\t", train_df[i].isna().mean()*100)


# In[20]:


test_df = test_df.drop(["Cabin"], axis=1)

test_df['Age'].fillna(test_df['Age'].median(), inplace=True) #filling Nan values of Age
train_df['Embarked'].fillna(train_df['Embarked'].mode(), inplace=True)


# In[21]:


test_df.info()


# In[22]:


PassengerId = test_df["PassengerId"]

test_df = test_df.drop(["PassengerId", "Fare", "Ticket", "Name"], axis = 1)   


# In[23]:


test_df[cat_col[0]] = enc1.transform(test_df[cat_col[0]].astype('str'))

test_df[cat_col[1]] = enc2.transform(test_df[cat_col[1]].astype('str'))


# In[24]:


test_df.head()


# In[25]:


y_test_pred = model.predict(test_df)


# In[26]:


submission = pd.DataFrame({
        "PassengerId": PassengerId,
        "Survived": y_test_pred
    })


submission.to_csv('./submission.csv', index=False)


# In[27]:


final_res =  pd.DataFrame(pd.read_csv('submission.csv'))


# In[28]:


final_res.head()


# In[29]:


# number of row and cols of a dataframe
row, cols = final_res.shape

row, cols

