#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import numpy as np
from pyvi import ViTokenizer
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB


# ## 1. Data Pre-processing
# 
# - Load trainning data: trainning.txt

# In[2]:


with open('trainning.txt', encoding='utf8') as file:
    data = file.read()


# In[3]:


data_split = data.split('\n')

data_split


# - Remove label from trainning

# In[4]:


label_data =  re.findall(r"(__\w+__)", data)

len(label_data)


# In[5]:


label_data


# - Vietnamese processing by ViTokenizer

# In[6]:


text_data =[]
for i in range(len(label_data)):
    e = data_split[i].replace(label_data[i], "")
    e = e.replace('\t', '')
    text_lower = e.lower()
    text_token = ViTokenizer.tokenize(text_lower)
    text_data.insert(i, text_token)
    
len(text_data)


# In[7]:


text_data


# - Stop words removing

# In[8]:


stop_word = []
with open("stopwords.txt",encoding="utf-8") as f :
    text = f.read()
    for word in text.split() :
      stop_word.append(word)
    f.close()
    punc = list(punctuation)
stop_word = stop_word + punc


# In[9]:


stop_word


# - Not alphabet removing

# In[10]:


training_data = []
for d in text_data:
  sent = []
  for word in d.split(" ") :
          if (word not in stop_word) :
              if ("_" in word) or (word.isalpha() == True):
                  sent.append(word)
  training_data.append(" ".join(sent))


# In[11]:


training_data


# ## 2. Extracting features
# - tfidf calculating

# In[12]:


tf = TfidfVectorizer(min_df=5,max_df=0.8,max_features=3000,sublinear_tf=True)
tf.fit(training_data)
X = tf.transform(training_data)


# In[13]:


X.shape


# ## 3. Classification algorithms | Naive Bayes - SVM

# - Convert label from str to number

# In[14]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
label_training = le.fit_transform(label_data)

len(label_training)


# In[15]:


len(set(label_training))


# In[16]:


label_training


# - Split data: train & validation

# In[17]:


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(training_data, label_training, test_size=0.2, random_state=38)


# In[18]:


len(X_train)


# - Naive Bayes: - Train model

# In[19]:


text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])

text_clf1 = text_clf.fit(X_train, y_train)


# In[20]:


text_clf1


# - Test accuracy: Naive Bayes

# In[21]:


text_clf.score(X_valid, y_valid)


# - SVM: - Training

# In[22]:


text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, max_iter=5, random_state=42))])

text_clf_svm = text_clf_svm.fit(X_train, y_train)


# - Test accuracy: SVM

# In[23]:


predicted_svm = text_clf_svm.predict(X_valid)
np.mean(predicted_svm == y_valid)


# ## 4. Classify

# - Load  data

# In[24]:


with open('testing.txt', encoding='utf8') as file:
    data1 = file.read()


# In[25]:


data_test_split = data1.split('\n')

len(data_test_split)


# In[26]:


data_test_split


# - Predict label

# In[27]:


predicted = text_clf.predict(data_test_split)

predicted_label = le.inverse_transform(predicted)

len(predicted_label)


# In[28]:


predicted_label


# - Write to file: predicts.txt

# In[29]:


with open('predicts.txt', 'w') as f:
    for item in predicted_label:
        f.write("%s\n" % item)


# - draft

# In[30]:


# text_data1 =[]
# for i in range(len(data_test_split)):
# #     e = data_test_split[i].replace(label_data[i], "")
# #     e = e.replace('\t', '')
#     text_lower = data_test_split[i].lower()
#     text_token = ViTokenizer.tokenize(text_lower)
#     text_data1.insert(i, text_token)
    
# len(text_data1)

# test_data1 = []
# for d in text_data1:
#   sent = []
#   for word in d.split(" ") :
#           if (word not in stop_word) :
#               if ("_" in word) or (word.isalpha() == True):
#                   sent.append(word)
#   test_data1.append(" ".join(sent))

# test_data1


# predicted = text_clf.predict(test_data1)

# predicted[100]

