#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv("D:/DS@ai-ml/M.Sc Data Sc/DATA set/heart.csv")


# In[4]:


df


# In[5]:


df.shape


# In[6]:


df.describe


# In[7]:


df.target


# In[8]:


df["target"]


# In[9]:


df["target"].value_counts()


# In[10]:


df["target"].value_counts().plot(kind="bar",color=['blue','red'])


# In[11]:


df.info()


# In[12]:


df.describe()


# In[13]:


df.sex.value_counts()


# In[14]:


pd.crosstab(df.target,df.sex)


# In[15]:


pd.crosstab(df.target,df.sex).plot(kind="bar",figsize=[10,8],color=['blue','orange'])
plt.title("Heart Disease Frequency For Sex")
plt.legend(['Female','Male'])
plt.xlabel("0=no disease,1= disease")
plt.ylabel("amount")
plt.xticks(rotation=True)


# In[16]:


df.corr()


# In[17]:


corr_matrix = df.corr()
fig,ax=plt.subplots(figsize=(15,10))
ax = sns.heatmap(corr_matrix)
ax=sns.heatmap(corr_matrix,annot=True,linewidths=.5,fmt=".2f",cmap="YlGnBu")
bottom,top=ax.get_ylim()
ax.set_ylim(bottom+.5,top-.5)


# In[18]:


df.head()


# In[19]:


x=df.drop("target",axis=1)


# In[20]:


y=df["target"]


# In[21]:


y


# In[22]:


np.random.seed()


# In[23]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# In[24]:


from sklearn.model_selection import train_test_split

X = df.drop('target', axis=1)
y = df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[25]:


from sklearn.linear_model import LogisticRegression

lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, y_train)

print_score(lr_clf, X_train, y_train, X_test, y_test, train=True)
print_score(lr_clf, X_train, y_train, X_test, y_test, train=False)


# In[26]:


test_score = accuracy_score(y_test, lr_clf.predict(X_test)) * 100
train_score = accuracy_score(y_train, lr_clf.predict(X_train)) * 100

results_df = pd.DataFrame(data=[["Logistic Regression", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results_df


# In[27]:


from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)

print_score(knn_clf, X_train, y_train, X_test, y_test, train=True)
print_score(knn_clf, X_train, y_train, X_test, y_test, train=False)


# In[ ]:




