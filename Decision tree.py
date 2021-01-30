#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/COVID-19-Surveillance
# dataset link : http://archive.ics.uci.edu/ml/datasets/COVID-19+Surveillance
# email : amirsh.nll@gmail.com


# ### <p style=color:blue>Decision Tree for Divorce Predictors Data Set </p>
# 

# #### The Dataset
# The Dataset is from UCIMachinelearning and it provides you all the relevant information needed for the prediction of Divorce. It contains 54 features and on the basis of these features we have to predict that the couple has been divorced or not. Value 1 represent Divorced and value 0 represent not divorced. Features are as follows:
# 1. If one of us apologizes when our discussion deteriorates, the discussion ends.
# 2. I know we can ignore our differences, even if things get hard sometimes.
# 3. When we need it, we can take our discussions with my spouse from the beginning and correct it.
# 4. When I discuss with my spouse, to contact him will eventually work.
# 5. The time I spent with my wife is special for us.
# 6. We don't have time at home as partners.
# 7. We are like two strangers who share the same environment at home rather than family.
# 8. I enjoy our holidays with my wife.
# 9. I enjoy traveling with my wife.
# 10. Most of our goals are common to my spouse.
# 11. I think that one day in the future, when I look back, I see that my spouse and I have been in harmony with each other.
# 12. My spouse and I have similar values in terms of personal freedom.
# 13. My spouse and I have similar sense of entertainment.
# 14. Most of our goals for people (children, friends, etc.) are the same.
# 15. Our dreams with my spouse are similar and harmonious.
# 16. We're compatible with my spouse about what love should be.
# 17. We share the same views about being happy in our life with my spouse
# 18. My spouse and I have similar ideas about how marriage should be
# 19. My spouse and I have similar ideas about how roles should be in marriage
# 20. My spouse and I have similar values in trust.
# 21. I know exactly what my wife likes.
# 22. I know how my spouse wants to be taken care of when she/he sick.
# 23. I know my spouse's favorite food.
# 24. I can tell you what kind of stress my spouse is facing in her/his life.
# 25. I have knowledge of my spouse's inner world.
# 26. I know my spouse's basic anxieties.
# 27. I know what my spouse's current sources of stress are.
# 28. I know my spouse's hopes and wishes.
# 29. I know my spouse very well.
# 30. I know my spouse's friends and their social relationships.
# 31. I feel aggressive when I argue with my spouse.
# 32. When discussing with my spouse, I usually use expressions such as ‘you always’ or ‘you never’ .
# 33. I can use negative statements about my spouse's personality during our discussions.
# 34. I can use offensive expressions during our discussions.
# 35. I can insult my spouse during our discussions.
# 36. I can be humiliating when we discussions.
# 37. My discussion with my spouse is not calm.
# 38. I hate my spouse's way of open a subject.
# 39. Our discussions often occur suddenly.
# 40. We're just starting a discussion before I know what's going on.
# 41. When I talk to my spouse about something, my calm suddenly breaks.
# 42. When I argue with my spouse, ı only go out and I don't say a word.
# 43. I mostly stay silent to calm the environment a little bit.
# 44. Sometimes I think it's good for me to leave home for a while.
# 45. I'd rather stay silent than discuss with my spouse.
# 46. Even if I'm right in the discussion, I stay silent to hurt my spouse.
# 47. When I discuss with my spouse, I stay silent because I am afraid of not being able to control my anger.
# 48. I feel right in our discussions.
# 49. I have nothing to do with what I've been accused of.
# 50. I'm not actually the one who's guilty about what I'm accused of.
# 51. I'm not the one who's wrong about problems at home.
# 52. I wouldn't hesitate to tell my spouse about her/his inadequacy.
# 53. When I discuss, I remind my spouse of her/his inadequacy.
# 54. I'm not afraid to tell my spouse about her/his incompetence.

# Generally, logistic Machine Learning in Python has a straightforward and user-friendly implementation. It usually consists of these steps:<br>
# 1. Import packages, functions, and classes<br>
# 2. Get data to work with and, if appropriate, transform it<br>
# 3. Create a classification model and train (or fit) it with existing data<br>
# 4. Evaluate your model to see if its performance is satisfactory<br>
# 5. Apply your model to make predictions<br>

# #### Import packages, functions, and classes

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import accuracy_score 
from sklearn import tree


# #### Get data to work with and, if appropriate, transform it

# In[2]:


df = pd.read_csv('divorce.csv',sep=';')
y=df.Class
x_data=df.drop(columns=['Class'])
df.head(10)


# #### Data description

# In[3]:


sns.countplot(x='Class',data=df,palette='hls')
plt.show()

count_no_sub = len(df[df['Class']==0])
count_sub = len(df[df['Class']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no divorce is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub)
print("percentage of divorce", pct_of_sub*100)


# #### Normalize data

# In[4]:


x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)).values
x.head()


# #### correlation of all atribute

# In[5]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), cmap='viridis');


# #### Split data set 

# In[6]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.4,random_state=400)
print("x_train: ",x_train.shape)
print("x_test: ",x_test.shape)
print("y_train: ",y_train.shape)
print("y_test: ",y_test.shape)


# #### Create a classification model and train (or fit) it with existing data

# Step 1. Import the model you want to use<br>
# Step 2. Make an instance of the Model<br>
# Step 3. Training the model on the data, storing the information learned from the data<br>
# Step 4. Predict labels for new data <br>

# In[7]:


clft = DecisionTreeClassifier()
clft = clft.fit(x_train,y_train)
y_predt = clft.predict(x_test)# step 4


# #### Report

# In[8]:


print(classification_report(y_test, clft.predict(x_test)))
print('Accuracy of Decision Tree classifier on test set: {:.2f}'.format(clft.score(x_test, y_test)))


# #### Draw Decistion Tree

# In[9]:


plt.figure(figsize=(10,10))
temp = tree.plot_tree(clft.fit(x,y), fontsize=12)
plt.show()


# #### Confusion Matrix

# In[10]:


from sklearn.metrics import classification_report, confusion_matrix as cm
def confusionMatrix(y_pred,title,n):
    plt.subplot(1,2,n)
    ax=sns.heatmap(cm(y_test, y_pred)/sum(sum(cm(y_test, y_pred))), annot=True 
                   ,cmap='RdBu_r', vmin=0, vmax=0.52,cbar=False, linewidths=.5)
    plt.title(title)
    plt.ylabel('Actual outputs')
    plt.xlabel('Prediction')
    b, t=ax.get_ylim()
    ax.set_ylim(b+.5, t-.5)
    plt.subplot(1,2,n+1)
    axx=sns.heatmap(cm(y_test, y_pred), annot=True 
                   ,cmap='plasma', vmin=0, vmax=40,cbar=False, linewidths=.5)
    b, t=axx.get_ylim()
    axx.set_ylim(b+.5, t-.5)
    return

plt.figure(figsize=(8,6))
confusionMatrix(y_predt,'Decision Tree',1)
plt.show


# #### Result:
# So we have successfully trained our dataset into Decision Tree for predicting whether a couple will get divorced or not. And also got the accuracy & confusion matrix for Decision Tree
