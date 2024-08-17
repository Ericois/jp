#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Load Libreries
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import pyplot
from scipy.stats import norm
import datetime as dt


# In[57]:


# Import data
Loans_df = pd.read_csv("Task 3 and 4_Loan_Data.csv")


# In[58]:


Loans_df


# In[59]:


Loans_df.info()


# In[60]:


Loans_df.describe()


# # Rating map

# Creating a rating map that maps the FICO score of the borrowers to a rating where a lower rating signifies a better credit score.
# The process of doing this is known as quantization.
# Creating six buckets for FICO scores ranging from 300 to 850. 

# In[61]:


d = {range(800,851):1,range(750,800):2,range(700,750):3,
     range(650,700):4,range(600,650):5,range(300,600):6 }


# In[62]:


d


# In[63]:


Loans_df['Rating'] = Loans_df['fico_score'].apply(lambda x: next((v for k, v in d.items() if x in k), 0))


# In[64]:


Loans_df.groupby('fico_score')['default']


# In[65]:


Loans_df


# In[66]:


plt.figure(figsize=(9,5))
sns.countplot(x=Loans_df['Rating'],hue=Loans_df['default'])
plt.tight_layout()


# In[67]:


Loans_df.groupby('Rating')['default'].value_counts().sort_values()


# In[68]:


Loans_df['ratio_income_Totdebt'] = Loans_df['income']/Loans_df['total_debt_outstanding']
Loans_df['ratio_income_loan_amt'] = Loans_df['income']/Loans_df['loan_amt_outstanding']


# In[69]:


Loans_df


# # Predicting the PD (probability of default) for the borrowers using new ratings 

# In[70]:


Loans_df[['default','Rating']]


# In[71]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, roc_auc_score


# In[72]:


X = Loans_df.drop(['default','fico_score','customer_id'], axis=1)
Y = Loans_df['default']
id_users = Loans_df['customer_id']


# In[73]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.7, random_state=123)


# In[74]:


# Create, train, and fit a logistic regression model
from sklearn.linear_model import LogisticRegression
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(Y_train))


# In[75]:


# Create predictions of probability for loan status using test data
# .predict_proba creates an array of probabilities of default: [[non-defualt-0, default-1]]
lr_preds = clf_logistic.predict_proba(X_test)


# In[76]:


# # Create dataframes of predictions and true labels
lr_preds_df = pd.DataFrame(lr_preds[:,1][0:], columns = ['lr_pred_PD'])
true_df = Y_test


# In[77]:


# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), lr_preds_df], axis = 1))


# In[78]:


# Reassign loan status based on the threshold and print the predictions
lr_preds_df['lr_pred_loan_status_60'] = lr_preds_df['lr_pred_PD'].apply(lambda x: 1 if x > 0.60 else 0)
print("Non-Default / Default predictions at 60% Threshhold: ")
print(lr_preds_df['lr_pred_loan_status_60'].value_counts())


# In[79]:


Y_test.value_counts()


# In[80]:


# Print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion Matrix at 60% Threshhold: ")
print(confusion_matrix(Y_test, lr_preds_df['lr_pred_loan_status_60']))


# In[81]:


# Print the classification report
from sklearn.metrics import classification_report
target_names = ['Non-Default', 'Default']
print(classification_report(Y_test, lr_preds_df['lr_pred_loan_status_60'], target_names=target_names))


# In[82]:


# Print the accuracy score the model
print(clf_logistic.score(X_test, Y_test))


# In[83]:


# Plot the ROC curve of the probabilities of default
from sklearn.metrics import roc_curve


# In[84]:


lr_prob_default = lr_preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(Y_test, lr_prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Chart for LR on PD")
plt.xlabel("Fall-out")
plt.ylabel("Sensitivity")
plt.show()


# #  Fitting the Model to entire dataset

# In[85]:


Prob_to_default = clf_logistic.predict_proba(X)


# In[86]:


prob_data = pd.DataFrame(Prob_to_default[:,1][0:].round(decimals = 2), columns = ['Probability _to_Default'])


# In[87]:


prob_data


# In[93]:


X = pd.concat([id_users,X,Y, prob_data], axis = 1)


# In[94]:


X


# # Predicting the PD (probability of default) for the borrowers using only new ratings

# In[35]:


X = Loans_df['Rating'].to_numpy().reshape(-1,1)


# In[38]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.7, random_state=123)


# In[40]:


# Create, train, and fit a logistic regression model
from sklearn.linear_model import LogisticRegression
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(Y_train))


# In[41]:


# Create predictions of probability for loan status using test data
# .predict_proba creates an array of probabilities of default: [[non-defualt-0, default-1]]
lr_preds = clf_logistic.predict_proba(X_test)


# In[42]:


# # Create dataframes of predictions and true labels
lr_preds_df = pd.DataFrame(lr_preds[:,1][0:], columns = ['lr_pred_PD'])
true_df = Y_test


# In[43]:


# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), lr_preds_df], axis = 1))


# In[45]:


# Print the accuracy score the model
print(clf_logistic.score(X_test, Y_test))


# In[46]:


lr_prob_default = lr_preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(Y_test, lr_prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Chart for LR on PD")
plt.xlabel("Fall-out")
plt.ylabel("Sensitivity")
plt.show()


# #  Fitting the Model to entire dataset

# In[47]:


Prob_to_default = clf_logistic.predict_proba(X)


# In[48]:


prob_data = pd.DataFrame(Prob_to_default[:,1][0:].round(decimals = 2), columns = ['Probability _to_Default'])


# In[52]:


X = pd.concat([Loans_df, prob_data], axis = 1)


# In[53]:


X


# Model built in this way clearly understimate risk of default for low fico scores or low income/total debt ratio , anyway
# it could be used to get a "at priori" probability of default for a specific rating category

# In[ ]:




