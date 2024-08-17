#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[166]:


# Import data
Loans_df = pd.read_csv("Loan_Data.csv")


# In[100]:


Loans_df.head()


# In[101]:


Loans_df.info()


# In[102]:


Loans_df.describe()


# In[103]:


Loans_df.tail(10)


# In[104]:


Loans_df['default'].value_counts()


# # Exploratory Data Analysis
# 
# 
# Using quick visualisations, we can explore the relationship between different variables in the dataset.
# 
# Let's start with a dual histogram of the FICO score of the borrowers, depending on the deafault (i.e. if a borrower defaults).

# In[7]:


Loans_df['fico_score'].hist(bins=30,alpha=0.6,label='default=1')
plt.xlabel('FICO')


# In[8]:


Loans_df[Loans_df['default']==1]['fico_score'].hist(bins=30,alpha=0.6,label='default=1')
Loans_df[Loans_df['default']==0]['fico_score'].hist(bins=30,alpha=0.6,label='default=0')
plt.legend()
plt.xlabel('FICO')


# In[9]:


Loans_df[Loans_df['default']==1]['income'].hist(bins=30,alpha=0.6,label='default=1')
Loans_df[Loans_df['default']==0]['income'].hist(bins=30,alpha=0.6,label='default=0')
plt.legend()
plt.xlabel('income')


# In[10]:


sns.jointplot(x='fico_score',y='income',data=Loans_df)


# In[11]:


Loans_df[Loans_df['default']==1]['loan_amt_outstanding'].hist(bins=30,alpha=0.6,label='default=1')
Loans_df[Loans_df['default']==0]['loan_amt_outstanding'].hist(bins=30,alpha=0.6,label='default=0')
plt.legend()
plt.xlabel('loan_amt_outstanding')


# In[12]:


Loans_df[Loans_df['fico_score']>=600]['income'].hist(bins=30,alpha=0.6,label='fico_score >= 600')
Loans_df[Loans_df['fico_score']<600]['income'].hist(bins=30,alpha=0.6,label='fico_score < 600')
plt.legend()
plt.xlabel('income')


# In[13]:


Loans_df[Loans_df['fico_score']>=600]['loan_amt_outstanding'].hist(bins=30,alpha=0.6,label='fico_score >= 600')
Loans_df[Loans_df['fico_score']<600]['loan_amt_outstanding'].hist(bins=30,alpha=0.6,label='fico_score < 600')
plt.legend()
plt.xlabel('loan_amt_outstanding')


# In[14]:


#emp_length variable
df = Loans_df.groupby('years_employed')['years_employed'].count().reset_index(name='count').sort_values('years_employed',ascending=True)
fig, ax = plt.subplots()
fig.set_size_inches(14,6)
q = sns.barplot(x="years_employed", y="count", data=df)
q.set_xticklabels(q.get_xticklabels(),rotation=30)
plt.title('Total Employee Length')


# In[15]:


#credit_lines_outstanding variable
df = Loans_df.groupby('credit_lines_outstanding')['credit_lines_outstanding'].count().reset_index(name='count').sort_values('credit_lines_outstanding',ascending=True)
fig, ax = plt.subplots()
fig.set_size_inches(14,6)
q = sns.barplot(x="credit_lines_outstanding", y="count", data=df)
q.set_xticklabels(q.get_xticklabels(),rotation=30)
plt.title('credit lines outstanding ')


# In[16]:


plt.figure(figsize=(9,5))
sns.countplot(x=Loans_df['years_employed'],hue=Loans_df['default'])
plt.tight_layout()


# In[17]:


plt.figure(figsize=(9,5))
sns.countplot(x=Loans_df['credit_lines_outstanding'],hue=Loans_df['default'])
plt.tight_layout()


# In[18]:


Loans_df.groupby('credit_lines_outstanding')['default'].count().reset_index(name='count').sort_values('credit_lines_outstanding',ascending=True)


# In[105]:


Loans_df['ratio_income_Totdebt'] = Loans_df['income']/Loans_df['total_debt_outstanding']
Loans_df['ratio_income_loan_amt'] = Loans_df['income']/Loans_df['loan_amt_outstanding']


# In[106]:


Loans_df


# In[21]:


Loans_df[Loans_df['default']==1]['ratio_income_Totdebt'].hist(bins=30,alpha=0.6,label='default=1')
Loans_df[Loans_df['default']==0]['ratio_income_Totdebt'].hist(bins=30,alpha=0.6,label='default=0')
plt.legend()
plt.xlabel('ratio_income_Totdebt')


# In[22]:


Loans_df[Loans_df['default']==1]['ratio_income_loan_amt'].hist(bins=30,alpha=0.6,label='default=1')
Loans_df[Loans_df['default']==0]['ratio_income_loan_amt'].hist(bins=30,alpha=0.6,label='default=0')
plt.legend()
plt.xlabel('ratio_income_loan_amt')


# In[23]:


Loans_df[Loans_df['fico_score']>=600]['ratio_income_Totdebt'].hist(bins=30,alpha=0.6,label='fico_score >= 600')
Loans_df[Loans_df['fico_score']<600]['ratio_income_Totdebt'].hist(bins=30,alpha=0.6,label='fico_score < 600')
plt.legend()
plt.xlabel('ratio_income_Totdebt')


# In[24]:


Loans_df[Loans_df['fico_score']>=600]['ratio_income_loan_amt'].hist(bins=30,alpha=0.6,label='fico_score >= 600')
Loans_df[Loans_df['fico_score']<600]['ratio_income_loan_amt'].hist(bins=30,alpha=0.6,label='fico_score < 600')
plt.legend()
plt.xlabel('ratio_income_loan_amt')


# In[25]:


df=Loans_df.groupby(['credit_lines_outstanding'])['total_debt_outstanding'].mean().reset_index(name='mean').sort_values('credit_lines_outstanding',ascending=True)
fig, ax = plt.subplots()
fig.set_size_inches(14,6)
q = sns.barplot(x="credit_lines_outstanding", y="mean", data=df)
q.set_xticklabels(q.get_xticklabels(),rotation=30)
plt.title('credit_lines_outstanding to mean total debt')


# In[26]:


df=Loans_df.groupby(['credit_lines_outstanding'])['ratio_income_Totdebt'].mean().reset_index(name='mean').sort_values('credit_lines_outstanding',ascending=True)
fig, ax = plt.subplots()
fig.set_size_inches(14,6)
q = sns.barplot(x="credit_lines_outstanding", y="mean", data=df)
q.set_xticklabels(q.get_xticklabels(),rotation=30)
plt.title('credit_lines_outstanding to mean ratio income/Total debt')


# In[27]:


df=Loans_df.groupby(['credit_lines_outstanding'])['ratio_income_loan_amt'].mean().reset_index(name='mean').sort_values('credit_lines_outstanding',ascending=True)
fig, ax = plt.subplots()
fig.set_size_inches(14,6)
q = sns.barplot(x="credit_lines_outstanding", y="mean", data=df)
q.set_xticklabels(q.get_xticklabels(),rotation=30)
plt.title('credit_lines_outstanding to mean ratio income/loan amount')


# In[28]:


df=Loans_df.groupby(['credit_lines_outstanding'])['fico_score'].mean().reset_index(name='mean').sort_values('credit_lines_outstanding',ascending=True)
fig, ax = plt.subplots()
fig.set_size_inches(14,6)
q = sns.barplot(x="credit_lines_outstanding", y="mean", data=df)
q.set_xticklabels(q.get_xticklabels(),rotation=30)
plt.title('credit_lines_outstanding to mean fico score')


# # Correlation in Credit Features

# In[29]:


# Sample figsize in inches
fig, ax = plt.subplots(figsize=(20,10))

# Imbalanced DataFrame Correlation
corr = Loans_df.corr()
sns.heatmap(corr, cmap='YlGnBu', annot_kws={'size':30}, ax=ax)
ax.set_title("Correlation Matrix", fontsize=14)
plt.show()


# # Create predictions of probability for loan status using test data

# In[50]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, roc_auc_score


# In[31]:


X = Loans_df.drop(['default'], axis=1)


# In[32]:


Y = Loans_df['default']


# In[33]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.6, random_state=123)


# In[34]:


# Create, train, and fit a logistic regression model
from sklearn.linear_model import LogisticRegression
clf_logistic = LogisticRegression(solver='lbfgs').fit(X_train, np.ravel(Y_train))


# In[35]:


# Create predictions of probability for loan status using test data
# .predict_proba creates an array of probabilities of default: [[non-defualt-0, default-1]]
lr_preds = clf_logistic.predict_proba(X_test)


# In[36]:


# # Create dataframes of predictions and true labels
lr_preds_df = pd.DataFrame(lr_preds[:,1][0:], columns = ['lr_pred_PD'])
true_df = Y_test


# In[37]:


# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), lr_preds_df], axis = 1))


# In[38]:


import math


# In[39]:


lr_preds_df.round(decimals=2).value_counts()


# In[40]:


Y_test.value_counts()


# In[41]:


# Reassign loan status based on the threshold and print the predictions
lr_preds_df['lr_pred_loan_status_60'] = lr_preds_df['lr_pred_PD'].apply(lambda x: 1 if x > 0.60 else 0)
print("Non-Default / Default predictions at 60% Threshhold: ")
print(lr_preds_df['lr_pred_loan_status_60'].value_counts())


# In[42]:


# Print the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion Matrix at 60% Threshhold: ")
print(confusion_matrix(Y_test, lr_preds_df['lr_pred_loan_status_60']))


# In[43]:


# Print the classification report
from sklearn.metrics import classification_report
target_names = ['Non-Default', 'Default']
print(classification_report(Y_test, lr_preds_df['lr_pred_loan_status_60'], target_names=target_names))


# In[44]:


# Print the accuracy score the model
print(clf_logistic.score(X_test, Y_test))


# In[45]:


# Plot the ROC curve of the probabilities of default
from sklearn.metrics import roc_curve


# In[46]:


lr_prob_default = lr_preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(Y_test, lr_prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Chart for LR on PD")
plt.xlabel("Fall-out")
plt.ylabel("Sensitivity")
plt.show()


# In[48]:


lr_preds_df.round(decimals=2)


# In[107]:


Loans_df.head()


# In[108]:


Loans_df.drop(['default'], axis=1,inplace=True)


# In[109]:


Loans_df


# In[110]:


Prob_to_default = clf_logistic.predict_proba(Loans_df)


# In[111]:


prob_data = pd.DataFrame(Prob_to_default[:,1][0:].round(decimals = 2), columns = ['Probability _to_Default'])


# In[112]:


Loans_df=pd.concat([Loans_df, prob_data], axis = 1)


# In[113]:


Loans_df


# Assuming a recovery rate of 10%, this can be used to give the expected loss on a loan.

# In[121]:


# Recovery rate
recovery_rate = 0.1  


# In[122]:


# Loss Given Default (LGD)
Loans_df['lgd'] = (1 - recovery_rate) * Loans_df['loan_amt_outstanding']


#  Expected Loss (EL)

# In[125]:


Loans_df['expected_loss'] = Loans_df['Probability _to_Default'] * Loans_df['lgd']


# In[127]:


Loans_df.tail()


# In[129]:


Loans_df.sort_values('expected_loss',ascending=False)


# # Creating, Training, and Fitting a XGBoost Model to Oversampled Data

# In[130]:


import xgboost as xgb


# In[131]:


# Import data
Loans_df = pd.read_csv("Loan_Data.csv")


# In[132]:


X = Loans_df.drop(['default'], axis=1)


# In[133]:


Y = Loans_df['default']


# In[134]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3, random_state=123)


# In[135]:


model = xgb.XGBClassifier()
clf_xgbt = model.fit(X_train, np.ravel(Y_train))


# In[144]:


# Create predictions of probability for loan status using test data
# .predict_proba creates an array of probabilities of default: [[non-defualt-0, default-1]]
xgbt_preds = clf_xgbt.predict_proba(X_test)


# In[145]:


# Create dataframes of predictions and labels
xgbt_preds_df = pd.DataFrame(xgbt_preds[:,1][0:], columns = ['xgbt_pred_PD'])
true_df = Y_test


# In[146]:


# Concatenate and print the two data frames for comparison
print(pd.concat([true_df.reset_index(drop = True), xgbt_preds_df], axis = 1))


# In[148]:


# Reassign loan status based on the threshold and print the predictions
xgbt_preds_df['xgbt_pred_loan_status_60'] = xgbt_preds_df['xgbt_pred_PD'].apply(lambda x: 1 if x > 0.60 else 0)
print("Non-Default / Default  predictions at 60% Threshhold: ")
print(xgbt_preds_df['xgbt_pred_loan_status_60'].value_counts())

# Print the confusion matrix
print("Confusion Matrix at 60% Threshhold: ")
print(confusion_matrix(Y_test, xgbt_preds_df['xgbt_pred_loan_status_60']))

# Print the classification report
target_names = ['Non-Default', 'Default']
print(classification_report(Y_test, xgbt_preds_df['xgbt_pred_loan_status_60'], target_names=target_names))


# In[149]:


# Print the accuracy score the model
print(clf_xgbt.score(X_test, Y_test))

# Plot the ROC curve of the probabilities of default
from sklearn.metrics import roc_curve

xgb_prob_default = xgbt_preds[:, 1]
fallout, sensitivity, thresholds = roc_curve(Y_test, xgb_prob_default)
plt.plot(fallout, sensitivity, color = 'darkorange')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.title("ROC Chart for XGBoost on PD")
plt.xlabel("Fall-out")
plt.ylabel("Sensitivity")
plt.show()

# Compute the AUC and store it in a variable
from sklearn.metrics import roc_auc_score

xgb_auc = roc_auc_score(Y_test, xgb_prob_default)


# In[150]:


xgbt_preds_df


# In[153]:


Loans_df['default'].value_counts()


# In[154]:


Loans_df.drop(['default'], axis=1,inplace=True)


# In[155]:


Prob_to_default = clf_xgbt.predict_proba(Loans_df)


# In[159]:


prob_data = pd.DataFrame(Prob_to_default[:,1][0:].round(decimals = 2), columns = ['Probability _to_Default'])


# In[160]:


prob_data['Probability _to_Default'].apply(lambda x: 1 if x > 0.60 else 0).value_counts()


# In[161]:


Loans_df=pd.concat([Loans_df, prob_data], axis = 1)


# In[162]:


# Recovery rate
recovery_rate = 0.1  


# In[163]:


# Loss Given Default (LGD)
Loans_df['lgd'] = (1 - recovery_rate) * Loans_df['loan_amt_outstanding']


# In[164]:


# Expected Loss

Loans_df['expected_loss'] = Loans_df['Probability _to_Default'] * Loans_df['lgd']


# In[165]:


Loans_df.sort_values('expected_loss',ascending=False)


# In[ ]:




