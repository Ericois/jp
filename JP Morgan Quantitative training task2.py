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


# In[2]:


# Import data
Nat_gas_df = pd.read_csv("Nat_gas.csv")


# In[3]:


Nat_gas_df.head()


# In[4]:


# Convert Month into Datetime
Nat_gas_df['Dates']=pd.to_datetime(Nat_gas_df['Dates'])


# In[5]:


Nat_gas_df.head()


# In[6]:


Nat_gas_df.set_index('Dates',inplace=True)


# In[58]:


Nat_gas_df.head(18)


# In[8]:


inj_dates = ['2021-06-30','2021-07-31','2021-08-31'] # injection dates
inj_dates


# In[ ]:


#Nat_gas_df.loc[T]['Prices'][1]


# In[10]:


inj_prices = Nat_gas_df.loc[inj_dates]['Prices'] # Prices at time of injection
inj_prices


# In[11]:


wtdhr_dates = ['2021-12-31','2022-01-31','2022-02-28'] # Withdrawal  dates
wtdhr_dates


# In[12]:


wtdhr_prices = Nat_gas_df.loc[wtdhr_dates]['Prices'] # Prices at time of Withdrawal 
wtdhr_prices


# In[13]:


# Costs

injection_cost = 10000 #$
withdrawal_cost = 10000 #$
transp_cost = 50000 #$
max_inventory = 1500000 #BTU total
max_injection_rate_monthly = 500000 #BTU monthly
max_withdrawal_rate_monthly = 500000 #BTU monthly
storage_cost_monthly = 100000 #$


# In[46]:


planned_injection = 0

for i in range(0,len(inj_prices)):
     planned_injection += inj_prices[i]*max_injection_rate_monthly

planned_injection        


# In[47]:


planned_Withdrawal = 0

for i in range(0,len(wtdhr_prices)):
     planned_Withdrawal += wtdhr_prices[i]*max_withdrawal_rate_monthly

planned_Withdrawal        


# In[16]:


Margin = planned_Withdrawal - planned_injection
Margin


# In[25]:


inj_dates[0] # firt injection


# In[26]:


wtdhr_dates[-1] # last withdrawal


# In[52]:


storage_time = len(pd.date_range(start=inj_dates[0], end=wtdhr_dates[-1], freq='M'))
storage_time


# In[53]:


total_storage_cost = storage_time*storage_cost_monthly
total_storage_cost


# In[ ]:





# In[56]:


Net_margin = ( Margin - transp_cost*(len(wtdhr_dates)+len(inj_dates)) - 
                    total_storage_cost - 
                    injection_cost*len(inj_dates) - 
                     withdrawal_cost*len(wtdhr_dates) )





# Value of the contract
Net_margin


# In[ ]:




