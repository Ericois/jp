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
Nat_gas = pd.read_csv("Nat_gas.csv",index_col=0)


# In[3]:


Nat_gas.tail(10)


# In[4]:


Nat_gas.info()


# In[5]:


Nat_gas.describe()


# In[6]:


ax=Nat_gas["Prices"].plot(figsize=(10,10))


# In[7]:


print(Nat_gas.index)


# In[8]:


Nat_gas["SMA_3"]=Nat_gas.Prices.rolling(3).mean()
Nat_gas["SMA_12"]=Nat_gas.Prices.rolling(12).mean()


# In[9]:


Nat_gas


# In[10]:


Nat_gas.loc[:,["Prices","SMA_3","SMA_12"]].plot(figsize=(10,10))
plt.legend(loc="upper left",fontsize=8)
plt.show()


# In[11]:


time = np.arange(1, len(Nat_gas) + 1)
Nat_gas['time'] = time
data = Nat_gas[['time', 'Prices']]
data.tail()


# In[12]:


time


# In[13]:


Nat_gas.tail()


# In[14]:


reg = np.polyfit(data['time'], data["Prices"], deg = 1)
reg


# In[15]:


trend = np.polyval(reg, data['time'])
std = data['Prices'].std()
plt.figure(figsize=(10,6))
plt.plot(data['time'].values, data['Prices'].values)
plt.plot(data['time'].values, trend, 'r--')
plt.plot(data['time'].values, trend - std, 'g--')
plt.plot(data['time'].values, trend + std, 'g--');


# In[16]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[17]:


result = seasonal_decompose(Nat_gas['Prices'], model='multiplicative',period = 12)


# In[18]:


result.plot()
plt.show()


# In[19]:


plt.figure(figsize = (16,7))
result.seasonal.plot();


# In[22]:


plt.figure(figsize = (16,7))
result.trend.plot();


# # Forecasting using SARIMA

# In[28]:


# Import data
Nat_gas_df = pd.read_csv("Nat_gas.csv")


# In[29]:


Nat_gas_df.head()


# In[26]:


from statsmodels.tsa.statespace.sarimax import SARIMAX


# In[31]:


# Convert Month into Datetime
Nat_gas_df['Dates']=pd.to_datetime(Nat_gas_df['Dates'])


# In[32]:


Nat_gas_df.head()


# In[34]:


Nat_gas_df.set_index('Dates',inplace=True)


# In[56]:


Nat_gas_df.head()


# In[38]:


Nat_gas_df.plot()


# In[39]:


### Testing For Stationarity

from statsmodels.tsa.stattools import adfuller


# In[41]:


test_result=adfuller(Nat_gas_df['Prices'])


# In[42]:


#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(Prices):
    result=adfuller(Prices)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
    


# In[43]:


adfuller_test(Nat_gas_df['Prices'])


# In[45]:


import pmdarima as pm
from pmdarima.model_selection import train_test_split


# In[49]:


#Let's run auto_arima() function to get best p,d,q,P,D,Q values

pm.auto_arima(Nat_gas_df['Prices'], seasonal=True, m=12,max_p=7, max_d=5,max_q=7, max_P=4, max_D=4,max_Q=4,
               trace=True,error_action='ignore', suppress_warnings=True, stepwise=True).summary()


# In[47]:


# Let's split the data into train and test set

train_data = Nat_gas_df[:len(Nat_gas_df)-12]
test_data = Nat_gas_df[len(Nat_gas_df)-12:]


# In[48]:





# In[84]:


arima_model = SARIMAX(train_data['Prices'], order = (2,1,2), seasonal_order = (1,1,1,12))
arima_result = arima_model.fit()
arima_result.summary()


# In[85]:


arima_pred = arima_result.predict(start = len(train_data), end = len(Nat_gas_df)-1, typ="levels").rename("ARIMA Predictions")
arima_pred


# In[86]:


test_data['Prices'].plot(figsize = (16,5), legend=True)
arima_pred.plot(legend = True);


# In[87]:


Nat_gas_df['forecast']=arima_result.predict(start=35,end=47,dynamic=True)
Nat_gas_df[['Prices','forecast']].plot(figsize=(12,8))


# In[89]:


Nat_gas_df.tail(12)


# In[90]:


test_data['ARIMA_Predictions'] = arima_pred


# In[91]:


test_data


# In[93]:


from sklearn.metrics import mean_squared_error
from statsmodels.tools.eval_measures import rmse


# In[94]:


arima_rmse_error = rmse(test_data['Prices'], arima_pred)
arima_mse_error = arima_rmse_error**2
mean_value = Nat_gas_df['Prices'].mean()

print(f'MSE Error: {arima_mse_error}\nRMSE Error: {arima_rmse_error}\nMean: {mean_value}')


# # Run entire code after entering date

# In[97]:


arima_pred_full = arima_result.predict(start = 0, end = len(Nat_gas_df)-1, typ="levels").rename("ARIMA Predictions")


# In[100]:


#enter Month and year in following format 'yyyy/mm/dd' -- KEEP day at end of the month AND THEN Run the ENTIRE code
#example : userdate = 2024-05-31
userdate = '2024-05-31'
arima_pred_full.loc[userdate]


# In[102]:


# Actual Price
Nat_gas_df.loc[userdate]['Prices']


# # Extrapolate 1 year in the future

# In[ ]:





# In[109]:


#enter Month and year in following format 'yyyy/mm/dd' -- KEEP day at end of the month AND THEN Run the ENTIRE code
#example : userdate = 2024-05-31
userdate = '2020-12-31'
start = Nat_gas_df.index.get_loc(userdate)
arima_pred_future = arima_result.predict(start = start, end = start+12, typ="levels").rename("ARIMA Predictions")


# In[111]:


arima_pred_future.tail(1)


# In[113]:


# Actual Price in next 12 months
Nat_gas_df.iloc[start+12]


# # Forecasting using Prophet

# In[2]:


from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


# In[3]:


#read in the data 
df = pd.read_csv('Nat_Gas.csv')
df.head()


# In[4]:


#Select the date and the price 
df = df[['Dates', 'Prices']]
#Rename the features:
df = df.rename(columns={'Dates':'ds', 'Prices':'y'})
df.head()


# In[5]:


# Get last 12 rows of data and store them into a new variable 
last = df[len(df)-12:]
last


# In[6]:


#Get all rows except the last 12 
df = df[:-12]
df.head()


# In[7]:


#Creating Prophet Object (Model)
fbp = Prophet(seasonality_mode='multiplicative', mcmc_samples=360)

#Train the model 
fbp.fit(df)
future = fbp.make_future_dataframe(periods=24, freq='M')
forecast = fbp.predict(future)


# In[8]:


#plot the data
plot_plotly(fbp, forecast)


# In[9]:


forecast.tail(15)


# In[10]:


#Show the models prediction for 8/31/24
forecast[forecast.ds == '8/31/24']['yhat']


# In[11]:


#Actual price for 8/31/24
last[last.ds == '8/31/24']['y']


# # Run entire code after entering date

# In[22]:


future = fbp.make_future_dataframe(periods=120, freq='MS')
forecast = fbp.predict(future)

#enter Month and year in following format 'mm/01/yy' -- KEEP day @ '01' AND THEN Run the ENTIRE code
#example : userdate = 01/01/24
userdate = '01/01/24'
forecast[forecast.ds == userdate]['yhat']


# In[23]:


#Actual price for user date
last[last.ds == '1/31/24']['y']


# In[24]:


forecast[forecast.ds == userdate].index.values


# In[37]:


# Forecast of Price next 12 months
T = forecast[forecast.ds == userdate].index.values +12
forecast.iloc[T]['yhat']


# In[ ]:




