# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 19:48:20 2020

@author: saksh

"""
#starter gun
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from io import BytesIO
import pandas_datareader.data as dat_aq
from datetime import datetime, timedelta
#recession data for comparison
usrec = dat_aq.DataReader('USREC', 'fred', start=datetime(1950, 1, 1), end=datetime(2010, 10, 1))

'''
function to pull the exogeneous and endogeneous variables from database
'''
def pull_data(endog, db_name, start_time, end_time, frequency='QS', exog=None):
    if exog is not None:
        endog_dat = dat_aq.DataReader(endog, db_name, start = (start_time-timedelta(days=92)), end = end_time).iloc[1:]
        exog_dat = dat_aq.DataReader(exog, db_name, start = (start_time), end = end_time).iloc[1:]
        endog_dat = endog_dat.resample('QS').mean()
        exog_dat = exog_dat.resample('QS').mean()
        return endog_dat, exog_dat
    elif exog is None:
        endog_dat = dat_aq.DataReader(endog, db_name, start = (start_time-timedelta(days=92)), end = end_time).iloc[1:]
        endog_dat = endog_dat.resample('QS').mean()
        return endog_dat

'''
pre-enigineering data: 3 options available, 
1) Log Mean Return
2) Log Returns
3) Percentage Change
Any option can be chosen based on the properties of data
'''

def ready_data(data, ret_type):
    if ret_type == 'log_mean_ret':
        data['log'] = np.log(data.iloc[:,0])
        data['log'].fillna(0, inplace = True)
        data['log_ret'] = data['log'].diff()
        data['log_ret'].fillna(method = 'bfill', inplace = True)
        data['log_ret'] = data['log_ret'] - data['log_ret'].mean()
        return data.log_ret
    elif ret_type == 'log_ret':
        data['log'] = np.log(data.iloc[:,0])
        data['log'].fillna(0, inplace = True)
        data['log_ret'] = data['log'].diff()
        data['log_ret'].fillna(method = 'bfill', inplace = True)
        return data.log_ret
    elif ret_type == 'pct_chg':
        data['pct_chg'] = data.iloc[:,0].pct_change()*100
        data['pct_chg'].fillna(method = 'bfill', inplace = True)
        return data.pct_chg

'''
Markov switching model has 2 differnet types available:
1) Markov - Hamilton Model
2) Markov - Filardo Model
_The filardo model allows usage of two variables, where the exogeneous variable is used calculate to the time-variant probability
'''
def model_fit(endog, model_type = 'Markov_Hamilton', regimes = 2, model_order = 4, exog = None):
    if model_type == 'Markov_Hamilton':
        model = sm.tsa.MarkovAutoregression(endog, k_regimes = regimes, order = model_order, switching_ar = False)
        np.random.seed(12345)
        result = model.fit(search_reps=20)
        return result
    elif model_type == 'Markov_Filardo':
        model = sm.tsa.MarkovAutoregression(endog, k_regimes = regimes, order = model_order, switching_ar = False, exog_tvtp = sm.add_constant(exog))
        np.random.seed(12345)
        result = model.fit(search_reps=20)
        return result
    
'''
Wrapper function for the functions.
Will incorporate input from users to improve accessibility
'''
def SOP_Phase_1(endog, db_name, start_time, end_time, frequency, endog_ret_type, model_type, regimes, model_order, exog=None, exog_ret_type=None):
    if exog is not None:
        endog_data, exog_data = pull_data(endog, db_name, start_time, end_time, frequency, exog = exog)
        endog_ready = ready_data(endog_data, endog_ret_type)
        exog_ready = ready_data(exog_data, exog_ret_type)
        #print(endog_ready)
        #print(exog_ready)
        result = model_fit(endog_ready, model_type, regimes, model_order, exog = exog_ready)
        return result
    elif exog is None:
        endog_data = pull_data(endog, db_name, start_time, end_time, frequency)
        endog_ready = ready_data(endog_data, endog_ret_type)
        #print(endog_ready)
        result = model_fit(endog_ready, model_type, regimes, model_order)
        return result
    
result = SOP_Phase_1('GDPC1', 'fred', datetime(1985,1,1), datetime(2019, 10, 1), 'QS', 'log_mean_ret', 'Markov_Filardo', 2, 4, exog = 'T10Y3M', exog_ret_type = 'log_mean_ret')
#result = SOP_Phase_1('GDPC1', 'fred', datetime(1990,1,1), datetime(2018, 10, 1), 'QS', 'pct_chg', 'Markov_Hamilton', 2, 4)
result.summary()

fig, axes = plt.subplots(2, figsize = (12,7))
ax = axes[0]
ax.plot(result.filtered_marginal_probabilities[0])
ax.fill_between(usrec.index, 0, 1, where=usrec['USREC'].values, color='k', alpha=0.1)
ax.set_xlim(datetime(1985, 1, 1), datetime(2019, 10, 1))
ax.set(title = "Recession probability")

ax = axes[1]
ax.plot(result.smoothed_marginal_probabilities[0])
ax.fill_between(usrec.index, 0, 1, where=usrec['USREC'].values, color='k', alpha=0.1)
ax.set_xlim(datetime(1985, 1, 1), datetime(2019, 10, 1))
ax.set(title = "Smoothed Recession probability")
fig.tight_layout()
plt.show()
