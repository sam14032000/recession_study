# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 18:51:01 2020

@author: saksh
Module file to anble function access for recession study
Study Participants: Saksham Agrawal, Kratik Lodha

All functions are defined in here have docstring defining fucntion
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from io import BytesIO
import pandas_datareader.data as dat_aq
from datetime import datetime, timedelta
import math


'''
function to pull the exogeneous and endogeneous variables from database
'''
def pull_data(endog, db_name, start_time=None, end_time=None, frequency='QS'):
    if start_time is not None or end_time is not None:
        endog_dat = dat_aq.DataReader(endog, db_name, start = (start_time), end = end_time)
    else:
        endog_dat = dat_aq.DataReader(endog, db_name)

    endog_dat.dropna(inplace=True)
    endog_dat = endog_dat.astype('float')
    endog_dat = endog_dat.resample(frequency).mean()
    return endog_dat
        
'''
pre-enigineering data: 3 options available, 
1) Log Mean Return
2) Log Returns
3) Percentage Change
Any option can be chosen based on the properties of data
'''

def ready_data(data, ret_type):
    df = pd.DataFrame(index = data.index)
    #df.index = data.index
    df['Data'] = data
    if ret_type == 'log_mean_ret':
        df['log'] = np.log(df.iloc[:,0])
        df['log'].fillna(0, inplace = True)
        df['log_ret'] = df['log'].diff()
        df['log_ret'].fillna(method = 'bfill', inplace = True)
        df['log_ret'] = df['log_ret'] - df['log_ret'].mean()
        return df.log_ret[1:]
    elif ret_type == 'log_ret':
        df['log'] = np.log(df.iloc[:,0])
        df['log'].fillna(0, inplace = True)
        df['log_ret'] = df['log'].diff()
        df['log_ret'].fillna(method = 'bfill', inplace = True)
        return df.log_ret[1:]
    elif ret_type == 'pct_chg':
        df['pct_chg'] = df.iloc[:,0].pct_change()*100
        df['pct_chg'].fillna(method = 'bfill', inplace = True)
        return df.pct_chg[1:]
    elif ret_type is None:
        return df[1:]

'''
Markov switching model has 2 differnet types available:
1) Markov - Hamilton Model
2) Markov - Filardo Model
_The filardo model allows usage of two variables, where the exogeneous variable is used to calculate the time-variant probability
'''
def model_fit(endog, model_type = 'Markov_Hamilton', regimes = 2, model_order = 4, exog = None):
    if model_type == 'Markov_Hamilton':
        model = sm.tsa.MarkovAutoregression(endog, k_regimes = regimes, order = model_order, switching_ar = False) 
    elif model_type == 'Markov_Filardo':
        model = sm.tsa.MarkovAutoregression(endog, k_regimes = regimes, order = model_order, switching_ar = False, exog_tvtp = sm.add_constant(exog))
    
    np.random.seed(12345)
    result = model.fit(search_reps=20)
    return result, model


'''
Wrapper function for the functions.
Will incorporate input from users to improve accessibility
'''
def SOP_Phase_1(endog, model_type, regimes, model_order, endog_ret_type=None, exog=None, exog_ret_type = None):
    if exog is not None:
        endog_ready = ready_data(endog, endog_ret_type)
        exog_ready = ready_data(exog, exog_ret_type)
        #print(len(exog_ready))
        #print(len(endog_ready))
        plot_data(exog_ready)
    elif exog is None:
        endog_ready = ready_data(endog, endog_ret_type)
        exog_ready = None
    
    plot_data(endog_ready)
    result, model = model_fit(endog_ready, model_type, regimes, model_order, exog = exog_ready)
    return result, model
    
def plot_data(dataset):
    dataset.plot(figsize = (12,3))
    plt.show()
    
def rec_periods(dataset):
    return dat_aq.DataReader(dataset, 'fred', start = datetime(1960,1,1), end = datetime(2019,10,1))

def rec_prob_graph(result, rec_periods, graph_start, graph_end, invert=False):
    if invert is True:
        marg_prob = 1-result.filtered_marginal_probabilities[0]
        smooth_prob = 1-result.smoothed_marginal_probabilities[0]
    else:
        marg_prob = result.filtered_marginal_probabilities[0]
        smooth_prob = result.smoothed_marginal_probabilities[0]    
    
    fig, axes = plt.subplots(2, figsize = (12,7))
    ax = axes[0]
    ax.plot(marg_prob)
    ax.fill_between(rec_periods.index, 0, 1, where=rec_periods.iloc[:,0].values, color='k', alpha=0.1)
    ax.set_xlim(graph_start, graph_end)
    ax.set(title = "Recession probability")

    ax = axes[1]
    ax.plot(smooth_prob)
    ax.fill_between(rec_periods.index, 0, 1, where=rec_periods.iloc[:,0].values, color='k', alpha=0.1)
    ax.set_xlim(graph_start, graph_end)
    ax.set(title = "Smoothed Recession probability")
    fig.tight_layout()
    plt.show()
    
    rms_rec_periods = rec_periods[result.filtered_marginal_probabilities[0].index[0]:result.filtered_marginal_probabilities[0].index[-1]].resample('QS').mean()
    rms_calc_marg = pd.DataFrame()
    rms_calc_marg['recession'] = rms_rec_periods.iloc[:,0]
    rms_calc_marg['prob'] = result.filtered_marginal_probabilities[0]
    rms_calc_marg['sqr_diff'] = (rms_rec_periods.iloc[:,0] - rms_calc_marg['prob'])**2
    rms_marg = rms_calc_marg['sqr_diff'].mean()
    rms_marg = math.sqrt(rms_marg)
    print(rms_marg)
    print('precision = ', 1/rms_marg)
    
    rms_calc_smooth = pd.DataFrame()
    rms_calc_smooth['recession'] = rms_rec_periods.iloc[:,0]
    rms_calc_smooth['prob'] = result.smoothed_marginal_probabilities[0]
    rms_calc_smooth['sqr_diff'] = (rms_rec_periods.iloc[:,0] - rms_calc_smooth['prob'])**2
    rms_smooth = rms_calc_smooth['sqr_diff'].mean()
    rms_smooth = math.sqrt(rms_smooth)
    print(rms_smooth)
    print('smooth_precision = ', 1/rms_smooth)
