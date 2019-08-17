#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 13:02:25 2019

@author: himanshu
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as sop
from scipy.integrate import quad
import time

import warnings
warnings.filterwarnings('ignore')

r = 0.005

optData = pd.read_csv('..'+os.sep+'data'+os.sep+'testProject.csv')
optData['Date'] = pd.to_datetime(optData['Date'], format='%Y-%m-%d')
optData['Maturity'] = pd.to_datetime(optData['Maturity'], format='%Y-%m-%d')
dates = list(optData['Date'].drop_duplicates())
expiries = sorted(list(optData['Maturity'].drop_duplicates()))

filtData = optData[(optData['Date'] == dates[0]) & (optData['Maturity'] == expiries[0])]
S0 = filtData['Underlying'].iloc[0]

#print(filtData)

def find_cdf(x, Q, B, nu):
    v=1/((1+Q*(np.exp(-B*(x-1))))**(1/nu))
    return 1-v

def find_call_value(S0, K, r, T, p0):
    Q, B, nu = p0
    func = lambda x: find_cdf(x, Q, B, nu)
    p = quad(func, K/S0, np.inf)
    return S0*np.exp(-r*T)*p[0]

def error_func(p0):
    global i, min_RMSE
    se = []
    for row, option in filtData.iterrows():
        T = (option['Maturity'] - option['Date']).days / 365.
        model_value = find_call_value(S0, option['Strike'], r, T, p0)
        se.append((model_value - option['Call']) ** 2)
    RMSE = np.sqrt(sum(se) / len(se))
    min_RMSE = min(min_RMSE, RMSE)
    #if i % 50 == 0:
    #    print('%4d |' % i, np.array(p0), '| %7.3f | %7.3f' % (RMSE, min_RMSE))
    #i += 1
    return RMSE

def generate_plot(opt, options):
    options['Model'] = 0.0
    for row, option in options.iterrows():
        T = (option['Maturity'] - option['Date']).days / 365.
        options.at[row, 'Model'] = find_call_value(S0, option['Strike'], r, T, opt)

    options = options.set_index('Strike')
    fig, ax = plt.subplots(2, sharex=True, figsize=(10, 8))
    options[['Call', 'Model']].plot(style=['b-', 'ro'],
                    title='%s' % str(option['Maturity'])[:10], ax=ax[0])
    ax[0].set_ylabel('option values')
    ax[0].grid(True)
    xv = options.index.values
    se = ((options['Model'] - options['Call'])**2)
    RMSE = np.sqrt(sum(se) / len(se))
    print(RMSE)
    ax[1] = plt.bar(xv - 5 / 2., options['Model'] - options['Call'],
                    width=5)
    plt.ylabel('difference')
    plt.xlim(min(xv) - 10, max(xv) + 10)
    plt.tight_layout()
    plt.grid(True)
    
i = 0
min_RMSE = 100.

t = time.time()
p0 = sop.brute(error_func, ((6, 10, 0.5), (100, 105, 0.5), (2, 3, 0.5)), finish=None)
#print(p0)
opt = sop.fmin(error_func, p0, xtol=0.00001,
                ftol=0.00001, maxiter=750, maxfun=1500)
print(time.time()-t)
#print(opt)
#generate_plot(opt, filtData)