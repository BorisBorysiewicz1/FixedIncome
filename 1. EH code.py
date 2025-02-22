"""
Fixed Income
Assignment 1, 2025 (Part 1)

Jeroen Kerkhof

This helps to get you started. It is obviously not complete.

"""

import numpy as np # matrix algebra
import matplotlib.pyplot as plt # plotting
import pandas as pd # data frames
import datetime as dt # time
import scipy.stats as st # statistics
import statsmodels.api as sm # regression models

import numpy.linalg as la # submodule for linear algebra (e.g. inverse)

# not necessary, but gives nicer graph settings https://seaborn.pydata.org/
import seaborn as sns
sns.set_style("whitegrid")

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------


def date_changers(date_int):
    """ creates datetime date from yyyymmdd-string"""
    date_string = str(date_int)
    return dt.datetime(int(date_string[0:4]), int(date_string[4:6]),
                       int(date_string[6:8]))

# -----------------------------------------------------------------------------


# fama-bliss regression
def regression(forwards, yield1, rx):
    """ """
    X = forwards - yield1
    X = sm.add_constant(X)
    results = sm.OLS(rx, X).fit()

    # alternative taking Newey-West standard errors
    # Clearly you need to determine an appropriate number of lags
    # results = sm.OLS(rx, X).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    return results


# -----------------------------------------------------------------------------


def call_black(f, k, ttm, sigma):
    """The Black call formula.

    :param f: forward rate
    :param k: strike
    :param ttm: time to maturity
    :param sigma: volatility
    """
    V = sigma**2 * ttm
    if isinstance(V, np.ndarray):
        if V.any() <= 0.0:
            raise RuntimeError("vol must be non-negative")
    else:
        if V <= 0.0:
            raise RuntimeError("vol must be non-negative")

    d_p = (np.log(f/k) + 0.5*V)/np.sqrt(V)
    d_m = d_p - np.sqrt(V)

    return f*st.norm.cdf(d_p) - k*st.norm.cdf(d_m)

# -----------------------------------------------------------------------------


def put_black(f, k, ttm, sigma):
    """The Black put formula.

    :param f: forward rate
    :param k: strike
    :param ttm: time to maturity
    :param sigma: volatility
    """
    V = sigma**2 * ttm
    if V <= 0:
        raise RuntimeError("vol must be non-negative")

    d_p = (np.log(f/k) + 0.5*V)/np.sqrt(V)
    d_m = d_p - np.sqrt(V)

    return k*st.norm.cdf(-d_m) - f*st.norm.cdf(-d_p)

# -----------------------------------------------------------------------------


def section_header(print_statement, print_line_length=80, print_line_start=5):
    """
    helper function for printing section headers in output
    """
    print(print_line_start * '#' + ' ' + print_statement + ' ' +
      (print_line_length - len(print_statement) - print_line_start - 2) * '#')

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

fama_bliss_file = "fb.csv"
fb = pd.read_csv(fama_bliss_file, header=1)
fb.columns = ['date', 'yield1', 'yield2', 'yield3', 'yield4', 'yield5']

# set the index
fb.set_index('date', inplace=True)
# use the Fama-Bliss data from 1964; transform to percentages
data = fb['1964-01-01':'2023-12-31'] / 100

# -----------------------------------------------------------------------------
# Data preparation

# done explicit
# feel free to use loops
# -----------------------------------------------------------------------------

# compute continuously compounding yields
data['P1'] = np.exp(-data['yield1']*1)
data['P2'] = np.exp(-data['yield2']*2)
data['P3'] = np.exp(-data['yield3']*3)
data['P4'] = np.exp(-data['yield4']*4)
data['P5'] = np.exp(-data['yield5']*5)

# log prices
data['p1'] = np.log(data['P1'])
data['p2'] = np.log(data['P2'])
data['p3'] = np.log(data['P3'])
data['p4'] = np.log(data['P4'])
data['p5'] = np.log(data['P5'])

# yields
data['y1'] = -data['p1']/1
data['y2'] = -data['p2']/2
data['y3'] = -data['p3']/3
data['y4'] = -data['p4']/4
data['y5'] = -data['p5']/5

# forwards
data['f1'] = data['y1']
data['f2'] = data['p1']-data['p2']
data['f3'] = data['p2']-data['p3']
data['f4'] = data['p3']-data['p4']
data['f5'] = data['p4']-data['p5']

print('yields: \n',
      np.round(data[['y1', 'y2', 'y3', 'y4', 'y5']].mean()*100, 2))

n = data.shape[0]

# make a simple plot from a data frame
data[['f1', 'f2', 'f3', 'f4', 'f5']].plot(grid=True,
                                          title="History of forward rates")
plt.show()

# make a simple plot from a data frame
data[['y1', 'y2', 'y3', 'y4', 'y5']].plot(grid=True,
                                          title="History of yields")
plt.show()

# -----------------------------------------------------------------------------
# Start analysis
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Compute Sharpe Ratios
# -----------------------------------------------------------------------------

section_header('Compute Sharpe Ratios')

p = data[['p1', 'p2', 'p3', 'p4', 'p5']].values
# p^{(n)}_{t+1} - p^{(n-1)}_t measured anually
# for the holding period returns
r_n = p[12:, 0:4] - p[:-12, 1:5]
# compute excess holding period returns
# r^{(n)}_{t+1} - y^{(1)}_t
rx_n = r_n - np.kron(data['y1'].values[:-12],
                     np.ones(4)).reshape(n-12, 4)

# rx_n_mean = # T ODO
# rx_n_std = #  TODO
# rx_sr = #  TODO

rx_n_mean = 0.05 * np.ones(4)
rx_n_std = 0.02 * np.ones(4)
rx_sr = 2.5 * np.ones(4)

sharpe_ratios = pd.DataFrame(np.array([rx_n_mean*10000.0,
                                       rx_n_std*10000.0,
                                       rx_sr]).T,
                             index=['SR2', 'SR3', 'SR4', 'SR5'],
                             columns=['E[rx]', 'sigma[rx]', 'SR'])

print(np.round(sharpe_ratios, 2))

# -----------------------------------------------------------------------------
# Fama-Bliss regressions
# -----------------------------------------------------------------------------

section_header('Fama-Bliss Regressions')

print('type(data)=', type(data))
print('type(data.values)=', type(data.values))


r2f = regression(data['f2'].values[:-12], data['y1'].values[:-12], rx_n[:, 0])

# (Much) more to do...

# -----------------------------------------------------------------------------
# Factor structure
# -----------------------------------------------------------------------------

section_header('Factor Structure')

# calculate the covariance matrix of excess returns
cov_excess_rates = np.cov(rx_n.T)

# get eigenvalues (w) and vectors (v)
w, v = la.eig(cov_excess_rates)

f = rx_n @ v

# make a simple plot
plt.plot(v)
plt.show()


factor1_reg_results_2 = sm.OLS(rx_n[:, 0], f[:, 0]).fit()
factor1_reg_results_3 = sm.OLS(rx_n[:, 1], f[:, 0]).fit()
