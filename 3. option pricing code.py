"""
Fixed Income
Assignment 1, 2025 (Part 2, option pricing)

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
# Valuation of Discount Bond option
# -----------------------------------------------------------------------------

section_header('Discount Bond Pricing')

def lgm_pricer_discount_bond_option_pricer(Z_0T, strike, G_T, num_sims,
                                           alpha, type='call'):
    """
    Pricing function for a discount bond
    """
    # generates an array of size num_sims with standard normal random variates
    eps = np.random.normal(0.0, 1.0, (num_sims,))
    Z_12 = eps # TODO

    if type =='call':
        # value * indicator (value positive)
        payoff = (Z_12 - strike)*((Z_12 - strike) > 0.0)
    else:
        payoff = (strike - Z_12)*((strike - Z_12) > 0.0)

    value = 1 # TODO
    std = 1 # TODO
    # calculate the quantile of the normal distribution
    z_a = st.norm.ppf(1.0 - alpha)
    # create a confidence interval with (1 - alpha) confidence
    CI = value + np.array([-1.0, 1.0])*z_a*std 

    return value, std, CI



# model parameters
kappa = 0.05
sigma = 0.01

# the current discount factors from time 0 to times 1 and 2
Z_0T = np.array([0.97, 0.95])

# product details
t = np.array([1.0, 2.0])
strike = 0.97

# number of simulations
num_sims = 10000

xi_t = 0.5*sigma**2 / kappa *(np.exp(2.0*kappa*t[0]) - 1.0)
psi = 1.0 # TODO
G_T = 1.0 # TODO

# level of confidence interval
alpha = 0.05

# calculate the value, standard deviation and confidence interval
val, std, ci = lgm_pricer_discount_bond_option_pricer(Z_0T, strike, G_T,
                                                      num_sims, alpha,
                                                      type='call')
