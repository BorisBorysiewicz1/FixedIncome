#!/usr/bin/env python
# coding: utf-8

# Getting started script for Fixed Income Assignment 1, 2025


from bound import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy.stats as st
import pandas as pd
import seaborn as sns
sns.set()
from scipy.stats import norm


# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

class InterpolationBase(object):
    """
    Base class for interpolation objects
    """
    def __init__(self, abscissae, ordinates):
        if not sorted(abscissae) or            len(abscissae) != len(ordinates):
               raise RuntimeError('abscissae/ordinates length mismatch')
        self.N = len(abscissae)
        self.abscissae, self.ordinates = abscissae, ordinates

    def _locate_single(self, x):
        i, j = bound(x, self.abscissae)
        x_lo, x_hi = self.abscissae[i], self.abscissae[j]
        y_lo, y_hi = self.ordinates[i], self.ordinates[j]

        return x_lo, x_hi, y_lo, y_hi

    def locate(self, x):
        if not isinstance(x, np.ndarray):
            return self._locate_single(x)
        else:
            n = x.shape[0]
            x_lo = np.zeros(n)
            x_hi = np.zeros(n)
            y_lo = np.zeros(n)
            y_hi = np.zeros(n)
            for i in range(0,n):
                x_lo[i], x_hi[i], y_lo[i], y_hi[i] = self._locate_single(x[i])

        return x_lo, x_hi, y_lo, y_hi

# -----------------------------------------------------------------------------

class InterpolationLinear(InterpolationBase):
    """
    Linear interpolation object
    """
    def __init__(self, abscissae, ordinates):
        InterpolationBase.__init__(self, abscissae, ordinates)

    def __call__(self, x):
        x_lo, x_hi, y_lo, y_hi =             InterpolationBase.locate(self, x)
        r = 1.0 - (x_hi - x)/(x_hi - x_lo)

        return r*(y_hi - y_lo) + y_lo

# -----------------------------------------------------------------------------

class InterpolationLoglinear(InterpolationBase):
    """
    Log Linear interpolation object
    """
    def __init__(self, abscissae, ordinates):
        InterpolationBase.__init__(self, abscissae, ordinates)

    def __call__(self, x):
        x_lo, x_hi, y_lo, y_hi =             InterpolationBase.locate(self, x)
        ln_ylo, ln_yhi = np.log(y_lo), np.log(y_hi)
        R = 1.0 - (x_hi - x)/(x_hi - x_lo)

        return np.exp(ln_ylo+(ln_yhi - ln_ylo)*R)

# -----------------------------------------------------------------------------

class Curve(object):
    """
    General curve object
    """
    def __init__(self, times, factors, interp):
        self.__impl = interp(times, factors)

    def __call__(self, t):
        return self.__impl(t)

# -----------------------------------------------------------------------------

def ois_swaps(f, maturities, rates):
    """
    compute valuation of fixed rate ois swap
    """
    # compute discrete set of discount factors
    df = np.ones(len(maturities))
    df[1:] = np.exp(-f*maturities[1:])
    # put in interpolator curve
    z = Curve(maturities, df, InterpolationLoglinear)
    # longest maturity
    max_maturity = maturities[-1]
    # annual period up to longest maturity
    times = np.arange(0.0, max_maturity+0.1)
    # pv01 up to longest maturity
    pv01 = z(times)
    pv01[0] = 0.0
    all_pv01 = np.cumsum(pv01)
    # filter maturities for the maturities needed
    pv01 = all_pv01[maturities.astype(np.int64)]
    # compute ois swap values
    swap_values = rates * pv01[1:] - (1.0 - df[1:])

    return swap_values

# -----------------------------------------------------------------------------

def forward(times, z):
    """

    :param times: time points
    :param z: discount curve function
    :return:
    """
    df = z(times)
    delta = times[1:] - times[:-1]
    fwd = (df[:-1] / df[1:] - 1.) / delta

    return fwd

# -----------------------------------------------------------------------------

def fixed_floating_swaps(f_6m, maturities_fixed, rates, z):
    """
    valuation formula for fixed-floating swaps (based on 6m forward)
    
    :param f_6m: 6m zero rates
    :param maturities_fixed: fixed maturity times
    :param maturities_float: floating maturity times
    :param rates: market swap rates
    :param z: ois discount funtion
    :return: value of receiver swap
    """
    # compute discrete set of discount factors
    df_6m = np.ones(len(maturities_fixed))
    df_6m[1:] = np.exp(-f_6m*maturities_fixed[1:])
    # put in interpolator curve
    z_6m = Curve(maturities_fixed, df_6m, InterpolationLoglinear)
    years = np.arange(0, maturities_fixed[-1]+0.1)
    half_years = np.arange(0, 2*maturities_fixed[-1]+0.1)*0.5
    df_fixed = z(years)
    df_fixed[0] = 0.0
    pv01_fixed = np.cumsum(df_fixed)
    pv01 = pv01_fixed[maturities_fixed.astype(np.int64)]

    fixed_leg = rates * pv01[1:]
    # floating leg cash flows
    forwards = forward(half_years, z_6m)
    float_cf = z(half_years[1:]) * forwards*0.5
    # select only full year sums
    float_leg = np.cumsum(float_cf)[1::2]

    return fixed_leg - float_leg

# -----------------------------------------------------------------------------

def value_swap(Z_0T, swap_rate, maturity):
    """
    assumes annual fixed rate and semi-annual float payments
    """
    fix_dates = np.arange(1, maturity+0.01)
    pv01 = np.sum(Z_0T(fix_dates))
    fixed_leg =  # TODO
    float_leg =  # TODO
    
    return fixed_leg - float_leg

# -----------------------------------------------------------------------------
# Start Script
# -----------------------------------------------------------------------------

# maturity dates
maturities = np.concatenate((np.arange(0.0, 14.1), np.arange(15.0, 30.1, 5.0)))
# start with constant guess
df = np.ones(len(maturities))
df[1:] = np.exp(-0.02*maturities[1:])
z_curve = Curve(maturities, df, InterpolationLoglinear)
# check
z_curve(maturities)

# calibrate
# market rates
ois_rates = np.array([0.1, 0.15, 0.23, 0.32, 0.4, 0.5, 0.7,
                      0.8, 0.9, 1.0, 1.2, 1.3, 1.38, 1.43,
                      1.5, 1.8, 1.9, 2.1])/100.0

# solution using a solver
f = fsolve(ois_swaps, -np.log(df[1:])/maturities[1:], args=(maturities,
                                                            ois_rates))

# add discount factors for all maturities
df[1:] = np.exp(-f*maturities[1:])

### VERIFY USING THE BOOTSTRAP ALGORITHM FROM THE CLASS

# create curve with maturities as x values and discount factors as y values
z_curve = Curve(maturities, df, InterpolationLoglinear)
# check
z_curve(maturities)

# Compute daily discount factors for next 30 years
days_in_year = 250
z_daily = np.zeros(days_in_year * 30)
days = np.zeros(days_in_year * 30)
for d in range(0, days_in_year * 30):
    z_daily[d] =  # TODO

print(z_daily)


f_daily = -np.log(z_daily[1:] / z_daily[:-1])*days_in_year
plt.plot(days[1:], f_daily)
plt.grid(True)
plt.title('daily forward rates')

fixed_rates_swaps_ois = np.array([0.02, 0.025, 0.03])
swap_prices = pd.Series(index=fixed_rates_swaps_ois)
for r in fixed_rates_swaps_ois:
    swap_prices[r] = value_swap(z_curve, r, 10)

print(swap_prices)

# -----------------------------------------------------------------------------
# Hull-White swaption pricer
# -----------------------------------------------------------------------------

def lgm_pricer_swaption_pricer(Z_0T, strike, G_T, n, alpha, type='call'):
    """

    """
    eps = np.random.normal(0.0, 1.0, (n,))
    Z_T = # TODO
    Z_T = Z_T * np.outer(np.ones(n), Z_0T[1:]/Z_0T[0])
    PV01 = # TODO PV01
    S = # TODO Swap Rate

    if type =='call':
        payoff = #TODO
    else:
        payoff = #TODO

    value = Z_0T[0] * np.mean(payoff)
    std = Z_0T[0] * np.std(payoff)
    z_a = st.norm.ppf(1.0 - alpha)
    CI = value + np.array([-1.0, 1.0])*z_a*std / np.sqrt(n)

    return value, std, CI

kappa = 0.05
sigma = 0.01
t = np.arange(5.0, 10.1)

Z_0T = z_curve(t)
print(Z_0T)

xi_t = 0.5*sigma**2 / kappa *(np.exp(2.0*kappa*t[0]) - 1.0)
psi = (1.0 - np.exp(-kappa*t)) / kappa
# for all maturities above 5
G_T = (psi[0] - psi[1:])**2 * xi_t

results = {}

# Calculate the call and put values for different numbers of simulations
products = ['call', 'put']
num_sims = np.array([10000, 40000])
alpha = 0.025
notional = 100e6
strike = 0.01
for product in products:
    results[product] = pd.DataFrame(index=['price', 'std', 'ci'], columns=num_sims)
    for i in range(0, len(num_sims)):
        price, std, ci = lgm_pricer_swaption_pricer(Z_0T, strike, G_T, num_sims[i], alpha, product)
        results[product][num_sims[i]].loc['price'] = price * notional
        results[product][num_sims[i]].loc['std'] = std * notional
        results[product][num_sims[i]].loc['ci'] = ci * notional

for product in products:
    print(product, '\n', np.round(results[product], 2))

