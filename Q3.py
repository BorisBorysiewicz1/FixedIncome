
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

import functions1 as f1 # own functions

fama_bliss_file = "fb.csv"
fb = pd.read_csv(fama_bliss_file, header=1)
fb.columns = ['date', 'yield1', 'yield2', 'yield3', 'yield4', 'yield5']

# set the index
fb.set_index('date', inplace=True)
# use the Fama-Bliss data from 1964; transform to percentages
data = fb['1964-01-01':'2023-12-31'] / 100

# compute continuously compounding yields and log prices
for i in range(1, 6):
    data[f'P{i}'] = np.exp(-data[f'yield{i}'] * i)
    data[f'p{i}'] = np.log(data[f'P{i}'])
    data[f'y{i}'] = -data[f'p{i}'] / i

# forwards
data['f1'] = data['y1']
for i in range(2, 6):
    data[f'f{i}'] = data[f'p{i-1}'] - data[f'p{i}']

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

#--------------------------------------------------------------------------------

# create prices and holding period returns
data['P1_lag'] = data['P1'].shift(1)
data['HPR1'] = data['P1'] / data['P1_lag'] - 1

for i in range(2, 6):
    data[f'P{i}_lag'] = data[f'P{i}'].shift(1)
    data[f'HPR{i}'] = data[f'P{i}'] / data[f'P{i}_lag'] - 1

# drop rows with NaN values created by shifting
data.dropna(inplace=True)

print(data[['HPR1', 'HPR2', 'HPR3', 'HPR4', 'HPR5']].head())

# compute excess returns
for i in range(1, 6):
    data[f'excess_return{i}'] = data[f'HPR{i}'] - data[f'yield{i}']

# display the first few rows of the excess returns
print(data[[f'excess_return{i}' for i in range(1, 6)]].head())

# compute forward rates
for i in range(2, 6):
    data[f'f{i}'] = data[f'p{i-1}'] - data[f'p{i}']

# display the first few rows of the forward rates
print(data[[f'f{i}' for i in range(1, 6)]].head())

#--------------------------------------------------------------------------------
# 1. compute the average log yield
# 2. Compute the Sharpe ratios for n = 2, 3, 4, 5

p = data[['p1', 'p2', 'p3', 'p4', 'p5']].values

# Compute holding period returns (annualized)
r_n = p[12:, 0:4] - p[:-12, 1:5]  # Shape (T-12, 4)

# Compute excess holding period returns: r^{(n)}_{t+1} - y^{(1)}_t
rx_n = r_n - np.tile(data['y1'].values[:-12].reshape(-1, 1), 4)  # Shape (T-12, 4)

rx_n_mean = rx_n.mean(axis=0)  # Expected excess returns
rx_n_std = rx_n.std(axis=0)    # Standard deviation of excess returns
rx_sr = rx_n_mean / rx_n_std   # Sharpe ratios

# we delete fixed value inputs and replace them with the computed values

# Create DataFrame for Sharpe ratios
sharpe_ratios = pd.DataFrame(
    np.array([rx_n_mean * 10000.0, rx_n_std * 10000.0, rx_sr]).T,
    index=['SR2', 'SR3', 'SR4', 'SR5'],
    columns=['E[rx]', 'sigma[rx]', 'SR']
)

print(np.round(sharpe_ratios, 2))

#--------------------------------------------------------------------------------

# 3. Estimate the Fama-Bliss regressions and report a, b, R2, se(a), se(b) for n = 2, 3, 4, 5

# Define the regression function (professor's version)
def regression(forwards, yield1, rx):
    """ """
    X = forwards - yield1
    X = sm.add_constant(X)
    results = sm.OLS(rx, X).fit()
    return results


# Estimate the Fama-Bliss regressions and report a, b, R2, se(a), se(b) for n = 2, 3, 4, 5
results = {}

for n in range(2, 6):
    # Define the dependent and independent variables for the first regression
    rx_nt1 = data[f'excess_return{n}']
    f_n_y1 = data[f'f{n}'] - data['y1']
    
    # Fit the first regression model
    model1 = regression(data[f'f{n}'], data['y1'], rx_nt1)
    
    # Define the dependent and independent variables for the second regression
    y1_tn1_y1 = data['y1'].shift(-n+1) - data['y1']
    y1_tn1_y1 = y1_tn1_y1.dropna()
    f_n_y1_shifted = f_n_y1.shift(-n+1).dropna()
    
    # Fit the second regression model
    model2 = regression(f_n_y1_shifted, data['y1'].shift(-n+1).dropna(), y1_tn1_y1)
    
    # Store the results
    results[n] = {
        'a_x': model1.params['const'],
        'b_x': model1.params[0],
        'R2_x': model1.rsquared,
        'se(a_x)': model1.bse['const'],
        'se(b_x)': model1.bse[0],
        'a_y': model2.params['const'],
        'b_y': model2.params[0],
        'R2_y': model2.rsquared,
        'se(a_y)': model2.bse['const'],
        'se(b_y)': model2.bse[0]
    }

# Display the results
for n in results:
    print(f"Results for n = {n}:")
    print(f"a_x: {results[n]['a_x']}, b_x: {results[n]['b_x']}, R2_x: {results[n]['R2_x']}, se(a_x): {results[n]['se(a_x)']}, se(b_x): {results[n]['se(b_x)']}")
    print(f"a_y: {results[n]['a_y']}, b_y: {results[n]['b_y']}, R2_y: {results[n]['R2_y']}, se(a_y): {results[n]['se(a_y)']}, se(b_y): {results[n]['se(b_y)']}")
    print()
