
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