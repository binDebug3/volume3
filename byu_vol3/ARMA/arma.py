# Name
# Date
# Class

from scipy.stats.distributions import norm
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pydataset import data as pydata
from statsmodels.tsa.stattools import arma_order_select_ic as order_select
import pandas as pd
import time
import statsmodels.api as sm
from statsmodels.tsa.api import VARMAX
from statsmodels.tsa.base.datetools import dates_from_str
from statsmodels.tsa.arima.model import ARIMAResults


def kalman(F, Q, H, time_series):
    # Get dimensions
    dim_states = F.shape[0]

    # Initialize variables
    # covs[i] = P_{i | i-1}
    covs = np.zeros((len(time_series), dim_states, dim_states))
    mus = np.zeros((len(time_series), dim_states))

    # Solve of for first mu and cov
    covs[0] = np.linalg.solve(np.eye(dim_states**2) - np.kron(F,F),np.eye(dim_states**2)).dot(Q.flatten()).reshape(
            (dim_states,dim_states))
    mus[0] = np.zeros((dim_states,))

    # Update Kalman Filter
    for i in range(1, len(time_series)):
        #FIXME
        t1 = np.linalg.solve(H.dot(covs[i-1]).dot(H.T),np.eye(H.shape[0]))
        t2 = covs[i-1].dot(H.T.dot(t1.dot(H.dot(covs[i-1]))))
        covs[i] = F.dot((covs[i-1] - t2).dot(F.T)) + Q
        mus[i] = F.dot(mus[i-1]) + F.dot(covs[i-1].dot(H.T.dot(t1))).dot(
                time_series[i-1] - H.dot(mus[i-1]))
    return mus, covs

def state_space_rep(phis, thetas, mu, std):
    # Initialize variables
    dim_states = max(len(phis), len(thetas)+1)
    dim_time_series = 1 #hardcoded for 1d time_series

    F = np.zeros((dim_states,dim_states))
    Q = np.zeros((dim_states, dim_states))
    H = np.zeros((dim_time_series, dim_states))

    # Create F
    F[0][:len(phis)] = phis
    F[1:,:-1] = np.eye(dim_states - 1)
    # Create Q
    Q[0][0] = std**2
    # Create H
    H[0][0] = 1.
    H[0][1:len(thetas)+1] = thetas

    return F, Q, H, dim_states, dim_time_series

##############################################################################

def arma_forecast_naive(filename='weather.npy', p=2, q=1, n=20):
    """
    Perform ARMA(1,1) on data. Let error terms be drawn from
    a standard normal and let all constants be 1.
    Predict n values and plot original data with predictions.

    Parameters:
        filename (str): data filename
        p (int): order of autoregressive model
        q (int): order of moving average model
        n (int): number of future predictions
    """
    raise NotImplementedError("Problem 1 Incomplete")

def arma_likelihood(filename='weather.npy', phis=np.array([0.9]), thetas=np.array([0]), mu=17., std=0.4):
    """
    Transfer the ARMA model into state space.
    Return the log-likelihood of the ARMA model.

    Parameters:
        filename (str): data filename
        phis (ndarray): coefficients of autoregressive model
        thetas (ndarray): coefficients of moving average model
        mu (float): mean of errorm
        std (float): standard deviation of error

    Return:
        log_likelihood (float)
    """
    raise NotImplementedError("Problem 2 Incomplete")

def model_identification(filename='weather.npy', p_max=4, q_max=4):
    """
    Identify parameters to minimize AIC of ARMA(p,q) model

    Parameters:
        filename (str): data filename
        i (int): maximum order of autoregressive model
        j (int): maximum order of moving average model

    Returns:
        phis (ndarray (p,)): coefficients for AR(p)
        thetas (ndarray (q,)): coefficients for MA(q)
        mu (float): mean of error
        std (float): std of error
    """
    raise NotImplementedError("Problem 3 Incomplete")

def arma_forecast(filename='weather.npy', phis=np.array([.72]), thetas=np.array([-.26]), mu=.36, std=1.55, n=30):
    """
    Forecast future observations of data.

    Parameters:
        filename (str): data filename
        phis (ndarray (p,)): coefficients of AR(p)
        thetas (ndarray (q,)): coefficients of MA(q)
        mu (float): mean of ARMA model
        std (float): standard deviation of ARMA model
        n (int): number of forecast observations

    Returns:
        new_mus (ndarray (n,)): future means
        new_covs (ndarray (n,)): future standard deviations
    """
    raise NotImplementedError("Problem 4 Incomplete")

def sm_arma(filename='weather.npy', p_max=3, q_max=3, n=30):
    """
    Build an ARMA model with statsmodel and
    predict future n values.

    Parameters:
        filename (str): data filename
        i (int): maximum order of autoregressive model
        j (int): maximum order of moving average model
        n (int): number of values to predict

    Return:
        aic (float): aic of optimal model
    """
    raise NotImplementedError("Problem 5 Incomplete")

def sm_varma(start='1959-09-30', end='2012-09-30'):
    """
    Build an VARMAX model with statsmodel and
    predict future n values.

    Parameters:
        start (str): the data at which to begin forecasting
        end (str): the date at which to stop forecasting

    Return:
        aic (float): aic of optimal model
    """
    raise NotImplementedError("Problem 6 Incomplete")

def manaus(start='1983-01-31', end='1995-01-31', i=4, j=4):
    """
    Plot the ARMA(p,q) model of the River Negro height
    data using statsmodels built-in ARMA class.

    Parameters:
        start (str): the data at which to begin forecasting
        end (str): the date at which to stop forecasting
        i (int): max_ar parameter
        j (int): max_ma parameter
    Return:
        aic_min_order (tuple): optimal order based on AIC
        bic_min_order (tuple): optimal order based on BIC
    """
    # Get dataset
    raw = pydata('manaus')
    # Convert to DateTimeIndex
    manaus = pd.DataFrame(raw.values,index=pd.date_range('1903-01','1993-01',freq='M'))
    manaus = manaus.drop(0,axis=1)
    # Set new column title
    manaus.columns = ['Water Level']

    raise NotImplementedError("Problem 7 Incomplete")
