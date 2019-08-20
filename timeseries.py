'''Timeseries module gathers a few standard statistical functions 
to facilitate Time Series ops
'''

import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen


def kpss_test(timeseries):
    '''
    Kwiatkowski–Phillips–Schmidt–Shin (KPSS) tests are used for testing a null hypothesis,
    that an observable time series is stationary around a deterministic trend
    (i.e. trend-stationary) against the alternative of a unit root. 
    
    Args: 
        timeseries - array of values
        
    returns nothing
    
    '''
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c')
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print (kpss_output)


def decompose(timeseries):
    ''' 
    Classical decompose function to render timeseries stationary 
    Plots decomposition, trend, seasonal, residual
    Returns: decomposition, trend, seasonal, residual timeseries
    '''
    decomposition = seasonal_decompose(timeseries)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    plt.plot(timeseries, color = 'blue', label='Original')
    plt.plot(trend, color = 'red', label='Trend')
    plt.plot(seasonal, color = 'green', label='Seasonality')
    plt.plot(residual, color = 'orange', label='Residuals')
    plt.tight_layout()
    plt.legend()
    plt.show()
    return decomposition, trend, seasonal, residual

def acf(ts_log_diff):
    '''
    Function acf computes and plots estimates of the autocovariance or autocorrelation function. 
    Function pacf is the function used for the partial autocorrelations. 
    Function ccf computes the cross-correlation or cross-covariance of two univariate series.
    '''
    #ACF and PACF plots:
    lag_acf = acf(ts_log_diff, nlags=20)
    lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')

    #Plot ACF:
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    plt.show()

    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()

def ar(ts_log):
    '''
    AR is an acronym that stands for AutoRegressive  
    
    Plots fitted function
    Takes timeseries argument
    Returns results
    '''
    model = ARIMA(ts_log, order=(2, 1, 0))  
    results_AR = model.fit(disp=-1)  
    plt.plot(ts_log)
    plt.plot(results_AR.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_AR.fittedvalues-ts_log)**2))
    results_AR
    
def ma(ts_log):
    '''
    MA is an acronym that stands for Moving Average. 
    
    Plots fitted function
    Takes timeseries argument
    Returns results
    '''
    model = ARIMA(ts_log, order=(0, 1, 2))  
    results_MA = model.fit(disp=-1)  
    plt.plot(ts_log)
    plt.plot(results_MA.fittedvalues, color='red')
    plt.title('RSS: %.4f'% sum((results_MA.fittedvalues-ts_log)**2))
    results_MA

def arima(ts_log, order=(2, 1, 2)):
    '''
    ARIMA is an acronym that stands for AutoRegressive Integrated Moving Average. 
    It is a generalization of the simpler AutoRegressive Moving Average
    Adds the notion of integration. 
    
    Plots fitted function
    Args:
        timeseries 
        The (p,d,q) order of the model for the number of:
            AR parameters, 
            differences,
            MA parameters to use.
    
    Returns results
        
    Example:
    
    result = timeseries.arima(values, (5,1,0))
    This sets the lag value to 5 for autoregression, 
    uses a difference order of 1 to make the time series stationary, 
    and uses a moving average model of 0.
            
    '''
    model = ARIMA(ts_log, order)
    results_ARIMA = model.fit(disp=-1)
    
    plt.plot(ts_log)
    plt.plot(results_ARIMA.fittedvalues, color='red')
#     plt.title('RSS: %.4f'% sum((results_ARIMA.fittedvalues-ts_log)**2))
    
    residuals = pd.DataFrame(results_ARIMA.resid)
    residuals.plot()
    plt.show()
    residuals.plot(kind='kde')
    plt.show()
    print(residuals.describe())
    return results_ARIMA
    

maxlag=12
test = 'ssr_chi2test'
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):    
    """
    Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

    data      : pandas dataframe containing the time series variables
    variables : list containing names of the time series variables.
    """
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df

#grangers_causation_matrix(df, variables = df.columns)        



def cointegration_test(df, alpha=0.05):
    """
    Perform Johanson's Cointegration Test and Report Summary
    """
    out = coint_johansen(df,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Signif  \n', '--'*20)
    for col, trace, cvt in zip(df.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

#cointegration_test(df)

# alternative ADCF test with result printouts
def adfuller_test(series, signif=0.05, name='', verbose=False):
    """
    Perform ADFuller to test for Stationarity of given series and print report
    """
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']
    def adjust(val, length= 6): return str(val).ljust(length)

    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")


from pandas import DataFrame
from pandas import concat

# define lstm shifted dataframe as in article 
# https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learning dataset.
	Arguments:
		data: Sequence of observations as a list or NumPy array.
		n_in: Number of lag observations as input (X).
		n_out: Number of observations as output (y).
		dropnan: Boolean whether or not to drop rows with NaN values.
	Returns:
		Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

