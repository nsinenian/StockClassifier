
import numpy as np
import pandas as pd

import math
import os

from datetime import datetime
from datetime import timedelta

# Import our custom that retrieves financial market data from #http://financialmodelingprep.com
import api
import api.tickers as tickers  # Information about stock, ETF, etc. tickers
import api.stocks  as stocks   # Information about stocks and stock indices

from api.stocks import Stock   # Information about a particular stock
from api.stocks import Index   # Information about a particular index

from api.stocks import TIMEDELTA_QUARTER
from api.stocks import TIMEDELTA_MONTH
from api.stocks import TIMEDELTA_YEAR

from sklearn.preprocessing import MinMaxScaler

# We can use a window size of 5 to smooth out some of the noise
# while retaining fidelity with respect to the general trends
# observed in the data.

# Let's compute the gain of the S&P 500 index over the last 
# quarter. We will compare each sualtock's performance over the
# last quarte to this value.

# Note that the S&P 500 index is a capital-weighted index, so
# larger-cap stocks make up a larger portion of the fraction.
# Essentially the question we are asking is whether any given
# stock will outperform the index or "market". Investors can 
# choose to invest in index-tracking ETFs instead of a given stock.

def get_sp500_gain_for_interval(interval,offset,output=False):
    """ Get the gain for the S&P 500 over the specified interval
    
    Args:
        interval: The time interval for gain calculation as a datetime.timedelta
        offset: The offset of interval relative to today as a datetime.timedelta
        
    Returns:
        The fractional gain or loss over the interval.
    """
    end_date   = datetime.today().date()
    if offset is not None:
        end_date -= offset

    start_date = end_date - interval
    
    sp500_index = Index('^GSPC')
    sp500_time_series = sp500_index.get_historical_data(start_date,end_date)
    sp500_close = sp500_time_series['close'] 
    sp500_close_smooth = sp500_close.ewm(span=5).mean()

    sp500_end_of_interval = round(sp500_close_smooth.values[-1],2)
    sp500_start_of_interval = round(sp500_close_smooth.values[0],2)
    sp500_gain_during_interval = round(sp500_end_of_interval / sp500_start_of_interval,4)
    
    if output:
        print("Value start of interval: ",sp500_start_of_interval)
        print("Value end of interval: ",sp500_end_of_interval)
        print("Approximate gain: ",sp500_gain_during_interval)
        print("")
    
        plt.plot(sp500_close.values,'.',label='Close',color=plt.cm.viridis(.4),markerfacecolor=plt.cm.viridis(.9),markersize=12)
        plt.plot(sp500_close_smooth.values,'-',label='5-day EWMA',color=plt.cm.viridis(0.3))
        plt.title('S&P 500 Index')
        plt.xlabel('Trading days')
        plt.ylabel('Close Price')
        plt.legend()
        
    return sp500_gain_during_interval

def get_stock_label_func(p_interval,p_offset):
    """ Generates a function that returns a stock label 
            
    Args:
        p_interval: The prediction interval as a datetime.timedelta
        p_offset: The offset of d_interval relative to today as a datetime.timedelta

    Returns:
        A function that can be called (for a specified stock) to get the stock label
    """
    ref_value = get_sp500_gain_for_interval(p_interval,p_offset,output=False) 
    
    def get_stock_label(symbol,output=False):
        """ Generates a stock label for training and/or validation dataset
            
        Raises:
            LookupError: If the stock could not be found

        Returns:
            An integer value (0 or 1) indicating the stock label
        """
        end_date = datetime.today().date()
        if p_offset is not None:
            end_date -= p_offset

        start_date = end_date - p_interval
        
        try:
            close_price = stock_data[symbol].get_historical_prices(start_date,end_date)['close']
        except:
            close_price = Stock(symbol).get_historical_prices(start_date,end_date)['close']
            
        close_price = close_price.ewm(span=3).mean()

        stock_gain = close_price.values[-1] / close_price.values[0]
        stock_relative_gain = round( (stock_gain) / ref_value,4)
        
        stock_label = 0 if stock_relative_gain < 1 else 1
        
        if output:
            print("Gain during interval: ",round(stock_gain,4))
            print("Reference value: ",ref_value)
            print("Gain relative to reference value: ",stock_relative_gain)
            print("Label: ",stock_label)
        
        return stock_label
        
    return get_stock_label
