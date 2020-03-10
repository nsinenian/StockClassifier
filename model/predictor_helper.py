
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

def get_stock_cat_features_func(d_interval,d_offset):
    """ Generates a function that returns categorical features for a stock
            
    Args:
        d_interval: The data interval as a datetime.timedelta (e.g., 6*TIMEDELTA_QUARTER for 6 quarters of data)
        d_offset: The offset of d_interval relative to today as a datetime.timedelta

    Returns:
        A tuple consisting of array that specifies which categorical feature are to be embedded (as opposed to 
        stand-alone features) and a function that can be called to get categorical features for a stock. The
        array should include the embedding dimension for the feature, or 0 if it is not to be embedded.
    """
    # Get list of sectors and map each sector to an index (normalized)
    sector_list = np.array(['Energy',
                            'Consumer Cyclical',
                            'Real Estate',
                            'Utilities',
                            'Industrials',
                            'Basic Materials',
                            'Technology',
                            'Healthcare',
                            'Financial Services',
                            'Consumer Defensive'])
    
    industry_list = np.array(['Agriculture',
                              'Insurance - Life',
                              'Medical Diagnostics & Research',
                              'Online Media',
                              'Oil & Gas - E&P',
                              'Homebuilding & Construction',
                              'Oil & Gas - Drilling',
                              'Oil & Gas - Refining & Marketing',
                              'Advertising & Marketing Services',
                              'Utilities - Regulated',
                              'Consulting & Outsourcing',
                              'Autos',
                              'Travel & Leisure',
                              'Oil & Gas - Integrated',
                              'Brokers & Exchanges',
                              'Application Software',
                              'Manufacturing - Apparel & Furniture',
                              'Medical Devices',
                              'Retail - Apparel & Specialty',
                              'Oil & Gas - Services',
                              'Consumer Packaged Goods',
                              'Insurance - Property & Casualty',
                              'Drug Manufacturers',
                              'Real Estate Services',
                              'Airlines',
                              'Insurance',
                              'Farm & Construction Machinery',
                              'Semiconductors',
                              'Medical Distribution',
                              'Steel',
                              'Restaurants',
                              'Waste Management',
                              'Entertainment',
                              'Chemicals',
                              'REITs',
                              'Insurance - Specialty',
                              'Metals & Mining',
                              'Retail - Defensive',
                              'Biotechnology',
                              'Conglomerates',
                              'Utilities - Independent Power Producers',
                              'Building Materials',
                              'Health Care Plans',
                              'Tobacco Products',
                              'Oil & Gas - Midstream',
                              'Transportation & Logistics',
                              'Business Services',
                              'Truck Manufacturing',
                              'Beverages - Non-Alcoholic',
                              'Personal Services',
                              'Banks',
                              'Medical Instruments & Equipment',
                              'Industrial Distribution',
                              'Asset Management',
                              'Forest Products',
                              'Industrial Products',
                              'Communication Equipment',
                              'Packaging & Containers',
                              'Credit Services',
                              'Engineering & Construction',
                              'Computer Hardware',
                              'Aerospace & Defense',
                              'Beverages - Alcoholic',
                              'Health Care Providers',
                              'Communication Services',
                              'Employment Services'])
    
    sector_dict = { sector : i for i, sector in enumerate(sector_list)}
    industry_dict = { industry : i for i, industry in enumerate(industry_list)}

    # SP500 range is on the order of USD 1B to USD 1T, scale accordingly
    MIN_MARKET_CAP = 1.0e9
    MAX_MARKET_CAP = 1.0e12
    
    # For the specified d_offset we will make a cyclic label corresponding
    # to the month of the year (1-12) using sine and cosine functions
    end_date = datetime.today().date()
    if d_offset is not None:
        end_date -= d_offset
       
    # Encoding which month (fractional) the data ends. This is universal
    # in that it will work for any intervals and offsets of interest.
    month_decimal = end_date.month + end_date.day/30.0;
    month_angle = 2*math.pi*month_decimal/12.0
    month_x = math.cos(month_angle)
    month_y = math.sin(month_angle)
    
    # The feature structure (# of embeddings for each feature or 0 if not to be embedded)
    cat_feature_embeddings = [len(sector_list)+1, len(industry_list)+1, 0, 0]
        
    def get_stock_cat_features(symbol):
        """ Gets categorical features associated with a paticular stock
        
        Args:
            symbol: A stock ticker symbol such as 'AAPL' or 'T'

        Raises:
            LookupError: If any categorical feature is unavailable of NaN for the stock.

        Returns:
            Categorical stock features as an array of M x 1 values (for M features). Categorical
            features to be embedded are appear first in the returned array
        """    
        try:
            profile = stock_data[symbol].get_company_profile()
        except:
            profile = Stock(symbol).get_company_profile()
            
        sector = profile.loc[symbol,'sector']
        industry = profile.loc[symbol,'industry']
        
        try:
            sector_feature = sector_dict[sector]
        except:
            sector_feature = len(sector_list)
            
        try:
            industry_feature = industry_dict[industry]
        except:
            industry_feature = len(industry_list)
        
        # Get market capitalization corresponding to d_offset
        if d_offset is None:
            quarter_offset = 0
        else:
            quarter_offset = int(d_offset / TIMEDELTA_QUARTER)
            
        # Get the "latest" key metrics as of the data interval
        try:
            key_metrics = stock_data[symbol].get_key_metrics(quarters=1,offset=quarter_offset)
        except:
            key_metrics = Stock(symbol).get_key_metrics(quarters=1,offset=quarter_offset)
            
        market_cap = key_metrics['Market Cap'][0]
        
        # Scalar value (approx 0-1) corresponding to market capitalization
        market_cap_feature = math.log(float(market_cap)/MIN_MARKET_CAP,MAX_MARKET_CAP/MIN_MARKET_CAP)
        
        features = np.array( [sector_feature, industry_feature, market_cap_feature, month_x, month_y],dtype='float32')
    
        if np.isnan(features).any():
            raise LookupError
            
        return features
  
    return cat_feature_embeddings, get_stock_cat_features

def get_stock_daily_features_func(d_interval,d_offset):
    """ Generates a function that returns daily features for a stock
            
    Args:
        d_interval: The data interval as a datetime.timedelta (e.g., 6*TIMEDELTA_QUARTER for 6 quarters of data)
        d_offset: The offset of d_interval relative to today as a datetime.timedelta

    Returns:
        A function that can be called to get daily features for a stock
    """
    end_date = datetime.today().date()
    if d_offset is not None:
        end_date -= d_offset

    start_date = end_date - d_interval
    
    # Th S&P 500 index will have a closing value for every trading day. Each of the stocks
    # should also have the same number of values unless they were suspended and didn't trade or 
    # recently became public.
    trading_day_count = len(Index('^GSPC').get_historical_data(start_date,end_date))
    
    def get_stock_daily_features(symbol,output=False):
        """ Gets daily features associated with a paticular stock

        Args:
            symbol: A stock ticker symbol such as 'AAPL' or 'T'

        Raises:
            LookupError: If any categorical feature is unavailable of NaN for the stock.

        Returns:
            Daily stock features as an array of M x N values (for M features with N values)
        """
        try:
            historical_data = stock_data[symbol].get_historical_prices(start_date,end_date)
        except:
            historical_data = Stock(symbol).get_historical_prices(start_date,end_date)
        
        # Smooth and normalize closing price relative to initial price for data set
        close_price = historical_data['close'].ewm(span=5).mean()
        close_price = close_price / close_price.iat[0]
        close_price = np.log10(close_price)
    
        # Smooth and normalize volume relative to average volume
        average_volume = historical_data['volume'].mean()
        volume = historical_data['volume'].ewm(span=5).mean()
        volume = volume / average_volume
        volume = np.log10(volume+1e-6)
        
        # Ensure equal lengths of data (nothing missing)
        if len(volume) != len(close_price):
            raise LookupError
        
        # Ensure we have the right number of data points for the period
        if len(close_price) != trading_day_count:
            raise LookupError
        
        features = np.array([close_price, volume],dtype='float32')
        
        if np.isnan(features).any():
            raise LookupError
            
        return features
        
    return get_stock_daily_features

def get_stock_quarterly_features_func(d_interval,d_offset):
    """ Generates a function that returns quarterly features for a stock
            
    Args:
        d_interval: The data interval as a datetime.timedelta (e.g., 6*TIMEDELTA_QUARTER for 6 quarters of data)
        d_offset: The offset of d_interval relative to today

    Returns:
        A function that can be called to get quarterly features for a stock
    """
    
    # Quarterly features can only be used if prediction intervals 
    if d_interval < TIMEDELTA_QUARTER:
        raise ValueError("The specified data interval is less than one quarter")
    
    end_date = datetime.today().date()    
    if d_offset is not None:
        end_date -= d_offset
        
    start_date = end_date - d_interval
    
    quarter_count =  int(d_interval / TIMEDELTA_QUARTER)
    
    if d_offset is None:
        quarter_offset = 0
    else:
        quarter_offset = int(d_offset / TIMEDELTA_QUARTER)
        
    price_to_earnings_scaler = MinMaxScaler()
    price_to_sales_scaler = MinMaxScaler()
    price_to_free_cash_flow_scaler = MinMaxScaler()
    dividend_yield_scaler = MinMaxScaler()

    price_to_earnings_scaler.fit_transform(np.array([0,200]).reshape(-1, 1))
    price_to_sales_scaler.fit_transform(np.array([0,200]).reshape(-1, 1))
    price_to_free_cash_flow_scaler.fit_transform(np.array([0,200]).reshape(-1, 1))
    dividend_yield_scaler.fit_transform(np.array([0,1]).reshape(-1, 1))
    
    def get_stock_quarterly_features(symbol):
        """ Gets quarterly features associated with a paticular stock
        
        Args:
            symbol: A stock ticker symbol such as 'AAPL' or 'T'

        Raises:
            LookupError: If any categorical feature is unavailable of NaN for the stock.

        Returns:
            Quarterly stock features as an array of M x N values (for M features and N values)
        """
        try:
            key_metrics = stock_data[symbol].get_key_metrics(quarter_count,quarter_offset)
        except:
            key_metrics = Stock(symbol).get_key_metrics(quarter_count,quarter_offset)
        
        key_metrics['PE ratio'] = price_to_earnings_scaler.transform(key_metrics['PE ratio'].values.reshape(-1,1))
        key_metrics['Price to Sales Ratio'] = price_to_sales_scaler.transform(key_metrics['Price to Sales Ratio'].values.reshape(-1,1))
        key_metrics['PFCF ratio'] = price_to_free_cash_flow_scaler.transform(key_metrics['PFCF ratio'].values.reshape(-1,1))   
        key_metrics['Dividend Yield'] = dividend_yield_scaler.transform(key_metrics['Dividend Yield'].values.reshape(-1,1))

        try:
            financials = stock_data[symbol].get_income_statement(quarter_count,quarter_offset)
        except:
            financials = Stock(symbol).get_income_statement(quarter_count,quarter_offset)
        
        # Apply scaling for diluted EPS (we want growth relative to t=0)
        financials['EPS Diluted'] = ( financials['EPS Diluted'].astype(dtype='float32') / float(financials['EPS Diluted'].iat[0]) )
    
        features = np.array([
            key_metrics['PE ratio'],
            key_metrics['Price to Sales Ratio'],
            key_metrics['PFCF ratio'],
            key_metrics['Dividend Yield'],
            financials['EPS Diluted'],
            financials['Revenue Growth'],
        ],dtype='float32')
        
        if np.isnan(features).any():
            raise LookupError
           
        return features
        
    return get_stock_quarterly_features
