# Copyright (c) 2020 Nareg Sinenian
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

import requests
import json
import numpy as np
import pandas as pd

import bs4 as bs
import pickle
import requests

import api
from   api import __cache_dir__

from datetime import datetime
from datetime import timedelta

from api import TIMEDELTA_QUARTER 
from api import TIMEDELTA_MONTH
from api import TIMEDELTA_YEAR  

TRADING_DAYS_PER_QUARTER = 63

class Index:

    def __init__(self,symbol):
        self.symbol = symbol
        self.cached_historical_data = None
        self.cache_start_date = None
        self.cache_end_date = None

    def cache_data(self,start,end):
        historical_data_df = self.get_historical_data(start,end)
        self.cached_historical_data = historical_data_df
        self.cache_start_date = start
        self.cache_end_date = end

    def clear_cache(self):
        self.cached_historical_data = None
        self.cache_start_date = None
        self.cache_end_date = None

    def get_historical_data(self,start,end):    
        if end < start or end > datetime.today().date():
            raise ValueError

        start_date = start.strftime("%Y-%m-%d")
        end_date = end.strftime("%Y-%m-%d")

        try:
            if end > self.cache_end_date or start < self.cache_start_date:
                raise LookupError

            stockdata_df = pd.DataFrame(self.cached_historical_data)
        except:
            try:
                # Transmit HTTP GET request and JSON-formatted response
                response = requests.get(f'https://financialmodelingprep.com/api/v3/historical-price-full/index/{self.symbol}?from={start_date}&to={end_date}')
            except:
                raise LookupError

            stockdata_df = pd.DataFrame(response.json()['historical'])
        
            # Convert date column to a datetime object
            stockdata_df['date'] = pd.to_datetime(stockdata_df['date'],infer_datetime_format=True)

            # Reverse so that most recent data point is at the end
            stockdata_df = stockdata_df.iloc[::-1,:].reset_index(drop=True)

        return stockdata_df[(stockdata_df.date < end_date) & (stockdata_df.date > start_date)].reset_index(drop=True)

class Stock():

    def __init__(self,symbol):
        self.symbol = symbol

        self.cached_company_profile = None
        self.cached_historical_prices = None
        self.cached_key_metrics = None
        self.cached_income_statement = None

        self.cache_start_date = None
        self.cache_end_date = None

    def cache_data(self,start,end):
        company_profile_df = self.get_company_profile()
        historical_prices_df = self.get_historical_prices(start,end)

        quarters = int( (end-start).days / TRADING_DAYS_PER_QUARTER )
        offset   = int( (datetime.now().date() - end).days / TRADING_DAYS_PER_QUARTER )

        key_metrics_df = self.get_key_metrics(quarters,offset)
        income_statement_df = self.get_income_statement(quarters,offset)

        self.cached_company_profile = company_profile_df
        self.cached_historical_prices = historical_prices_df
        self.cached_key_metrics = key_metrics_df
        self.cached_income_statement = income_statement_df

        self.cache_start_date = start
        self.cache_end_date = end

    def clear_cache(self):
        self.cached_company_profile = None
        self.cached_historical_prices = None
        self.cached_key_metrics = None
        self.cached_income_statement = None

        self.cache_start_date = None
        self.cache_end_date = None

    def get_company_profile(self):
        
        if self.cached_company_profile is not None:
            return self.cached_company_profile

        try:
            response = requests.get(f'https://financialmodelingprep.com/api/v3/company/profile/{self.symbol}')
        except:
            raise LookupError

        stockdata_df = pd.DataFrame(response.json())
        stockdata_df.drop(labels=['symbol'],axis=1,inplace=True)

        stockdata_df = stockdata_df.transpose()
        stockdata_df.rename(index={'profile':self.symbol}, inplace=True)
        
        return stockdata_df

    # Get historical daily time-series data for a particular stock
    def get_historical_prices(self,start,end):
                
        if end < start or end > datetime.today().date():
            raise ValueError

        start_date = start.strftime("%Y-%m-%d")
        end_date = end.strftime("%Y-%m-%d")

        try:
            if end > self.cache_end_date or start < self.cache_start_date:
                raise LookupError

            stockdata_df = pd.DataFrame(self.cached_historical_prices)
        except:
            try:
                # Transmit HTTP GET request and JSON-formatted response
                response = requests.get(f'https://financialmodelingprep.com/api/v3/historical-price-full/{self.symbol}?from={start_date}&to={end_date}')
            except:
                raise LookupError

            stockdata_df = pd.DataFrame(response.json()['historical'])
        
            # Convert date column to a datetime object
            stockdata_df['date'] = pd.to_datetime(stockdata_df['date'],infer_datetime_format=True)

            # Reverse so that most recent data point is at the end
            # No need for this for stocks (but we do for indices; the API endpoints are inconsistent!)
            #stockdata_df = stockdata_df.iloc[::-1,:].reset_index(drop=True)

        return stockdata_df[(stockdata_df.date < end_date) & (stockdata_df.date > start_date)].reset_index(drop=True)


    # Get historical quarterly time-series data for a particular stock
    def get_key_metrics(self,quarters,offset=0):
        try:
            stockdata_df = pd.DataFrame(self.cached_key_metrics)

            if len(stockdata_df) < (quarters + offset):
                raise LookupError       
        except:   
            try:
                response = requests.get(f'https://financialmodelingprep.com/api/v3/company-key-metrics/{self.symbol}?period=quarter')
            except:
                raise LookupError

            stockdata_df = pd.DataFrame(response.json()['metrics'])

            # Reverse so that most recent data point is at the end
            stockdata_df = stockdata_df.iloc[::-1,:].reset_index(drop=True)

            # Convert date column to a datetime object
            stockdata_df['date'] = pd.to_datetime(stockdata_df['date'],infer_datetime_format=True)

        # Retain the most quarters we want
        return stockdata_df.iloc[-(quarters+offset):len(stockdata_df)-offset,:].reset_index(drop=True)


    # Get historical quarterly income statements for a particular stock
    def get_income_statement(self,quarters,offset=0):
        try:
            stockdata_df = pd.DataFrame(self.cached_income_statement)

            if len(stockdata_df) < (quarters + offset):
                raise LookupError       
        except:   
            try:
                response = requests.get(f'https://financialmodelingprep.com/api/v3/financials/income-statement/{self.symbol}?period=quarter')
            except:
                raise LookupError

            stockdata_df = pd.DataFrame(response.json()['financials'])

            # Reverse so that most recent data point is at the end
            stockdata_df = stockdata_df.iloc[::-1,:].reset_index(drop=True)

            # Convert date column to a datetime object
            stockdata_df['date'] = pd.to_datetime(stockdata_df['date'],infer_datetime_format=True)

        # Retain the most quarters we want
        return stockdata_df.iloc[-(quarters+offset):len(stockdata_df)-offset,:].reset_index(drop=True)

def get_sector_performance():
    try:
        response = requests.get('https://financialmodelingprep.com/api/v3/stock/sectors-performance')
        performance_df = pd.DataFrame(response.json()['sectorPerformance'])
        performance_df.set_index('sector',inplace=True)
    except:
        raise LookupError

    return performance_df

def get_sector_list():
    sector_performance_df = get_sector_performance().transpose()
    return sector_performance_df.columns.values