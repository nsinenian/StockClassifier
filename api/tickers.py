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

def get_etf_tickers():

    path = os.path.join(__cache_dir__,'tickers_etf.npy')
    
    if os.path.exists(path):
        return list(np.load(path,allow_pickle=True))

    response = requests.get('https://financialmodelingprep.com/api/v3/symbol/available-etfs')
    etf_tickers_df = pd.DataFrame(response.json())
    np.save(path,etf_tickers_df['symbol'].values)
    
    return list(etf_tickers_df['symbol'].values)

def get_mutual_fund_tickers():
    
    path = os.path.join(__cache_dir__,'tickers_mutual_funds.npy')
    
    if os.path.exists(path):
        return list(np.load(path,allow_pickle=True))
    
    response = requests.get('https://financialmodelingprep.com/api/v3/symbol/available-mutual-funds')
    mutual_fund_tickers_df = pd.DataFrame(response.json())
    np.save(path,mutual_fund_tickers_df['symbol'].values)
    
    return list(mutual_fund_tickers_df['symbol'].values)

def get_euronext_tickers():

    path = os.path.join(__cache_dir__,'tickers_euronext.npy')
    
    if os.path.exists(path):
        return list(np.load(path,allow_pickle=True))
    
    response = requests.get('https://financialmodelingprep.com/api/v3/symbol/available-euronext')
    euronext_tickers_df = pd.DataFrame(response.json())
    np.save(path,euronext_tickers_df['symbol'].values)
        
    return list(euronext_tickers_df['symbol'].values)
        
def get_tsx_tickers():

    path = os.path.join(__cache_dir__,'tickers_tsx.npy')
    
    if os.path.exists(path):
        return list(np.load(path,allow_pickle=True))
    
    response = requests.get('https://financialmodelingprep.com/api/v3/symbol/available-tsx')
    tsx_tickers_df = pd.DataFrame(response.json())
    np.save(path,tsx_tickers_df['symbol'].values)
    
    return list(tsx_tickers_df['symbol'].values)

def get_us_stock_tickers():

    path = os.path.join(__cache_dir__,'tickers_us_stocks.npy')
    
    if os.path.exists(path):
        return list(np.load(path,allow_pickle=True))
    
    available_tickers = set(get_available_tickers())

    tickers_to_remove = list(get_etf_tickers() + get_mutual_fund_tickers() + get_euronext_tickers() + get_tsx_tickers())

    for ticker in tickers_to_remove:
        if ticker in available_tickers:
            available_tickers.remove(ticker)
            
    np.save(path,np.array(list(available_tickers)))
        
    return list(available_tickers)

def get_available_tickers():

    path = os.path.join(__cache_dir__,'tickers_all_available.npy')
    
    if os.path.exists(path):
        return list(np.load(path,allow_pickle=True))
    
    # This includes stocks, mutual funds and ETFs
    response = requests.get('https://financialmodelingprep.com/api/v3/company/stock/list')
    response_data = response.json()
    available_tickers_df = pd.DataFrame(response_data['symbolsList'])
    np.save(path,available_tickers_df['symbol'].values)
    
    return list(available_tickers_df['symbol'].values)

def get_sp500_tickers():
    
    path = os.path.join(__cache_dir__,'tickers_sp500.npy')
    
    if os.path.exists(path):
        return list(np.load(path,allow_pickle=True))
    
    # Use beautifulsoup to find S&P 500 table on Wikipedia page
    response = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(response.text, 'html.parser') 
    table = soup.find('table', {'class': 'wikitable sortable'})
    
    # Read table into pandas dataframe
    sp500_list_df = pd.read_html(str(table))[0]
    
    # The list of tickers was generated from Wikipedia, and as such has some noteworthy nuances.
    # It includes some repeats for Class A & B shares (e.g., GOOG vs GOOGL). For this
    # reason, we remove duplicate symbols based on CIK (a unique key given to an entity by the SEC)
    sp500_list_df.drop_duplicates(subset=['CIK'],inplace=True)
    
    # Retain available tickers only
    sp500_list = list(sp500_list_df['Symbol'].values)
    
    available_tickers = set(get_available_tickers())
    sp500_tickers = [ticker for ticker in sp500_list if ticker in available_tickers]
    np.save(path,sp500_tickers)
    
    return sp500_tickers