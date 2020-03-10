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

from sklearn.metrics import f1_score

from datetime import datetime
from datetime import timedelta

from api import TIMEDELTA_QUARTER 
from api import TIMEDELTA_MONTH
from api import TIMEDELTA_YEAR  

from api.tickers import get_sp500_tickers

from model.benchmark_helper import get_stock_label_func
from model.benchmark_helper import get_sp500_gain_for_interval

def confusion_matrix(y_predictions,y_labels,output=False):
    """ Calculates a confusion matrix given predictions and labels

    Args:
        y_predictions: Predictions from a model (iterable)
        y_labels: Labels from a model (iterable)
        output: True to print the confusion matrix

    Returns:
        The confusion matrix (2D numpy array) associated with predictions and labels
        Format of the matrix is [ [TP FP], [TN, FN] ].
    """

    y_predictions = np.array(y_predictions)
    y_labels = np.array(y_labels)

    true_pos  = np.logical_and(y_predictions, y_labels).sum()
    false_pos = np.logical_and(y_predictions, 1-y_labels).sum()
    true_neg  = np.logical_and(1-y_predictions, 1-y_labels).sum()
    false_neg = np.logical_and(1-y_predictions, y_labels).sum()

    
    if output:
        print( '           %10s' % 'True' + '%10s' % 'False')
        print(f'Positives: %10s' % f'{true_pos}' + '%10s' % f'{false_pos}')
        print(f'Negatives: %10s' % f'{true_neg}' + '%10s' % f'{false_neg}')
        print('')

    return np.array([[true_pos, false_pos],
                     [true_neg,false_neg]])


def binary_measures(confusion_matrix, output=False):
    """ Calculates statistical measures from a confusion matrix

    Args:
        confusion_matrix: A 2D array of the form [ [TP FP], [TN, FN] ]
        output: True to print statistical measures

    Returns:
        A tuple consisting of the F1 score, precision and recall associated with the confusion matrix
    """

    true_pos, false_pos = confusion_matrix[0][0], confusion_matrix[0][1]
    true_neg, false_neg = confusion_matrix[1][0], confusion_matrix[1][1]

    if true_pos == 0 and false_pos == 0:
        precision = 0
    else:
        precision = round(true_pos / (true_pos + false_pos),2)

    if true_pos == 0 and false_neg == 0:
        recall = 0
    else:
        recall = round(true_pos / (true_pos + false_neg),2)

    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    if output:
        print(f'F1 score:  {round(f1,2)}')
        print(f'Precision: {round(precision,2)}')
        print(f'Recall:    {round(recall,2)}')

    return f1, precision, recall


def get_analyst_confusion_matrix_for_quarter(quarter_offset=1,output=False):
    """Returns confusion matrix associated with historical predictions

    Args:
        quarter_offset: The offset relative to today (must be historical, therefore >= 1)
        
    Returns:
        1 for overperform, 0 for underperform)
    """

    if output:
        print("Collecting analyst ratings...")
    get_stock_label = get_stock_label_func(TIMEDELTA_QUARTER,timedelta())
    prediction_dict = get_analyst_rating_for_quarter(quarter_offset)

    predictions = list()
    labels = list()

    if output:
        print(f"Calculating actual performance for {len(prediction_dict)} tickers: ")
    for ticker in prediction_dict.keys():
        if output:
            print(ticker,end=' ')
        try:
            labels.append(get_stock_label(ticker))
            predictions.append(prediction_dict[ticker])
        except:
            continue;

    return confusion_matrix(predictions,labels,output)

    
def get_analyst_rating_for_quarter(quarter_offset=0):
    """Returns analyst predictions for last quarter

    Args:
        quarter_offset: The offset relative to today (0 corresponds to latest analyst rating)
        
    Returns:
        1 for overperform, 0 for underperform)
    """
    def scrape_historical_ratings_for_ticker(symbol):
    
        try: 
            response = requests.get(f'https://www.marketbeat.com/stocks/NASDAQ/{symbol}/price-target/')
            soup = bs.BeautifulSoup(response.text, 'html.parser') 
            table = soup.find('table', {'class': 'scroll-table'})
            ratings_df = pd.read_html(str(table))[0]
            
            if ratings_df.shape != (5,5):
                raise LookupError
        except:
            raise LookupError
            
        return ratings_df
    
    sp500_tickers = get_sp500_tickers();

    if quarter_offset == 0:
        column = 'Today'
    elif quarter_offset == 1:
        column = '90 Days Ago'
    elif quarter_offset == 2:
        column = '180 Days Ago'
    else:
        raise LookupError("Data unavailable for the requested quarter")
    
    predictions = dict()

    for ticker in sp500_tickers:
        
        try:
            analyst_prediction_df = scrape_historical_ratings_for_ticker(ticker)

            # Scoring system we are using (consistent with Yahoo Finance) is 1-5 with:
            # 1 = strong outperform (strong buy) and 5 = strong underperform (strong sell)
            # MarketBeat's scoring is shifted by one and inverted (e.g., 0-4 with 0 = strong sell)
            # For this reason, we will subtract the score from 5 to convert to 1-5 scale
            analyst_score = 5.0 - float(analyst_prediction_df[column][1])

            predictions[ticker] = 1 if analyst_score < 3 else 0
        except:
            continue
            
    return predictions