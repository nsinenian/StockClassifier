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
import torch
import torch.utils.data
import numpy as np

from datetime import timedelta
from scipy.special import expit

from api.stocks import Stock   # Information about a particular stock
from api.stocks import TIMEDELTA_QUARTER
from api.stocks import TIMEDELTA_MONTH
from api.stocks import TIMEDELTA_YEAR

from model.dataset          import StockDataset
from model.predictor_helper import get_stock_cat_features_func
from model.predictor_helper import get_stock_daily_features_func
from model.predictor_helper import get_stock_quarterly_features_func

class StockClassifierPredictor():

    def __init__(self,model,device):
        """ 
        Args:
            model: The PyTorch model that we wish to use to make predictions.
            device: Where the model and data should be loaded (gpu or cpu).
        """
        self.model = model
        self.device = device

        self.get_stock_cat_features = get_stock_cat_features_func(self.model.data_interval,None)
        self.get_stock_daily_features = get_stock_daily_features_func(self.model.data_interval,None)
        self.get_stock_quarterly_features = get_stock_quarterly_features_func(self.model.data_interval,None)
        
    def predict(self,stock_ticker):
        """ 
        Args:
            stock_ticker: The stock ticker symbol for prediction

        Returns:
            Either a 1 or 0 indicating whether stock will outperform the market
        """
        
        self.model.to(self.device)
        self.model.eval()

        dataset = StockDataset.from_data([stock_ticker],
                                         p_interval=self.model.prediction_interval,
                                         d_interval=4*self.model.data_interval,
                                         offsets=[timedelta()],
                                         c_features_func_gen=get_stock_cat_features_func,
                                         d_features_func_gen=get_stock_daily_features_func,
                                         q_features_func_gen=get_stock_quarterly_features_func,
                                         label_func_gen=None,
                                         output=False)
        if len(dataset) == 0:
            raise LookupError

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=len(dataset),
                                                 shuffle=False,
                                                 collate_fn=StockDataset.collate_data)
        for x, _ in dataloader:
            x = [tensor.to(self.device) for tensor in x]
            y_pred = self.model(x).detach().numpy()

        return int(round(expit(y_pred[0])))