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

import argparse
import json
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import numpy as np

from datetime import datetime
from scipy.special import expit

class StockClassifierEstimator():

    def __init__(self,model,optimizer,loss_fn,device):
        """ 
            Args:
                model: The PyTorch model that we wish to train.
                optimizer: The optimizer to use during training.
                loss_fn: The loss function used for training. 
                device: Where the model and data should be loaded (gpu or cpu).
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
    
    def fit(self,train_loader,epochs,output=True):
        """ 
            Args:
                train_loader: The PyTorch DataLoader that should be used during training.
                epochs: The total number of epochs to train for.
                output: True to print results during training

            Returns:
                The total training loss
        """
        self.model.to(self.device)
        self.model.train() # Make sure that the model is in training mode.

        for epoch in range(1, epochs + 1):

            total_loss = 0

            for x, y in train_loader:
                        
                # The train_loader will feed us a batch of data
                # stacked for the batch_size (number of stocks)
                # and provided seperately as daily and quarterly
                # data. e.g., x[frequency_index][stock_index][2D index]
                for tensor in x:
                    if type(tensor) is 'torch.Tensor':
                        tensor.requires_grad_(True)
                
                x = [tensor.to(self.device) for tensor in x]
                
                y = y.type(torch.float32)
                y = y.to(self.device)
        
                self.optimizer.zero_grad()
            
                # get predictions from model
                y_pred = self.model(x)
                
                # perform backprop
                loss = self.loss_fn(y_pred, y)
                loss.backward()
                
                self.optimizer.step()
                
                total_loss += loss.data.item()

            if output:
                print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))

        self.model.eval()

        self.model.last_train_date = datetime.now() 
        self.model.train_count += 1

        return total_loss / len(train_loader)

    def validate(self,validate_loader):
        
        self.model.eval()

        y_pred, y_label = np.array([]), np.array([])

        for x, y in validate_loader:

            x = [tensor.to(self.device) for tensor in x]

            y_pred = np.append(y_pred, self.model(x).detach().numpy())
            y_label = np.append(y_label, y.detach().numpy())

        return np.round(expit(y_pred)), y_label