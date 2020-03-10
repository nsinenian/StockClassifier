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

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataLoader

from torch.nn.utils.rnn import pad_packed_sequence

import numpy as np
import os

class StockClassifier(nn.Module):
    
    def __init__(self,
                 prediction_interval,
                 data_interval,
                 categorical_features,
                 categorical_features_embedding_dims,
                 daily_features,
                 quarterly_features,
                 embeddings_hidden_dim=10,
                 hidden_dim=100,
                 output_dim=1):
        """ Defines layers of a neural network.

        Args:
            prediction_interval: Interval over which predictions are to be made as datetime.timedelta (encoded in how labels are generated)
            data_interval: Amount of training data as datetime.timedelta (represented in sequence length of features)
            categorical_features: The number of categorical features in the dataset
            categorical_features_embedding_dims: Embedding dimensions for each of the categorical features
            daily_features: The number of features having daily frequency
            quarterly_features: The number of features having quartelry frequency
            embeddings_hidden_dim: The number of hidden embedding dimensions for embedded categorical features
            hidden_dim: The number of hidden dimensions in the post-LSTM linear network
            output_dim: The output dimension of the network
        """
        super(StockClassifier, self).__init__()

        # Feature metadata
        self.prediction_interval = prediction_interval
        self.data_interval = data_interval
        self.categorical_features =  categorical_features
        self.categorical_features_embedding_dims = categorical_features_embedding_dims
        self.daily_features = daily_features
        self.quarterly_features = quarterly_features

        # Hyperparameters (embedding dims)
        self.embeddings_hidden_dim = embeddings_hidden_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.last_train_date = None
        self.train_count = 0

        # For time-series frequency feature create a new LSTM of the appropriate feature size e.g., if 
        # daily data includes only adjusted price, then input_size should be set to 1 for the daily LSTM
        self.lstm_daily = nn.LSTM(input_size=daily_features,hidden_size=daily_features,batch_first=True)
        self.lstm_quarterly = nn.LSTM(input_size=quarterly_features,hidden_size=quarterly_features,batch_first=True)

        # Create a list of embeddings for each of the categorical features to be embedded
        self.categorical_embeddings = nn.ModuleList()

        self.categorical_embeddings_count = 0
        for embedding in self.categorical_features_embedding_dims:
            if embedding == 0:
                continue

            self.categorical_embeddings.append(nn.Embedding(embedding,self.embeddings_hidden_dim,padding_idx=0))
            self.categorical_embeddings_count+=1

        # Features to combine after embeddings and LSTMs
        self.linear_input_size = \
            self.categorical_embeddings_count * self.embeddings_hidden_dim + \
            self.categorical_features - self.categorical_embeddings_count + \
            self.daily_features + self.quarterly_features                                                 

        self.fc1 = nn.Linear(in_features=self.linear_input_size,out_features=self.hidden_dim)
        self.fc2 = nn.Linear(in_features=self.hidden_dim,out_features=int(self.hidden_dim/2))
        self.fc3 = nn.Linear(in_features=int(self.hidden_dim/2),out_features=int(hidden_dim/2))
        self.fc4 = nn.Linear(in_features=int(self.hidden_dim/2),out_features=self.output_dim)
        self.fcd = nn.Dropout(0.1)
    
    def forward(self, x):
        """ Feedforward behavior of the net.
        Args:
            x: Tensors holding data in the format x[frequency_index][stock_index][2D index feature/sequence indexing]

        Returns:
            A binary classification for the stock(s) (logits output)
        """
        x_daily, h_daily = self.lstm_daily(x[1])
        x_quarterly, h_quarterly = self.lstm_quarterly(x[2])

        x_categorical = x[0]

        x_embedding_out = torch.tensor([])

        # Iterate over categorical data to be embedded
        for i in range(self.categorical_embeddings_count):
            x_embedding_out = torch.cat([x_embedding_out,self.categorical_embeddings[i](x_categorical[:,i].long())],axis=1)

        x_daily, _ = pad_packed_sequence(x_daily,batch_first=True)
        x_daily = x_daily[:,-1,:]

        x_quarterly = x_quarterly[:,-1,:]

        lstm_out = torch.cat([x_embedding_out,x_categorical[:,self.categorical_embeddings_count:],x_daily,x_quarterly],axis=1)

        y = self.fc1(lstm_out)
        y = self.fc2(F.relu(y))
        y = self.fcd(F.relu(y))
        y = self.fc3(y)
        y = self.fc4(F.relu(y))

        return y[:,-1]

    @classmethod
    def from_file(cls,model_dir):
        """ Load the PyTorch model from the `model_dir` directory.

        Args:
            model_dir: The directory where model data is stored (model.pth and model_info.pth)

        Returns:
            A new model based on data from disk
        """

        # First, load the parameters used to create the model.
        model_info = {}
        model_info_path = os.path.join(model_dir, 'model_info.pth')
        with open(model_info_path, 'rb') as f:
            model_info = torch.load(f)

        # Determine the device and construct the model.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = cls(
            model_info['prediction_interval'],
            model_info['data_interval'],
            model_info['categorical_features'],
            model_info['categorical_features_embedding_dims'],
            model_info['daily_features'],
            model_info['quarterly_features'],
            model_info['embeddings_hidden_dim'],
            model_info['hidden_dim'],
            model_info['output_dim'])

        model.last_train_date = model_info['last_train_date']
        model.train_count = model_info['train_count']

        # Load the stored model parameters.
        model_path = os.path.join(model_dir, 'model.pth')
        with open(model_path, 'rb') as f:
            model.load_state_dict(torch.load(f))

        # set to eval mode, could use no_grad
        model.to(device).eval()

        return model

    def __str__(self):
        output = ''
        output += f'Last Train Date:     {self.last_train_date}\n'
        output += f'Train Count:         {self.train_count}\n'
        output += f'Prediction Interval: {self.prediction_interval}\n'
        output += f'Data Interval:       {self.data_interval}\n'
        output += f'Trained Features: \n'
        output += f'    Categorical: {self.categorical_features}\n'
        output += f'    Daily:       {self.daily_features}\n'
        output += f'    Quarterly:   {self.quarterly_features}\n'
        output += f'Dimensions: \n'
        output += f'    Embedded:    {self.embeddings_hidden_dim}\n'
        output += f'    Hidden:      {self.hidden_dim}\n'
        output += f'    Output:      {self.output_dim}'
        return output

    def to_file(self,model_dir):
        """ Save the PyTorch model to the `model_dir` directory.

        Args:
            model_dir: The directory where model will be stored (model.pth and model_info.pth)
        """
        model_info_path = os.path.join(model_dir, 'model_info.pth')
        with open(model_info_path, 'wb') as f:
            model_info = {
                'last_train_date' : self.last_train_date,
                'train_count' : self.train_count,
                'prediction_interval' : self.prediction_interval,
                'data_interval' : self.data_interval,
                'categorical_features': self.categorical_features,
                'categorical_features_embedding_dims' : self.categorical_features_embedding_dims,
                'daily_features':  self.daily_features,
                'quarterly_features': self.quarterly_features,
                'embeddings_hidden_dim' : self.embeddings_hidden_dim,
                'hidden_dim' : self.hidden_dim,
                'output_dim' : self.output_dim,
            }
            torch.save(model_info, f)

        # Save the model parameters
        model_path = os.path.join(model_dir, 'model.pth')
        with open(model_path, 'wb') as f:
            torch.save(self.cpu().state_dict(), f)

