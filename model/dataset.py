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
import torch.utils.data.dataset as dataset
import torch.utils.data.dataloader as dataLoader

from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import os

from abc import ABC

class StockDataset(dataset.Dataset,ABC):
    """Stock dataset."""
    
    def __init__(self,p_interval,d_interval,offsets,features,labels=None):

        try:
            self.c_features_embedding_dims = features[0]
            self.c_features = features[1]
            self.d_features = features[2]
            self.q_features = features[3]
            self.labels = labels
            self.p_interval = p_interval
            self.d_interval = d_interval
            self.offsets = offsets

        except:
            raise ValueError

    @classmethod
    def concat(cls,datasets):
        """ Concatenates datasets to make a new dataset (for use with K-folding)
        Args:
            datasets: An iterable of StockDatasets

        Retruns:
            The concatenated dataset
        """
        baseline_ds = datasets[0]
        for ds in datasets:
            if ds.get_prediction_interval() != baseline_ds.get_prediction_interval():
                raise ValueError("Mismatch in prediction interval")

            if ds.get_data_interval() != baseline_ds.get_data_interval():
                raise ValueError("Mismatch in data interval")

            if not np.array_equal(ds.get_offsets(),baseline_ds.get_offsets()):
                raise ValueError("Mismatch in data offsets")

            if ds.get_categorical_feature_count() != baseline_ds.get_categorical_feature_count():
                raise ValueError("Mismatch in categorical features")

            if not np.array_equal(ds.get_categorical_feature_embedding_dims(),baseline_ds.get_categorical_feature_embedding_dims()):
                raise ValueError("Mismatch in categorical feature embedding dimensions")

            if ds.get_daily_feature_count() != baseline_ds.get_daily_feature_count():
                raise ValueError("Mismatch in daily features")

            if ds.get_quarterly_feature_count() != baseline_ds.get_quarterly_feature_count():
                raise ValueError("Mismatch in quarterly features")

        c_features_embedding_dims = ds.get_categorical_feature_embedding_dims()
        c_features = np.concatenate([ ds.c_features for ds in datasets])
        d_features = np.concatenate([ ds.d_features for ds in datasets])
        q_features = np.concatenate([ ds.q_features for ds in datasets])
        labels = np.concatenate([ ds.labels for ds in datasets])

        return cls(baseline_ds.get_prediction_interval(),
                   baseline_ds.get_data_interval(),
                   baseline_ds.get_offsets(),
                   [c_features_embedding_dims,c_features, d_features, q_features],
                   labels)
        
    @classmethod
    def from_data(cls,
                  stocks,
                  p_interval,
                  d_interval,
                  offsets,
                  c_features_func_gen,
                  d_features_func_gen,
                  q_features_func_gen,
                  label_func_gen=None,
                  output=False):
        """ Creates a dataset using the specified data generator functions
        Args:
            stocks: The data interval as a datetime.timedelta (e.g., 6*TIMEDELTA_QUARTER for 6 quarters of data)
            p_interval: The prediction interval, as a datetime.timedelta object
            d_interval: The data interval, as a datetime.timedelta object
            offsets: An iterable of offsets to use for prediction and data relative to today, as a datetime.timedelta object
            c_features_func_gen: A function that accepts d_interval and offset arguments and returns a 
                                 function that provides categorical features data for a specified stock
            d_features_func_gen: A function that accepts d_interval and offset arguments and returns a 
                                 function that provides daily features data for a specified stock
            q_features_func_gen: A function that accepts d_interval and offset arguments and returns a 
                                 function that provides quarterly features data for a specified stock
            label_func_gen: A function that accepts p_interval and offset arguments and returns a function
                            that provides labels for a specified stock
            
        Returns:
            A Dataset object that includes feature and label data for the specified stocks over the specified interval
        """
        success_stocks = list()
        problem_stocks = list()
    
        c_features = list()
        d_features = list()
        q_features = list()
        labels = list()
        
        for offset in offsets:  
            # For each specified data offset, prepare functions that will
            # be used to generate data for the specified intervals      
            c_features_embedding_dims, c_features_func = c_features_func_gen(d_interval,offset+p_interval)
            d_features_func = d_features_func_gen(d_interval,offset+p_interval)
            q_features_func = q_features_func_gen(d_interval,offset+p_interval)

            if label_func_gen is not None:
                label_func = label_func_gen(p_interval,offset)
            else:
                label_func = None
            
            for stock in stocks:
                try:
                    # Attempt to get all data first, if not available exception will be thrown
                    c = c_features_func(stock)
                    d = d_features_func(stock)
                    q = q_features_func(stock)

                    if label_func:
                        l = label_func(stock)

                    # Time-series features will need to be transposed for our LSTM input
                    c_features.append(c.transpose().astype(dtype='float32',copy=False))
                    d_features.append(d.transpose().astype(dtype='float32',copy=False))
                    q_features.append(q.transpose().astype(dtype='float32',copy=False))

                    if label_func:
                        labels.append(l)
                    
                    success_stocks.append(stock)
                except:
                    problem_stocks.append(stock)
                    continue
                if output:    
                    print(".", end = '')
        
        if output:
            print('')
            print(f'The following stocks were successfully processed: {", ".join(success_stocks)}')
            print('')
            print(f'The following tickers did not have complete data and were not processed: {", ".join(problem_stocks)}.')

        features = [c_features_embedding_dims,np.stack(c_features,axis=0),np.array(d_features),np.stack(q_features,axis=0)]
        labels = np.stack(labels,axis=0) if label_func is not None else None
            
        return cls(p_interval,d_interval,offsets,features,labels)

    @classmethod
    def from_file(cls,path):
        
        data = np.load(path,allow_pickle=True)['arr_0']
        meta, features, labels = data[0], data[1], data[2]

        return cls(meta[0],meta[1],meta[2],features,labels)
    
    def to_file(self, path, output=False):
        directory = os.path.dirname(path)

        if not os.path.exists(directory):
            os.makedirs(directory)

        meta = [self.p_interval,self.d_interval,self.offsets]
            
        features = [self.c_features_embedding_dims,
                    self.c_features,
                    self.d_features,
                    self.q_features]

        np.savez(path,[meta,features,self.labels])
        
        if output:
            print(f'Successfully wrote data to {path}')
                   
    def __len__(self):
        
        return len(self.c_features)

    def __getitem__(self, index):
        
        features = [self.c_features[index],
                    self.d_features[index],
                    self.q_features[index]]

        if self.labels is None:
            return (features, None)

        return (features, self.labels[index])

    @staticmethod
    def collate_data(batch):
        # Features is indexed as features[stock][frequency][sequence][feature]
        (features, labels) = zip(*batch)

        batch_size = len(features)
        
        # Concatenate (stack) categorical and quarterly features, as those will
        # have the same sequence length across all samples
        categorical_features = torch.stack([torch.from_numpy( features[i][0] ) for i in range(batch_size)],axis=0)
        quarterly_features = torch.stack([torch.from_numpy( features[i][2] ) for i in range(batch_size)] ,axis=0)
        
        # Daily features: the sequence lengths may vary depending on the
        # absolute time interval of the data (over some intervals there
        # are market holidays and hence less data). We will need to pad
        # and pack data using PyTorch.
        
        # Get length of daily features (e.g., sequence length)
        daily_features_lengths = [ len(features[i][1]) for i in range(batch_size)]
        
        # Generate array of torch tensors for padding; tensors will have incompatible sizes
        daily_features = [torch.from_numpy( features[i][1] ) for i in range(batch_size)] 
        
        # Pad tensors to the longest size
        daily_features_padded = pad_sequence(daily_features,batch_first=True,padding_value = -10)
        
        # Pack the batch of daily features
        daily_features_packed = pack_padded_sequence(daily_features_padded,daily_features_lengths,batch_first=True,enforce_sorted=False)

        features = [categorical_features,daily_features_packed,quarterly_features]
        labels = torch.from_numpy(np.array(labels)) if labels[0] is not None else None

        return features, labels

    def get_prediction_interval(self):
        return self.p_interval

    def get_data_interval(self):
        return self.d_interval

    def get_offsets(self):
        return self.offsets

    def get_categorical_feature_count(self):
        return len(self.c_features[0])
        
    def get_categorical_feature_embedding_dims(self):
        return self.c_features_embedding_dims

    def get_daily_feature_count(self):
        return self.d_features[0].shape[1]

    def get_quarterly_feature_count(self):
        return self.q_features[0].shape[1]