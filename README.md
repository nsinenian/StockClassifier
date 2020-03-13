# StockClassifier: Deep Learning Stock Classification

[**Quickstart**](#quickstart)
| [**Benchmarks**](#benchmarks)
| [**Organization**](#organization)
| [**Dependencies**](#dependencies)

A deep-learning binary classifier that can be used to predict whether a stock is going to outperform or underperform relative to a reference value (e.g., the S&P 500). The model consists of LSTM and linear layers that are trained to recognize patterns in time-series and categorical data pertaining to a large number of stocks, including daily time-series (price and volume), quarterly metrics (valuation, revenue and earnings data) along with discrete and continous categorical data (e.g., industry sector and market capitalization).


## Quickstart
A pre-trained model is provided that can be used to test out the classifier. After cloning the repository, the following Python code can be used to create a model instance from trained model data files:

```python
from model.classifier import StockClassifier
from model.predictor import StockClassifierPredictor

import torch
model = StockClassifier.from_file('data')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check model for training date and metadata
print(model)

# Create a predictor instance using our trained model.
predictor = StockClassifierPredictor(model,device)

# The stock_ticker argument is a string containing the ticker symbol (e.g., "AAPL")
y = predictor.predict(stock_ticker)
```

The result will be either a 0 or a 1 indicating whether the stock is predicted to overperform (1) or underperform(0) relative to the S&P 500 three months from now. An exception may be thrown if sufficient date is not available for the selected stock ticker (an internet connection is required to download data). The pretrained model was trained using only S&P 500 stock data. Please note that certain external packages are required as described below under [**dependencies**](#dependencies).

## Benchmarks

The methodology introduced in this project has been benchmarked against random stock selection of outperforms within the S&P 500. It has also been compared against historical analyst predictions for Q3/Q4 of 2019. The performance, as primarily measured by precision exceeds that of analysts predictions and of random selection. Bear in mind that no future earnings estimates or analysis is included in this model, and only historical data is utilized. Some metrics are summarized for the data set included in this repository. The datasets included in this repository includes S&P 500 stocks and spans the time interval Q2'18 through Q1'20. The performance of the pretrained model using this dataset is summarized in the table below. Note that the dataset is slightly biased towards negative classes (0s) and therefore a fair coin toss yields a precision of less than 0.5.

<table>
  <tr>
    <th></td>
    <th colspan="2">Fair Coin Toss</td>
    <th colspan="2">Analyst Recommendation</td>
    <th colspan="2">This Model</td>
  </tr>
  <tr>
    <td></td>
    <td>True</td>
    <td>False</td>
    <td>True</td>
    <td>False</td>
    <td>True</td>
    <td>False</td>
  </tr>
   <tr>
    <td>Positive</td>
    <td>1801</td>
    <td>2084</td>
    <td>94</td>
    <td>135</td>
    <td>1469</td>
    <td>917</td>
  </tr>
  <tr>
    <td>Negative</td>
    <td>910</td>
    <td>1031</td>
    <td>21</td>
    <td>6</td>
    <td>2198</td>
    <td>1242</td>
  </tr>
   <tr>
    <td>F1 Score</td>
    <td colspan="2">0.48</td>
    <td colspan="2">0.57</td>
    <td colspan="2">0.58</td>
  </tr>
  <tr>
    <td>Precision</td>
    <td colspan="2">0.47</td>
    <td colspan="2">0.41</td>
    <td colspan="2">0.62</td>
  </tr>
   <tr>
    <td>Recall</td>
    <td colspan="2">0.50</td>
    <td colspan="2">0.94</td>
    <td colspan="2">0.54</td>
  </tr>
</table>

## Organization

The framework consists of a Python API for accessing historical stock data from https://financialmodelingprep.com/. The notebooks use the API to inspect historical data for S&P 500 stocks, to generate features for model training and to generate predictions from a trained model. Model training and prediction generation is facilitated by a set of Python classes designed to work with PyTorch.

The project is setup to use historical S&P 500 data as the reference value for stock performance and historical stock gains are compared to this baseline over a period of one quarter (i.e., labeled 1 if they exceed the S&P 500 and 0 if they do not). Data for the four quarters preceding this one-quarter interval are then used as training data. The dataset as described (both four quarters of data and one quarter used to generate labels) is offset repeatedly to generate additional datasets. In total, twelve offsets are used over a period of one year to capture seasonality. The offsets are encoded in the model as categorical features, so that predictions can take advantage of any seasonal patterns on a monthly basis.

The Jupyter notebooks can be readily altered to include features of interest, to train a model for different time horizons or to capture a larger collection of stocks (e.g., Russell 2000) with little to no changes to the remaining code base, as described further below.

### Notebooks

The workflow for model training starts with a data visualization notebook (`data_visualization.ipynb`). This notebook can be used to inspect categorical features of stocks and time-series data, including daily and quarterly time-series of interest. After data inspection, features can be generated using a second notebook (`data_processing.ipynb`). The notebook is setup such that with 4 functions that generate features: one for each of the data frequencies involved (categorical, daily and quarterly) plus a function to generate labels. As such, only these four functions need to be modified to include new features of interest. 

Model training and prediction generation is implemented in two notebooks, respectively (`training.ipynb` and `prediction.ipynb`). These notebooks rely on the aforementioned Python classes implemented to work with PyTorch as discussed further below.
 

### Python Classes
The following classes have been implemented for use in training a model and generating predictions:

  - `StockDataSet`: A dataset derived from a PyTorch dataset to hold training and testing data
  - `StockClassifier`: A deep learning model definition class
  - `StockClassifierEstimator`: A class used to train a `StockClassifier` model (object instance)
  - `StockClassifierPredictor`: A class that can be used to load a trained model and obtain predictions

Unless significant changes to the model are desired, these classes can be used with a wide range of categorical features and time-series features with daily, quarterly frequencies. These classes are used in the data processing, training, and prediction notebooks.

## Dependencies
Use of the notebooks and Python classes require the following external dependences: 

   - PyTorch
   - NumPy
   - Pandas
   - Beautiful Soup
   - Matplotlib
