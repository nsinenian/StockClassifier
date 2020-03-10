# StockClassifier
Deep Learning Network for Quarterly Stock Performance Classification

This project consists of notebook and Python classes that can be used to train a PyTorch deep learning model to predict stock performance for the next quarter or few months. The framework consists of a Python API for accessing historical stock data from https://financialmodelingprep.com/. The notebooks use the API to inspect historical data for S&P 500 stocks, to generate features for model training and to generate predictions from a trained model. Model training and prediction generation is facilitated by a set of Python classes designed to work with PyTorch.

## Notebooks

The workflow for model training starts with a data visualization notebook (`data_visualization.ipynb`). This notebook can be used to inspect categorical features of stocks and time-series data, including daily and quarterly time-series of interest. After data inspection, features can be generated using a second notebook (`data_processing.ipynb`). The notebook is setup such that with 4 functions that generate features: one for each of the data frequencies involved (categorical, daily and quarterly) plus a function to generate labels. As such, only these four functions need to be modified to include new features of interest. 

Model training and prediction generation is implemented in two notebooks, respectively (`training.ipynb` and `prediction.ipynb`). These notebooks rely on the aforementioned Python classes implemented to work with PyTorch as discussed further below.
 

## Python Classes
The following classes have been implemented for use in training a model and generating predictions:

  - `StockDataSet`: A dataset derived from a PyTorch dataset to facilitate
  - `StockClassifier`: A deep learning model
  - `StockClassifierEstimator`: A class that can be used to train a `StockClassifier` model
  - `StockClassifierPredictor`: A class that can be used to load a trained model and obtain predictions

Unless significant changes to the model are desired, these classes can be used with a wide range of categorical features and time-series features with daily, quarterly frequencies. These classes are used in the data processing, training, and prediction notebooks.

## Dependencies
Use of the notebooks and Python classes requires the typical external dependences: PyTorch, NumPy, Pandas, Beautiful Soup, and Matplotlib.
