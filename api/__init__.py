import os

import requests
import json

import numpy as np
import pandas as pd

import datetime as datetime
from datetime import timedelta

TIMEDELTA_QUARTER = timedelta(days=90)
TIMEDELTA_MONTH = timedelta(days=30)
TIMEDELTA_YEAR  = timedelta(days=365)

__cache_dir__ = os.path.join('./cache/')
if not os.path.exists(__cache_dir__):
    os.makedirs(__cache_dir__)