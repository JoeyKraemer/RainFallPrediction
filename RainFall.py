import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

# Data preparation
df = pd.read_csv('weatherAUS.csv')
df.head()  # First 5 rows of a data set
df.info()  # Information about data columns

print(df.describe().T)  # Descriptive statistical measures of the dataset

print(df.isnull().sum())  # Amount of empty rows in each column

print(df.columns)  # Index of columns

# Replace empty flo
for column in df.columns:

    if df[column].dtype == 'float64':
        if df[column].isnull().sum() > 0:
            val = df[column].mean(axis=0)
            df[column] = df[column].fillna(val)

print(df.isnull().sum().sum())  # Check if all Null values are replaced
