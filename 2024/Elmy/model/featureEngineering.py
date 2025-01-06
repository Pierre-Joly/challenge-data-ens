import pandas as pd
import numpy as np

def convertToDatetime(X):
    X.DELIVERY_START = pd.to_datetime(X.DELIVERY_START, utc=True)
    X.index = pd.to_datetime(X.DELIVERY_START)
    return X

def fillMissingDates(X):
    start_date = X.index.min()
    end_date = X.index.max()
    full_range = pd.date_range(start=start_date, end=end_date, freq='H')

    X = X.reindex(full_range)

    return X

def fromDateToColumns(X):
    date = X['DELIVERY_START']
    
    # Dictionary of time periods and their respective cycle lengths
    time_periods = {
        'hour': 24,
        'day': 31,
        'month': 12,
        'year': 365
    }
    
    # Extract the time units and create sine/cosine transformations
    for period, cycle_length in time_periods.items():
        X[period] = getattr(date.dt, period)
        X[f'sin_{period}'] = np.sin(2 * np.pi * X[period] / cycle_length)
        X[f'cos_{period}'] = np.cos(2 * np.pi * X[period] / cycle_length)
    
    X = X.drop(columns=['DELIVERY_START'] + list(time_periods.keys()))
    
    return X

def featureEngineering(x_train, x_test, y_train, y_test):
    # filling missing dates and concatenate train and test set
    X_train = convertToDatetime(x_train)
    X_test = convertToDatetime(x_test)

    X = pd.concat([X_train, X_test])
    X = fillMissingDates(X)
    X = fromDateToColumns(X)

    # filling missing values
    ...

    # adding new features
    ...

    # normalizing data
    ...

    # Only return train set
    return X.loc[X_train.index], y_train


    
