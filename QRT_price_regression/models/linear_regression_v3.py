import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def metric(output, y):
    return spearmanr(output, y).correlation

def preprocess(X, Y=None):
    X['COUNTRY'] = X['COUNTRY'].apply(lambda x: -1 if x == 'FR' else (1 if x == 'DE' else 0))
    X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
    if Y is not None:
        Y = Y['TARGET']
        return X, Y
    return X

# After downloading the X_train/X_test/Y_train .csv files in your working directory:
X_train = pd.read_csv('data/X_train.csv')
Y_train = pd.read_csv('data/Y_train.csv')

lr = LinearRegression()

X_train, Y_train = preprocess(X_train, Y_train)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)

lr.fit(X_train, Y_train)

output_train = lr.predict(X_train)
output_test = lr.predict(X_test)

print('Spearman correlation for the train set: {:.1f}%'.format(100 * metric(output_train, Y_train)))
print('Spearman correlation for the test set: {:.1f}%'.format(100 * metric(output_test, Y_test)))

X_test = pd.read_csv('data/X_test.csv')
X_test = preprocess(X_test)
Y_test_submission = X_test[['ID']].copy()
Y_test_submission['TARGET'] = lr.predict(X_test)
Y_test_submission.to_csv('linear_v2.csv', index=False)

