import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, train_test_split
import xgboost as xgb
from sklearn.metrics import make_scorer

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

xgb_reg = xgb.XGBRegressor(objective='reg:squarederror',
                        colsample_bytree= 0.95,
                        eta=0.1,
                        gamma=0.2,
                        max_depth=7,
                        min_child_weight=2,
                        n_estimators=500,
                        reg_alpha=0.5,
                        reg_lambda=10,
                        subsample=0.97)

X_train, Y_train = preprocess(X_train, Y_train)
X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.2)

xgb_reg.fit(X_train, Y_train)

# Use best_model for further predictions
output_train = xgb_reg.predict(X_train)
output_test = xgb_reg.predict(X_test)

print('Spearman correlation for the train set: {:.1f}%'.format(100 * metric(output_train, Y_train)))
print('Spearman correlation for the test set: {:.1f}%'.format(100 * metric(output_test, Y_test)))
exit()
X_test = pd.read_csv('data/X_test.csv')
Y_test_submission_xgb = X_test[['ID']].copy()
Y_test_submission_xgb['TARGET'] = xgb_reg.predict(X_test)
Y_test_submission_xgb.to_csv('xgboost_v1.csv', index=False)


