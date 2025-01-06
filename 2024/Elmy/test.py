import pandas as pd

x = pd.read_csv('data/X_train.csv')

x.index = pd.to_datetime(x.DELIVERY_START, utc=True)
x.DELIVERY_START = pd.to_datetime(x.DELIVERY_START, utc=True)
print(x.DELIVERY_START)