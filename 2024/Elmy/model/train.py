import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

x_train = pd.read_csv('data/X_train.csv')
y_train = pd.read_csv('data/y_train.csv')
