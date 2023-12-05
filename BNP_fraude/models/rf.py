# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import gensim.downloader as api
import logging
import xgboost as xgb
from hyperopt import STATUS_OK, STATUS_FAIL, Trials, fmin, hp, tpe

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Word2V = api.load("word2vec-google-news-300")

def custom_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    return 'custom_avg_precision', average_precision_score(y_true, y_pred)

def get_w2v_vector(sentence, model):
    vec = np.zeros(300)
    try:
        for word in sentence.split():
            if word in model.key_to_index:
                vec += model[word]
    except:
        return vec
    return vec / len(sentence.split())

def preprocess(X_path,
            Y_path,
            cash_price_mean=None,
            cash_price_std=None,
            nbr_of_prod_purchas_mean=None,
            nbr_of_prod_purchas_std=None):
    
    X_path = X_path
    Y_path = Y_path

    n = 25
    X = pd.read_csv(X_path)
    y = pd.read_csv(Y_path)
    y = y.drop(['index', 'ID'], axis=1)

    start = 5
    columns = ['item' + str(i) for i in range(start, n)] + \
                ['make' + str(i) for i in range(start, n)] + \
                ['model' + str(i) for i in range(start, n)] + \
                ['goods_code' + str(i) for i in range(start, n)] + \
                ['Nbr_of_prod_purchas' + str(i) for i in range(start, n)] + \
                ['cash_price' + str(i) for i in range(start, n)]

    X = X.drop(columns, axis=1)

    columns = ['item' + str(i) for i in range(1, n)] + \
                ['make' + str(i) for i in range(1, n)] + \
                ['model' + str(i) for i in range(1, n)] + \
                ['goods_code' + str(i) for i in range(1, n)]
    
    n = start
    df = X
    categorical_columns = ['item', 'make', 'model']
    categorical_columns = [col+str(i) for col in categorical_columns for i in range(1, n)]

    for col in df.columns:
        if col in categorical_columns:  # Define your list of categorical columns
            # Apply Word2Vec transformation and split into separate columns
            w2v_df = df[col].apply(lambda x: pd.Series(get_w2v_vector(x, Word2V), dtype=np.float32))

            # Rename new columns
            w2v_df.columns = [f'{col}_w2v_{i}' for i in range(300)]

            # Concatenate with original DataFrame
            df = pd.concat([df, w2v_df], axis=1)

            # Optionally, drop the original categorical column
            df = df.drop(col, axis=1)
        elif col in columns:
            df = df.drop(col, axis=1)

    # Assuming df is your DataFrame
    # Compute mean and standard deviation for each type of numerical column
    cash_price_cols = [f'cash_price{i}' for i in range(1, n)]
    nbr_of_prod_purchas_cols = [f'Nbr_of_prod_purchas{i}' for i in range(1, n)]

    # Replace NaN values with 0
    for col in cash_price_cols + nbr_of_prod_purchas_cols:
        df[col].fillna(0, inplace=True)

    if cash_price_mean is None:
        cash_price_mean = df[cash_price_cols].values.flatten().mean()
        cash_price_std = df[cash_price_cols].values.flatten().std()

        nbr_of_prod_purchas_mean = df[nbr_of_prod_purchas_cols].values.flatten().mean()
        nbr_of_prod_purchas_std = df[nbr_of_prod_purchas_cols].values.flatten().std()

    # Normalize the columns
    for col in cash_price_cols:
        df[col] = (df[col] - cash_price_mean) / cash_price_std

    for col in nbr_of_prod_purchas_cols:
        df[col] = (df[col] - nbr_of_prod_purchas_mean) / nbr_of_prod_purchas_std

    return df, y, cash_price_mean, cash_price_std, nbr_of_prod_purchas_mean, nbr_of_prod_purchas_std

X_train_df, y_train_df, cash_price_mean, cash_price_std, nbr_of_prod_purchas_mean, nbr_of_prod_purchas_std = preprocess('data/X_train.csv', 'data/y_train.csv')

X_train, X_val, y_train, y_val = train_test_split(X_train_df, y_train_df, test_size=0.8, random_state=42)
# Define models and their respective hyperparameters

xgb_classifier = xgb.XGBClassifier(
    n_estimators=50,          # Number of gradient boosted trees. Equivalent to number of boosting rounds
    max_depth=5,               # Maximum tree depth for base learners
    min_child_weight=1,        # Minimum sum of instance weight (hessian) needed in a child
    eta=0,                   # Step size shrinkage used in update to prevents overfitting
    gamma=0.1,                   # Minimum loss reduction required to make a further partition on a leaf node of the tree
    subsample=0.8,             # Subsample ratio of the training instances
    colsample_bytree=0.8,      # Subsample ratio of columns when constructing each tree
    objective='binary:logistic', # Learning task parameter and the corresponding learning objective
    reg_alpha=0,             # L1 regularization term on weights
    reg_lambda=0,              # L2 regularization term on weights
    scale_pos_weight=20,        # Balancing of positive and negative weights
    random_state=42            # Random number seed
)
classifier = xgb_classifier

space = {
    'max_depth': 5,
    'gamma': hp.uniform('gamma', 1, 9),
    'reg_alpha': hp.quniform('reg_alpha', 0, 10, 1),
    'reg_lambda': hp.uniform('reg_lambda', 0, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'min_child_weight': hp.quniform('min_child_weight', 0, 10, 1),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'eta': hp.uniform('eta', 0.01, 0.2),
    'scale_pos_weight': 23,
    'n_estimators': 500,  # Fixed value
    'seed': 42  # Fixed value
}

def objective(space):
    clf = xgb.XGBClassifier(
        n_estimators=int(space['n_estimators']),
        max_depth=int(space['max_depth']),
        min_child_weight=space['min_child_weight'],
        eta=space['eta'],
        gamma=space['gamma'],
        subsample=space['subsample'],
        colsample_bytree=space['colsample_bytree'],
        objective='binary:logistic',
        reg_alpha=space['reg_alpha'],
        reg_lambda=space['reg_lambda'],
        scale_pos_weight=space['scale_pos_weight'],
        seed=int(space['seed'])
    )

    evaluation = [(X_train, y_train), (X_val, y_val)]
    clf.fit(X_train, y_train,
            eval_set=evaluation,
            eval_metric='logloss', # or your custom_eval function
            early_stopping_rounds=10,
            verbose=False)

    # Predict on validation set
    pred = clf.predict_proba(X_val)[:,1]  # Assuming you're dealing with binary classification
    loss = -average_precision_score(y_val, pred)  # Negative sign because fmin tries to minimize the loss

    return {'loss': loss, 'status': STATUS_OK}

trials = Trials()

best_hyperparams = fmin(fn = objective,
                        space = space,
                        algo = tpe.suggest,
                        max_evals = 100,
                        trials = trials)

print("The best hyperparameters are : ","\n")
print(best_hyperparams)
