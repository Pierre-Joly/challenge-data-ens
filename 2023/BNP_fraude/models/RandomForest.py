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

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Word2V = api.load("word2vec-google-news-300")

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

# Define models and their respective hyperparameters

xgb_classifier = xgb.XGBClassifier(
    n_estimators=500,          # Number of gradient boosted trees. Equivalent to number of boosting rounds
    max_depth=10,               # Maximum tree depth for base learners
    min_child_weight=0,        # Minimum sum of instance weight (hessian) needed in a child
    eta=0.01,                   # Step size shrinkage used in update to prevents overfitting
    gamma=0.1,                   # Minimum loss reduction required to make a further partition on a leaf node of the tree
    subsample=0.97,             # Subsample ratio of the training instances
    colsample_bytree=0.98,      # Subsample ratio of columns when constructing each tree
    objective='binary:logistic', # Learning task parameter and the corresponding learning objective
    reg_alpha=0,             # L1 regularization term on weights
    reg_lambda=0,              # L2 regularization term on weights
    scale_pos_weight=23.5,        # Balancing of positive and negative weights
    random_state=42            # Random number seed
)
classifier = xgb_classifier

# Create the pipeline
pipeline = make_pipeline(
    classifier
)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_df, y_train_df, test_size=0.2, random_state=42)
# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the pipeline using average_precision_score
print("Train score")
y_pred = pipeline.predict_proba(X_train)
average_precision = average_precision_score(y_train, y_pred[:, 1]) * 100
logger.info(f"Average precision score: {average_precision}")
print("Test score")
y_pred = pipeline.predict_proba(X_val)
average_precision = average_precision_score(y_val, y_pred[:, 1]) * 100
logger.info(f"Average precision score: {average_precision}")

# Save the trained model
import joblib
model_filename = "trained_rf_classifier.pkl"
joblib.dump(pipeline, model_filename)
logger.info(f"Trained model saved as {model_filename}")

##### Prediction #####
# Save prediction
X, _, _, _, _, _ = preprocess('data/X_test.csv', 'data/y_test.csv', cash_price_mean, cash_price_std, nbr_of_prod_purchas_mean, nbr_of_prod_purchas_std)

y_pred = pipeline.predict_proba(X)[:, 1]  # Get the probability of the positive class
# Create a DataFrame for predictions
prediction_df = pd.DataFrame({
    'ID': X['ID'],  # Replace 'ID' with the actual ID column name in your validation set
    'fraud_flag': y_pred
})
prediction_df.reset_index(inplace=True)
prediction_df = prediction_df.rename(columns={'index': 'index'})
prediction_df.to_csv('data/y_pred.csv', index=False)
