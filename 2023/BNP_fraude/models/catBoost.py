# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import gensim.downloader as api
import logging
import xgboost as xgb
from catboost import CatBoostClassifier

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess(X_path, Y_path):
    
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
        if col in categorical_columns:
            df[col] = df[col].fillna('').astype('category')
        elif col in columns:
            df = df.drop(col, axis=1)

    cash_price_cols = [f'cash_price{i}' for i in range(1, n)]
    nbr_of_prod_purchas_cols = [f'Nbr_of_prod_purchas{i}' for i in range(1, n)]

    for col in cash_price_cols + nbr_of_prod_purchas_cols:
        df[col].fillna(0, inplace=True)

    return df, y

X_train_df, y_train_df = preprocess('data/X_train.csv', 'data/y_train.csv')

categorical_columns = ['item', 'make', 'model']
categorical_columns = [col+str(i) for col in categorical_columns for i in range(1, 5)]

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_df, y_train_df, test_size=0.2, random_state=42)

# Define models and their respective hyperparameters
classifier = CatBoostClassifier(
    cat_features=categorical_columns, # Define categorical features
    loss_function='Logloss',        # Suitable for binary classification
    eval_metric='AUC',
    class_weights=[1, 10],          # Handling imbalanced data
)

# Create the pipeline
classifier
# Fit the pipeline
classifier.fit(X_train, y_train, eval_set=(X_val, y_val))

# Evaluate the pipeline using average_precision_score
print("Train score")
y_pred = classifier.predict_proba(X_train)
average_precision = average_precision_score(y_train, y_pred[:, 1]) * 100
logger.info(f"Average precision score: {average_precision}")
print("Test score")
y_pred = classifier.predict_proba(X_val)
average_precision = average_precision_score(y_val, y_pred[:, 1]) * 100
logger.info(f"Average precision score: {average_precision}")

# Save the trained model
import joblib
model_filename = "trained_rf_classifier.pkl"
joblib.dump(classifier, model_filename)
logger.info(f"Trained model saved as {model_filename}")

##### Prediction #####
# Save prediction
X, Y = preprocess('data/X_test.csv', 'data/y_test.csv')

y_pred = classifier.predict_proba(X)[:, 1]  # Get the probability of the positive class
# Create a DataFrame for predictions
prediction_df = pd.DataFrame({
    'ID': X['ID'],  # Replace 'ID' with the actual ID column name in your validation set
    'fraud_flag': y_pred
})
prediction_df.reset_index(inplace=True)
prediction_df = prediction_df.rename(columns={'index': 'index'})
prediction_df.to_csv('data/y_pred.csv', index=False)
