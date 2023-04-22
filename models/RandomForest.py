# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.base import BaseEstimator, TransformerMixin
import gensim.downloader as api
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the Word2Vec transformer class
class W2V(BaseEstimator, TransformerMixin):
    def __init__(self, num_words=None, **kwargs):
        self.num_words = num_words
        self.tokenizer = Tokenizer(num_words=num_words, **kwargs)

    def fit(self, X, y=None):
        self.Word2 = api.load("word2vec-google-news-300")
        return self

    def transform(self, X, y=None):
        x = np.array(X.values)
        for i in range(len(x)):
            for j in range(len(x[i])):
                tokens = x[i][j].split()
                embeddings = [self.Word2[token] for token in tokens if token in self.Word2.key_to_index]
                if len(embeddings) > 0:
                    mean = np.mean(embeddings)
                else:
                    mean = 0
                x[i][j] = mean
        return x


    def get_params(self, deep=True):
        return {"num_words": self.num_words}

# Load the dataset
X_train_file = 'data/X_train.csv'
y_train_file = 'data/Y_train.csv'

with open(X_train_file, 'r') as f:
    mixed_columns = ['item' + str(i) for i in range(1, 25)] + ['make' + str(i) for i in range(1, 25)] + ['model' + str(i) for i in range(1, 25)] + ['goods_code' + str(i) for i in range(1, 25)]
    mixed_columns_dtype = {col: str for col in mixed_columns}
    X_train_df = pd.read_csv(X_train_file, dtype=mixed_columns_dtype)

with open(y_train_file, 'r') as f:
    y_train_df = pd.read_csv(f)

cols_base = ['goods_code']
columns_to_drop = ['ID'] + [col + str(i) for col in cols_base for i in range(1, 25)]

X_train_df = X_train_df.drop(columns_to_drop, axis=1)
y_train_df = y_train_df['fraud_flag']

# Identify the columns to apply RNN tokenization
rnn_columns = ['make', 'item', 'model']  # Add more columns as needed
rnn_columns = [col + str(i) for col in rnn_columns for i in range(1, 25)]

# Identify the categorical and numerical columns
categorical_columns = rnn_columns
numerical_columns = [col for col in X_train_df.columns if col not in categorical_columns]

# Clean data
for col in categorical_columns:
    X_train_df[col] = X_train_df[col].fillna('')
for col in numerical_columns:
    X_train_df[col] = X_train_df[col].fillna(0)

# Define transformers
cat_pipeline = make_pipeline(W2V())
num_pipeline = make_pipeline(StandardScaler())

# Create the preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('cat_pipeline', cat_pipeline, categorical_columns),
    ('num_pipeline', num_pipeline, numerical_columns)
])

# Define models and their respective hyperparameters
rf_classifier = RandomForestClassifier(n_estimators=500, random_state=42)

# Create the pipeline
pipeline = make_pipeline(
    preprocessor,
    rf_classifier
)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_df, y_train_df, test_size=0.3, random_state=42)

# Fit the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the pipeline using average_precision_score
y_pred = pipeline.predict_proba(X_val)
average_precision = average_precision_score(y_val, y_pred[:, 1]) * 100
logger.info(f"Average precision score: {average_precision}")

# Save the trained model
import joblib
model_filename = "trained_rf_classifier.pkl"
joblib.dump(pipeline, model_filename)
logger.info(f"Trained model saved as {model_filename}")

# Load and use the trained model for predictions
loaded_pipeline = joblib.load(model_filename)
sample_input = X_val.iloc[:5, :]  # Take a sample input for prediction
sample_output = loaded_pipeline.predict_proba(sample_input)
logger.info(f"Sample input predictions: {sample_output}")
