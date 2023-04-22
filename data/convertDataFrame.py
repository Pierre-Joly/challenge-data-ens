"""
File: convertDataFrame.py

This file performs data preprocessing on four datasets stored in CSV format. The datasets contain information about transactions and their associated fraud flags. The goal is to convert the CSV files into pandas DataFrames for further analysis.

The file starts by importing the `pandas` library and defining two lists `liste_tag` and `liste_colonnes_X`. `liste_tag` lists the different tags or attributes in the transaction data, while `liste_colonnes_X` defines the column names of the resulting pandas DataFrame after preprocessing.

The main function in this file is `convertToDataFrame`, which reads in a CSV file and converts it into a pandas DataFrame. The function takes two inputs:
    - `path`: a string representing the file path of the CSV file to be read in
    - `liste_colonnes`: a list of strings representing the column names of the resulting pandas DataFrame

The function starts by reading in the CSV file using `pandas.read_csv`, and then renames the columns of the resulting DataFrame using the `liste_colonnes` input. Finally, the first row of the DataFrame is dropped and the resulting DataFrame is returned.

Finally, the file contains a `__main__` block, which uses the `convertToDataFrame` function to convert four separate CSV files into four separate pandas DataFrames. These four DataFrames are:
    - `df_X_train`: a DataFrame containing the training data for the transaction attributes
    - `df_X_test`: a DataFrame containing the testing data for the transaction attributes
    - `df_Y_train`: a DataFrame containing the training data for the fraud flags
    - `df_Y_test`: a DataFrame containing the testing data for the fraud flags

The resulting DataFrames are then printed to the console to verify their correctness.
"""
import pandas as pd

n = 25
liste_tag = ['item', 'cash_price', 'make', 'model', 'goods_code', 'Nbr_of_prod_purchas']
liste_colonnes_X = ['ID'] + [tag+str(i) for tag in liste_tag for i in range(1, n)] + ['Nb_of_items']
liste_colonnes_Y = ['index', 'ID', 'fraud_flag']

def convertToDataFrame(path: str, liste_colonnes: "list[str]") -> pd.DataFrame:
    """
    Converts a CSV file to a pandas DataFrame and renames the columns.

    Parameters:
    path (str): The file path of the input CSV file.
    liste_colonnes (list[str]): A list of column names to rename the columns in the data frame.

    Returns:
    pd.DataFrame: The data frame created from the CSV file with renamed columns.
    """
    df = pd.read_csv(path, sep=',', header=None, low_memory=False)
    for i, col in enumerate(df.columns):
        df.rename(columns={col: liste_colonnes[i]}, inplace=True)

    df = df.drop(df.index[0])
    return df


def importdata():
    df_X_train = convertToDataFrame(
        path='data/X_train.csv', liste_colonnes=liste_colonnes_X)
    df_X_test = convertToDataFrame(
        path='data/X_test.csv', liste_colonnes=liste_colonnes_X)
    df_Y_train = convertToDataFrame(
        path='data/Y_train.csv', liste_colonnes=liste_colonnes_Y)
    df_Y_test = convertToDataFrame(
        path='data/Y_test.csv', liste_colonnes=liste_colonnes_Y)

    return df_X_train, df_X_test, df_Y_train, df_Y_test


def importrelevantdata(path_X_train='data/X_train.csv', path_X_test='data/X_test.csv', path_Y_train='data/Y_train.csv', path_Y_test='data/Y_test.csv',liste_tag_w = ['goods_code', 'cash_price', 'Nbr_of_prod_purchas']):
    df_X_train = convertToDataFrame(
        path=path_X_train, liste_colonnes=liste_colonnes_X)
    df_X_test = convertToDataFrame(
        path=path_X_test, liste_colonnes=liste_colonnes_X)
    df_Y_train = convertToDataFrame(
        path=path_Y_train, liste_colonnes=liste_colonnes_Y)
    df_Y_test = convertToDataFrame(
        path=path_Y_test, liste_colonnes=liste_colonnes_Y)

    liste_colonnes_X_w = [
        'ID'] + [tag+str(i) for tag in liste_tag_w for i in range(1, n)] + ['Nb_of_items']

    df_X_train = df_X_train[liste_colonnes_X_w]
    df_X_test = df_X_test[liste_colonnes_X_w]

    return df_X_train, df_X_test, df_Y_train, df_Y_test


if __name__ == "__main__":
    """
    Reads four CSV files and converts them to four separate pandas DataFrames.
    """
    # df_X_train, df_X_test, df_Y_train, df_Y_test = importdata()

    # print(df_X_train.head())
    # print(df_X_test.head())
    # print(df_Y_train.head())
    # print(df_Y_test.head())

    # print(df_X_train.columns)

    df_X_train, df_X_test, df_Y_train, df_Y_test = importrelevantdata()

    # print(df_X_train.head())
    print(df_Y_train)
