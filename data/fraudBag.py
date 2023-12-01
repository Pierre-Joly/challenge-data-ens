from convertDataFrame import convertToDataFrame
from typing import List, Dict, Tuple
import pandas as pd
import itertools


def getItems(X: pd.DataFrame) -> List[str]:
    n = 25
    items = ['item' + str(i) for i in range(1, n)]
    df_X_items = X[items]
    df_X_items = df_X_items.fillna(value='')
    df_X_items['all_items'] = df_X_items[items].apply(
        lambda x: [item for item in x.tolist() if item != ''], axis=1)
    df_X_items = df_X_items.drop(items, axis=1)
    df_X_items = df_X_items.explode('all_items')
    df_X_items = df_X_items.values.tolist()
    df_X_items = [item[0] for item in df_X_items]
    return df_X_items


def getOccurence(df_X_items: List[str]) -> Dict[str, int]:
    dict = {}
    for item in df_X_items:
        if item in dict:
            dict[item] += 1
        else:
            dict[item] = 1
    return dict

def count_unique_items_across_columns(df: pd.DataFrame, base_columns: List[str], n: int) -> Dict[str, int]:
    unique_counts = {}
    for base_col in base_columns:
        combined_series = pd.Series(dtype=str)
        for i in range(1, n + 1):
            combined_series = combined_series.append(df[base_col + str(i)])
        unique_counts[base_col] = combined_series.nunique()
    return unique_counts

def selectItems(dict: Dict[str, int], threshold: int) -> Dict[str, int]:
    dict = {k: v for k, v in dict.items() if v > threshold}
    return dict


def getFraudItems(X: pd.DataFrame, Y: pd.DataFrame) -> pd.DataFrame:
    Y: pd.DataFrame = Y[Y['fraud_flag'] == '1']
    X: pd.DataFrame = X[X['ID'].isin(Y['ID'])]
    return X


def getStat(X: pd.DataFrame) -> Tuple[float, float, float, float]:
    nb_item = X['Nb_of_items'].astype(float)
    Nb_item_max = nb_item.max()
    Nb_item_min = nb_item.min()
    Nb_item_mean = nb_item.mean()
    Nb_item_var = nb_item.var()
    return Nb_item_max, Nb_item_min, Nb_item_mean, Nb_item_var


if __name__ == '__main__':
    # Get X_train and Y_train
    n = 25
    liste_tag = ['item', 'cash_price', 'make',
                 'model', 'goods_code', 'Nbr_of_prod_purchas']
    liste_colonnes_X = [
        'ID'] + [tag+str(i) for tag in liste_tag for i in range(1, n)] + ['Nb_of_items']
    liste_colonnes_Y = ['index', 'ID', 'fraud_flag']
    df_X_train = convertToDataFrame(
        path='data/X_train.csv', liste_colonnes=liste_colonnes_X)
    df_Y_train = convertToDataFrame(
        path='data/Y_train.csv', liste_colonnes=liste_colonnes_Y)

    # Get X_fraud
    df_X_fraud = getFraudItems(df_X_train, df_Y_train)

    # Get statistics on X_fraud items
    Nb_item_max, Nb_item_min, Nb_item_mean, Nb_item_var = getStat(df_X_fraud)

    print("Nombre max d'item pour un panier fraduleux : " + str(Nb_item_max))
    print("Nombre min d'item pour un panier fraduleux : " + str(Nb_item_min))
    print("Nombre moyen d'item pour un panier fraduleux : " + str(Nb_item_mean))
    print("Variance du nombre d'item pour un panier fraduleux : " + str(Nb_item_var))

    # Get items in X_fraud
    df_X_items = getItems(df_X_fraud)
    dict = getOccurence(df_X_items)

    print("Nombre d'ocurrence des objects dans les paniers frauduleux : ", dict)
    print("Nombre d'occurence nécessaire pour appartenir à 5% des paniers : ",
          int(len(df_X_fraud)*0.05))

    # Select items that appear in more than 5% of the fraud baskets
    dict = selectItems(dict, int(len(df_X_fraud)*0.05))

    print("Nombre d'ocurrence des objects qui apparaissent dans plus de 5% des paniers frauduleux : ", dict)
    print("Liste des objets qui apparaissent dans plus de 5% des paniers frauduleux : ", dict.keys())
    print("Nombre d'objets qui apparaissent dans plus de 5% des paniers frauduleux : ", len(
        dict.keys()))

    df_X_all_items = getItems(df_X_train)
    dict = getOccurence(df_X_all_items)

    print("Nombre d'ocurrence des objects dans les paniers : ", dict)
    print("Nombre d'occurence nécessaire pour appartenir à 5% des paniers : ",
          int(len(df_X_train)*0.05))

    # Select items that appear in more than 5% of the baskets
    dict = selectItems(dict, int(len(df_X_train)*0.05))

    print("Nombre d'ocurrence des objects qui apparaissent dans plus de 5% des paniers : ", dict)
    print("Liste des objets qui apparaissent dans plus de 5% des paniers : ", dict.keys())
    print("Nombre d'objets qui apparaissent dans plus de 5% des paniers : ", len(
        dict.keys()))

    # Get the probability of presence of an item in a basket and then a fraud basket
    dict = getOccurence(df_X_items)
    dict_all = getOccurence(df_X_all_items)
    proba_present_fraud = {k: v/len(df_X_fraud) for k, v in dict.items()}
    proba_present_gen = {k: v/len(df_X_train) for k, v in dict_all.items()}
    proba_fraud_si_present = {k: dict[k]/dict_all[k] for k in dict.keys()}
    print("Probabilité d'apparition d'un objet dans un panier frauduleux : ",
          proba_present_fraud)
    print("Probabilité d'apparition d'un objet dans un panier : ", proba_present_gen)
    print("Probabilité d'apparition d'un objet dans un panier frauduleux sachant qu'il est présent : ",
          proba_fraud_si_present)

    # I want to get the key and values of the top 5 values of proba_present_fraud, proba_present_gen and proba_fraud_si_present
    top_present_fraud = itertools.islice(
        sorted(proba_present_fraud.items(), key=lambda x: x[1], reverse=True), 5)
    top_present_gen = itertools.islice(
        sorted(proba_present_gen.items(), key=lambda x: x[1], reverse=True), 5)
    top_fraud_si_present = itertools.islice(
        sorted(proba_fraud_si_present.items(), key=lambda x: x[1], reverse=True), 5)

    print("Top 5 des objets qui apparaissent dans plus de 5% des paniers frauduleux : ", list(
        top_present_fraud))
    print("Top 5 des objets qui apparaissent dans plus de 5% des paniers : ", list(
        top_present_gen))
    print("Top 5 des objets qui apparaissent dans plus de 5% des paniers frauduleux sachant qu'ils sont présents : ", list(
        top_fraud_si_present))
    
    # I want to get the number of different items in each categorical column
    base_columns = ['item', 'make', 'model', 'goods_code']  # Base names of your categorical columns
    n = 24  # Number of instances for each category
    unique_item_counts = count_unique_items_across_columns(df_X_train, base_columns, n)
    print(unique_item_counts)
    

