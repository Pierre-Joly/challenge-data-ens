# Challenge Data ENS - BNP Paribas Personal Finance

This repository presents my work on the BNP Paribas Personal Finance challenge from [Challenge Data ENS](https://challengedata.ens.fr/challenges/104)

Find below the official description of the challenge. The English version is available at the end of the document.

## Comment démasquer les fraudeurs ? (*BNP*)

### Contexte
BNP Paribas Personal Finance est le n°1 du financement aux particuliers en France et en Europe au travers de ses activités de crédit à la consommation.

Filiale à 100% du groupe BNP Paribas, BNP Paribas Personal Finance compte plus de 20 000 collaborateurs et opère dans une trentaine de pays. Avec des marques comme Cetelem, Cofinoga ou encore Findomestic, l'entreprise propose une gamme complète de crédits aux particuliers disponibles en magasin, en concession automobile ou directement auprès des clients via ses centres de relation client et sur internet. BNP Paribas Personal Finance a complété son offre avec des produits d'assurance et d'épargne pour ses clients en Allemagne, Bulgarie, France et Italie. BNP Paribas Personal Finance a développé une stratégie active de partenariat avec les enseignes de distribution, les constructeurs et les distributeurs automobiles, les webmarchands, et d'autres institutions financières (banque et assurance), fondée sur son expérience du marché du crédit et sa capacité à proposer des services intégrés adaptés à l'activité et à la stratégie commerciale de ses partenaires. Il est aussi un acteur de référence en matière de crédit responsable et d'éducation budgétaire.

BNPP Personal Finance est, par nature, exposée au Risque de Crédit, et s'appuie fortement sur des modèles quantitatifs pour le gérer. La direction Central Risk Department s'assure de la pertinence des modèles de notations de risque utilisés par les entités locales et garantit un niveau élevé d'expertise en intégrant des nouvelles techniques de modélisation dans notre environnement.

L'équipe Credit Process Optimization fait partie du département Risk Personal Finance Credit Decision Process and Policies, elle contribue à la rationalisation et à l'optimisation du process risque de décision par une approche analytique. Nous soutenons les équipes Risk en local pour améliorer l'efficacité de leur process de décision crédit, y compris sur la partie fraude. Il s'agit de trouver le meilleur compromis entre la profitabilité, l'expérience client et les profils de risque.

La fraude est un problème majeur de plus en plus préoccupant pour les institutions financières du monde entier. Les criminels utilisent une grande variété de méthodes pour attaquer des organisations comme la nôtre, quels que soient les systèmes, les canaux, les process ou les produits.

***Le développement de méthodes de détection de la fraude est stratégique et essentiel pour nous. Les fraudeurs s'avèrent toujours très créatifs et ingénieux pour normaliser leurs comportements et les rendre difficilement identifiables. Une contrainte s'ajoute à cette problématique, la faible occurence de la fraude dans notre population.***

### But

L'objectif de ce challenge est de trouver la meilleure méthode pour transformer et agréger les données relatives au panier client d'un de nos parteneraires pour détecter les cas de fraude.

En utilisant ces données panier, les fraudeurs pourront être détectés, et ainsi refusés dans le futur.

### 1. Base de données

La base contient une liste d'achats effectués chez notre partenaire que nous avons financés. Les informations décrivent exclusivement le contenu du panier.

Pour chaque observation de la base, il y a 147 colonnes dont 144 peuvent être regroupées en 6 catégories :

    - item,

    - cash_price,

    - make,

    - model,

    - goods_code,

    - Nbr_of_prod_purchas.

Le panier se décompose au maximum en 24 items. Par exemple, si un panier contient 3 items alors toutes les informations relatives à ces 3 items seront renseignées dans les colonnes item1, item2, item3, cash_price1, cash_price_2, cash_price3, make1, make2, make3, model1, model2, model3, goods_code1, goods_code2, goods_code3, Nbr_of_prod_purchas1, Nbr_of_prod_purchas2 et Nbr_of_prod_purchas3. Les variables restantes (celles avec un indice > 3) seront vides .

Un item correspond à un produit ou un regroupement de produits équivalents. Par exemple, si un panier contient 3 Iphones 14, alors ces 3 produits seront regroupés dans un seul item. Par contre, si le client achète 3 produits Iphone différents, alors nous considèrerons ici 3 items.

La variable Nb_of_items correspond au nombre d'items dans le panier, tandis que la somme des variables Nbr_of_prod_purchas correspond au nombre de produits.

L’indicatrice fraud_flag permet de savoir si l’observation a été identifiée comme frauduleuse ou non.
#### 1.1. Description des variables en entrée (X)

ID (Num) : Identifiant unique
item1 à item24 (Char) :	Catégorie du bien de l'item 1 à 24 
cash_price1 à cash_price24 (Num) : Prix de l'item 1 à 24 	
make1 à make24 (Char) : Fabriquant de l'item 1 à 24 	
model1 à model24 (Char) : Description du modèle de l'item 1 à 24 	
goods_code1 à goods_code24 (Char) : Code de l'item 1 à 24 	
Nbr_of_prod_purchas1 à Nbr_of_prod_purchas24 (Num) : Nombre de produits dans l'item 1 à 24 	
Nb_of_items (Num) : Nombre total d'items 	

		
#### 1.2. Description de la variable de sortie (Y)

ID (Num)  : Identifiant unique
fraud_flag (Num) : Fraude = 1, Non Fraude = 0

	
#### 1.3. Taille de la base

Taille : 115 988 observations, 147 colonnes.

Distribution de Y :

    Fraude (Y=1) : 1 681 observations

    Non Fraude (Y=0) : 114 307 observations

Le taux de fraude sur l'ensemble de la base est autour de 1.4%.
### 2. Echantillons

La méthode d'échantillonnage appliquée est un tirage aléatoire simple sans remise. Ainsi, 80% de la base initiale a été utilisée pour générer l'échantillon de training et 20% pour l'échantillon de test.
#### 2.1. Echantillon d'entraînement

Taille : 92 790 observations, 147 colonnes.

Distribution de Y_train :

    Fraude (Y=1) : 1 319 observations

    Non Fraude (Y=0) : 91 471 observations

#### 2.2. Echantillon de test

Taille : 23 198 observations, 147 colonnes.

Distribution de Y_test :

    Fraude (Y=1) : 362 observations

    Non Fraude (Y=0) : 22 836 observations

Description du benchmark

Métrique d'évaluation

L'objectif est d'identifier une opération frauduleuse dans la population en prédisant un risque/probabilité de fraude. Par conséquent, la métrique à utiliser est l'aire sous la courbe Précision-Rappel, appelé également PR-AUC.

La courbe Précision-Rappel s'obtient en traçant la précision (TPTP+FNTP+FNTP​ ) sur l'axe des ordonnées et le rappel (TPTP+FPTP+FPTP​ ) sur l'axe des abcisses pour tout seuil de probabilité compris entre 0 et 1.

Cette métrique est appropriée pour évaluer correctement la performance d'un modèle sur la classe minoritaire dans le cadre d'une classification fortement déséquilibrée.

Le meilleur modèle correspondra à celui avec la valeur de PR-AUC la plus élevée.

Pour ce challenge, la PR-AUC sera estimée par la moyenne pondérées des précisions à chaque seuil avec le poids associé étant la variation du rappel entre le seuil précédent et le nouveau seuil :

PR-AUC=Σn(Rn−Rn−1)PnPR-AUC=Σn​(Rn​−Rn−1​)Pn​ ,

où PnPn​ et RnRn​ sont les précisions et recall du nème seuil.

N.B. Cette implémentation correspond à la métrique average_precision_score de sklearn.

Par conséquent, votre fichier de submission devra avoir le format suivant :

ID (Num) : Identifiant unique
fraud_flag (Num) : Probabilité estimée (décimale positive entre 0 et 1) pour la classe minoritaire (1). Plus la valeur est élevée, plus la probabilité est forte d'être une opération frauduleuse.

Vous pouvez utiliser les fichiers .csv Y_test_random et Y_test_benchmark pour vérifier le format attendu et la taille de votre fichier de submission pour ce challenge.
Benchmarks

    Benchmark 1 : PR-AUC1=0,017PR-AUC1​=0,017
    Le premier benchmark est naïf et considère un modèle qui prédit aléatoirement une probabilité entre 0 et 1.

    Benchmark 2 : PR-AUC2=0,14PR-AUC2​=0,14
    Le second benchmark intègre plusieurs étapes de pré-processing et utilise un modèle de Machine Learning optimisé pour prédire le risque de fraude.

-------------------------------------------------------------------
English version

## How to unmask fraudsters? (BNP)

### Context

BNP Paribas Personal Finance is the number one provider of personal financing in France and Europe through its consumer credit activities.

A 100% subsidiary of the BNP Paribas Group, BNP Paribas Personal Finance has more than 20,000 employees and operates in around thirty countries. With brands like Cetelem, Cofinoga, or Findomestic, the company offers a comprehensive range of consumer credit products available in stores, car dealerships, or directly to customers through its customer relationship centers and online. BNP Paribas Personal Finance has expanded its offering with insurance and savings products for its customers in Germany, Bulgaria, France, and Italy. BNP Paribas Personal Finance has developed an active partnership strategy with retail chains, manufacturers and car dealers, e-merchants, and other financial institutions (banks and insurance), based on its market experience and ability to offer integrated services tailored to the activity and commercial strategy of its partners. It is also a leading player in responsible credit and budgetary education.

BNPP Personal Finance is, by nature, exposed to Credit Risk and relies heavily on quantitative models to manage it. The Central Risk Department ensures the relevance of the risk rating models used by local entities and guarantees a high level of expertise by integrating new modeling techniques into our environment.

The Credit Process Optimization team is part of the Risk Personal Finance Credit Decision Process and Policies department. It contributes to the rationalization and optimization of the risk decision process through an analytical approach. We support local Risk teams in improving the efficiency of their credit decision process, including fraud detection. The goal is to find the best balance between profitability, customer experience, and risk profiles.

Fraud is an increasingly major concern for financial institutions worldwide. Criminals use a wide variety of methods to attack organizations like ours, regardless of systems, channels, processes, or products.

Developing fraud detection methods is strategic and essential for us. Fraudsters always prove to be very creative and ingenious in normalizing their behavior and making it difficult to identify. One constraint adds to this problem: the low occurrence of fraud in our population.

Objective
The objective of this challenge is to find the best method to transform and aggregate customer basket data from one of our partners to detect fraud cases.

By using this basket data, fraudsters can be detected and subsequently denied in the future.

1. Database
The database contains a list of purchases made at our partner's store that we have financed. The information describes exclusively the content of the basket.

For each observation in the database, there are 147 columns, of which 144 can be grouped into 6 categories:

- item,

- cash_price,

- make,

- model,

- goods_code,

- Nbr_of_prod_purchas.

The basket breaks down into a maximum of 24 items. For example, if a basket contains 3 items, all the information relating to these 3 items will be entered in the columns item1, item2, item3, cash_price1, cash_price_2, cash_price3, make1, make2, make3, model1, model2, model3, goods_code1, goods_code2, goods_code3, Nbr_of_prod_purchas1, Nbr_of_prod_purchas2, and Nbr_of_prod_purchas3. The remaining variables (those with an index > 3) will be empty.

An item corresponds to a product or a group of equivalent products. For example, if a basket contains 3 iPhone 14s, these 3 products will be grouped into a single item. On the other hand, if the customer buys 3 different iPhone products, we will consider 3 items here.

The variable Nb_of_items corresponds to the number of items in the basket, while the sum of the Nbr_of_prod_purchas variables corresponds to the number of products.

The fraud_flag indicator allows us to know whether the observation has been identified as fraudulent or not.

#### 1.1. Description of input variables (X)

ID (Num): Unique identifier
item1 to item24 (Char): Category of item 1 to 24
cash_price1 to cash_price24 (Num): Price of item 1 to 24
make1 to make24 (Char): Manufacturer of item 1 to 24
model1 to model24 (Char): Description of the model of item 1 to 24
goods_code1 to goods_code24 (Char): Code of item 1 to 24
Nbr_of_prod_purchas1 to Nbr_of_prod_purchas24 (Num): Number of products in item 1 to 24
Nb_of_items (Num): Total number of items

#### 1.2. Description of output variable (Y)

ID (Num): Unique identifier
fraud_flag (Num): Fraud = 1, Non-Fraud = 0

#### 1.3. Database size

Size: 115,988 observations, 147 columns.

Y distribution:

Fraud (Y=1): 1,681 observations

Non-Fraud (Y=0): 114,307 observations

The fraud rate in the entire database is around 1.4%.

### 2. Samples
The sampling method applied is simple random sampling without replacement. Thus, 80% of the initial database was used to generate the training sample and 20% for the test sample.

#### 2.1. Training sample

Size: 92,790 observations, 147 columns.

Y_train distribution:

Fraud (Y=1): 1,319 observations

Non-Fraud (Y=0): 91,471 observations

#### 2.2. Test sample

Size: 23,198 observations, 147 columns.

Y_test distribution:

Fraud (Y=1): 362 observations

Non-Fraud (Y=0): 22,836 observations

Description of the benchmark

Evaluation metric

The objective is to identify a fraudulent operation in the population by predicting a risk/probability of fraud. Therefore, the metric to use is the area under the Precision-Recall curve, also called PR-AUC.

The Precision-Recall curve is obtained by plotting the precision (TP/(TP+FN)) on the y-axis and the recall (TP/(TP+FP)) on the x-axis for all probability thresholds between 0 and 1.

This metric is suitable for properly evaluating the performance of a model on the minority class in the context of a highly imbalanced classification.

The best model will correspond to the one with the highest PR-AUC value.

For this challenge, the PR-AUC will be estimated by the weighted average of the precisions at each threshold, with the associated weight being the variation in recall between the previous threshold and the new threshold:

PR-AUC=Σn(Rn−Rn−1)Pn,

where Pn and Rn are the precisions and recall of the nth threshold.

N.B. This implementation corresponds to the average_precision_score metric from sklearn.

Therefore, your submission file should have the following format:

ID (Num): Unique identifier
fraud_flag (Num): Estimated probability (positive decimal between 0 and 1) for the minority class (1). The higher the value, the higher the probability of being a fraudulent operation.

You can use the .csv files Y_test_random and Y_test_benchmark to check the expected format and size of your submission file for this challenge.
Benchmarks

Benchmark 1: PR-AUC1=0.017
The first benchmark is naive and considers a model that randomly predicts a probability between 0 and 1.

Benchmark 2: PR-AUC2=0.14
The second benchmark integrates several pre-processing steps and uses an optimized Machine Learning model to predict the risk of fraud.

