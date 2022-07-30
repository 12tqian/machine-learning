import pandas as pd
import numpy as np
import itertools
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt
from multiprocessing import Pool
import logging
import sys

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=FORMAT)

hitters_df = pd.read_csv('data/Hitters.csv')
hitters_df.head()

logging.info("Number of null values:", hitters_df["Salary"].isnull().sum())

# logging.info the dimensions of the original Hitters data (322 rows x 20 columns)
logging.info("Dimensions of original data:", hitters_df.shape)

# Drop any rows the contain missing values, along with the player names
hitters_df_clean = hitters_df

# logging.info the dimensions of the modified Hitters data (263 rows x 20 columns)
logging.info("Dimensions of modified data:", hitters_df_clean.shape)

# One last check: should return 0
logging.info("Number of null values:", hitters_df_clean["Salary"].isnull().sum())

dummies = pd.get_dummies(hitters_df_clean[['League', 'Division', 'NewLeague']])

y = hitters_df_clean.Salary

# Drop the column with the independent variable (Salary), and columns for which we created dummy variables
X_ = hitters_df_clean.drop(['Salary', 'League', 'Division', 'NewLeague'], axis=1).astype('float64')

# Define the feature set X.
X = pd.concat([X_, dummies[['League_N', 'Division_W', 'NewLeague_N']]], axis=1)

def process_subset(feature_set):
    # Fit model on feature_set and calculate RSS
    model = sm.OLS(y,X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model":regr, "RSS":RSS}

def get_best(k):
    results = []
    
    # multiprocessing fails in jupyter
    combos = list(itertools.combinations(X.columns, k))
    
    tic = time.time()
    
    pool = Pool(processes=12)
    results = pool.map(process_subset, combos)

    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
    
    toc = time.time()
    logging.info("Processed", models.shape[0], "models on", k, "predictors in", (toc - tic), "seconds.")
    
    # Return the best model, along with some other useful information about the model
    return best_model

models_best = pd.DataFrame(columns=["RSS", "model"])

tic = time.time()
for i in range(1, 8):
    models_best.loc[i] = get_best(i)

toc = time.time()
logging.info("Total elapsed time:", (toc - tic), "seconds.")