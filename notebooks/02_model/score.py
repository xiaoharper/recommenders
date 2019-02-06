## This just a scoring script similar to that seen in batch scoring...

# set the environment path to find Recommenders
import pickle
import dask.dataframe as dd

## for reco_utils
import sys
sys.path.append("../../")
## must copy reco_utils here!

# import itertools
import logging

from reco_utils.recommender.sar.sar_singlenode import SARSingleNode

print("System version: {}".format(sys.version))
# print("Pandas version: {}".format(pd.__version__))

# top k items to recommend
TOP_K = 10
# Select Movielens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '20m'
# User Ids to filter
MIN_UID = 1
MAX_UID = 1000

## read in the model.
model_file = 'sar_model_%s_fit0.pkl' %(MOVIELENS_DATA_SIZE)
print('Reading %s as model file...' %(model_file))
with open(model_file, "rb") as input_file:
    model = pickle.load(input_file)

## read in teh relevant user ratings data
ratings_parquet_name = 'ratings_%s.parquet' %(MOVIELENS_DATA_SIZE)
df = dd.read_parquet(ratings_parquet_name)
print('Reading %s as ratings file...' %(ratings_parquet_name))
df2 = df[(df.UserId >= MIN_UID) & (df.UserId < MAX_UID)].compute()

## create scores
model.update(df2)
## create top_k
top_k = model.recommend_k_items(df2)
