## This just a scoring script similar to that seen in batch scoring...

# set the environment path to find Recommenders
import pickle
import dask.dataframe as dd

## for reco_utils
import os
import sys
sys.path.append("../../")
## must copy reco_utils here!

# import itertools
import logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)-8s %(message)s')
                    
from reco_utils.recommender.sar.sar_singlenode import SARSingleNode

print("System version: {}".format(sys.version))
# print("Pandas version: {}".format(pd.__version__))

## User Ids to filter
MIN_UID = int(sys.argv[1])
MAX_UID = int(sys.argv[2])

## directories used by AML
inputs_dir = sys.argv[3]
models_dir = sys.argv[4]
outputs_dir = sys.argv[5]

## parameters with defaults:
# Select Movielens data size: 100k, 1m, 10m, or 20m
MOVIELENS_DATA_SIZE = '10m'
# top k items to recommend
TOP_K = 10

if len(sys.argv) > 6:
    TOP_K = int(sys.argv[6])

if len(sys.argv) > 7:
    MOVIELENS_DATA_SIZE = int(sys.argv[7])


## read in the model.
model_file = os.path.join(models_dir, 'sar_model_%s_fit0.pkl' %(MOVIELENS_DATA_SIZE))
print('Reading %s as model file...' %(model_file))
with open(model_file, "rb") as input_file:
    model = pickle.load(input_file)

## read in teh relevant user ratings data
ratings_parquet_name = os.path.join(inputs_dir,'ratings_%s.parquet' %(MOVIELENS_DATA_SIZE))
df = dd.read_parquet(ratings_parquet_name)
print('Reading %s as ratings file...' %(ratings_parquet_name))
df2 = df[(df.UserId >= MIN_UID) & (df.UserId < MAX_UID)].compute()

## create scores
print('Creating scores.')
model.update(df2)
## create top_k
top_k = model.recommend_k_items(df2)

top_k_filename = os.path.join(outputs_dir,'top_k_%s_%s_to_%s.csv' %(MOVIELENS_DATA_SIZE, MIN_UID, MAX_UID))
print('Writing out top k to %s' %(top_k_filename))
top_k.to_csv(top_k_filename)