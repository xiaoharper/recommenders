# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# TODO we can make the script more general so that AzureML and Kubeflow can use the same file for training
import argparse
import os

import surprise
import pandas as pd

# Evaluation functions will be called by name
from reco_utils.evaluation.python_evaluation import (
    rmse,
    mae,
    rsquared,
    exp_var,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    map_at_k,
)
from reco_utils.recommender.surprise.surprise_utils import (
    compute_rating_predictions,
    compute_ranking_predictions,
)


parser = argparse.ArgumentParser()
# StudyID and TrialID. Passed by StudyJob controller
parser.add_argument("--study-id", type=str, dest="study_id")
parser.add_argument("--trial-id", type=str, dest="trial_id")
# Data paths
parser.add_argument("--datastore", type=str, dest="datastore", help="Datastore path")
parser.add_argument("--train-datapath", type=str, dest="train_datapath")
parser.add_argument("--test-datapath", type=str, dest="test_datapath")
parser.add_argument("--output-dir", type=str, dest="output_dir", help="output directory")
parser.add_argument("--surprise-reader", type=str, dest="surprise_reader")
parser.add_argument("--usercol", type=str, dest="usercol", default="userID")
parser.add_argument("--itemcol", type=str, dest="itemcol", default="itemID")
# Metrics
parser.add_argument(
    "--rating-metrics", type=str, nargs="*", dest="rating_metrics", default=[]
)
parser.add_argument(
    "--ranking-metrics", type=str, nargs="*", dest="ranking_metrics", default=[]
)
parser.add_argument("--k", type=int, dest="k", default=None)
parser.add_argument("--remove-seen", dest="remove_seen", action="store_true")

args = parser.parse_args()

model_dir = os.path.join(
    args.datastore,
    "{}-{}".format(args.output_dir, args.study_id),
    args.trial_id
)

if not os.path.exists(model_dir):
    raise ValueError("Model does not exist at {}".format(model_dir))

# Load data
train_data = pd.read_pickle(
    path=os.path.join(args.datastore, args.train_datapath)
)
test_data = pd.read_pickle(
    path=os.path.join(args.datastore, args.test_datapath)
)

# SVD test
svd = surprise.dump.load(os.path.join(model_dir, 'model.dump'))[1]

test_results = {}

rating_metrics = args.rating_metrics
if len(rating_metrics) > 0:
    predictions = compute_rating_predictions(svd, test_data, usercol=args.usercol, itemcol=args.itemcol)
    for metric in rating_metrics:
        test_results[metric] = eval(metric)(test_data, predictions)

ranking_metrics = args.ranking_metrics
if len(ranking_metrics) > 0:
    all_predictions = compute_ranking_predictions(
        svd,
        train_data,
        usercol=args.usercol,
        itemcol=args.itemcol,
        remove_seen=args.remove_seen)
    for metric in ranking_metrics:
        test_results[metric] = eval(metric)(test_data, all_predictions, col_prediction='prediction', k=args.k)

print(test_results)
