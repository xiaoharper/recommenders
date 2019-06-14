# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from reco_utils.kubernetes.manifest_constants import *


def _range_parameter(name, param_type, min_val, max_val):
    return RANGE_PARAM.format(name, param_type, min_val, max_val)


def _list_parameter(name, list_val):
    list_items = "".join([LIST_ITEM.format(v) for v in list_val])
    return LIST_PARAM.format(name, list_items)

# {WORKER_KIND}: TFJob - {WORKER_API_VER}: kubeflow.org/v1
#                Job - {WORKER_API_VER}: batch/v1

# {GOAL}: maximize or minimize
# {PRIMARY_METRIC}
# {IDEAL_METRIC_VALUE}
# {METRICS_LIST}
# {HYPERPARAM}
# {WORKER_SPEC} :"gpuTFJobTemplate.yaml" or "cpuTFJobTemplate.yaml" or "gpuJobTemplate.yaml"  or "cpuJobTemplate.yaml"
