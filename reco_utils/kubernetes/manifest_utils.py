
RANDOM_SPEC = """
    suggestionAlgorithm: "random"
    requestNumber: {}"""

BAYESIAN_SPEC = """
    suggestionAlgorithm: "bayesianoptimization"
    suggestionParameters:
      -
          name: "burn_in"
          value: "{1}"
    requestNumber: {0}"""

RANGE_PARAMETER = """
    - name: {0}
      parametertype: {1}
      feasible:
        min: \"{2}\"
        max: \"{3}\""""

LIST_PARAMETER = """
    - name: {0}
      parametertype: categorical
      feasible:
        list: {1}"""

LIST_ITEM = """
        - \"{}\""""


def _range_parameter(name, param_type, min_val, max_val):
    return RANGE_PARAMETER.format(name, param_type, min_val, max_val)


def _list_parameter(name, list_val):
    list_items = "".join([LIST_ITEM.format(v) for v in list_val])
    return LIST_PARAMETER.format(name, list_items)


# {GOAL}: maximize or minimize
# {PRIMARY_METRIC}
# {IDEAL_METRIC_VALUE}
# {METRICS_LIST}
METRIC_ITEM = """
    - {}"""

# {PARAMETERS}

# {WORKER_SPEC} :"gpuTFJobTemplate.yaml" or "cpuTFJobTemplate.yaml" or "gpuJobTemplate.yaml"  or "cpuJobTemplate.yaml"
