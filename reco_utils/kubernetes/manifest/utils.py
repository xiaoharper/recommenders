# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from reco_utils.kubernetes import manifest


def int_range_parameter(name, min_val, max_val):
    return manifest.RANGE_PARAM.format(name, 'int', min_val, max_val)


def float_range_parameter(name, min_val, max_val):
    return manifest.RANGE_PARAM.format(name, 'double', min_val, max_val)


def list_parameter(name, list_val):
    list_items = "".join([manifest.LIST_ITEM.format(v) for v in list_val])
    return manifest.LIST_PARAM.format(name, list_items)

# {WORKER_KIND}: TFJob - {WORKER_API_VER}: kubeflow.org/v1
#                Job - {WORKER_API_VER}: batch/v1

# {GOAL}: maximize or minimize
# {PRIMARY_METRIC}
# {IDEAL_METRIC_VALUE}
# {METRICS_LIST}
# {HYPERPARAM}


def make_hypertune_manifest(
    study_name,
    tag=None,
    search_type=manifest.SearchType.RANDOM,
    total_runs=1,
    concurrent_runs=1,
    primary_metric=None,
    goal=manifest.Goal.MAXIMIZE,
    ideal_metric_value=1.0,
    metrics=None,
    hyperparam=None,
    container_image=None,
    entry_script=None,
    param=None,
    storage_path=None,
    use_gpu=False,
):
    """Generate hyperparameter tuning StudyJob manifest.

    Args:
        study_name (str): Study name. StudyJob name will be <study_name>(-<tag>)
        tag (str): Additional tag. E.g., if study_name = 'mnist-random' and tag = '1',
            StudyJob name will be 'mnist-random-1'
        search_type (manifest.SearchType): Hyperparameter search algorithm
        total_runs (int): Number of total trials
        concurrent_runs (int): Number of concurrent trials
        primary_metric (str): Primary evaluation metric to optimize on
        goal (manifest.Goal): Whether minimize or maximize the primary metric
        ideal_metric_value (float): The best possible value of the primary metric
        metrics (list): List of evaluation metrics to track
        hyperparam (list): List of hyperparameter.
            E.g. [float_range_parameter('--learning-rate', 0.001, 0.05), ...]
        container_image (str): Name of the docker image
        entry_script (str): Path to the entry script in the image
        param (list): List of parameter to run the entry script.
            E.g. ["--log-dir=/train/{{{{.WorkerID}}}}", ...]
        storage_path (str): Mounted path of the persistent volume
        use_gpu (bool)

    Returns:
        str: StudyJob name
        str: StudyJob filename
    """

    # TODO use {{{{.StudyID}}}} for subpath

    if search_type not in {'random', 'bayesian'}:
        raise ValueError("Search type should be either 'random' or 'bayesian'")
    if total_runs < concurrent_runs:
        raise ValueError("Total runs should be equal or greater than concurrent runs")

    template_file = os.path.join('manifest', 'template.train.yaml')

    studyjob_name = '{}-{}'.format(study_name, search_type)
    if tag is not None:
        studyjob_name = studyjob_name + '-{}'.format(tag)

    os.makedirs(JOB_DIR, exist_ok=True)
    studyjob_file = os.path.join(JOB_DIR, '{}.yaml'.format(studyjob_name))

    spec = ""
    if search_type == 'random':
        spec = RANDOM_SPEC.format(concurrent_runs)
    elif search_type == 'bayesian':
        spec = BAYESIAN_SPEC.format(concurrent_runs, concurrent_runs)  # second param is 'burn-in'

    request_count = str(math.ceil(total_runs / concurrent_runs))

    generate_yaml_from_template(
        template_file,
        studyjob_file,
        **{
            'NAME': studyjob_name,
            'SPEC': spec,
            'REQUEST_COUNT': request_count,
        }
    )

    print("StudyJob spec has been generated. To start, run 'kubectl create -f {}'".format(studyjob_file))
    return studyjob_name, studyjob_file


def generate_model_testing_yaml(study_name, study_id, model_id):
    tfjob_name = '{}-test'.format(study_name)
    tfjob_file = '{}.yaml'.format(tfjob_name)

    generate_yaml_from_template(
        os.path.join('manifest', 'template.test.yaml'),
        tfjob_file,
        **{
            'NAME': tfjob_name,
            'MODEL_DIR': "\"/tmp/tensorflow/{0}/{1}_model\"".format(study_id, model_id)
        }
    )
    return tfjob_name, tfjob_file


def generate_yaml_from_template(template, filename, **kwargs):
    with open(template, 'r') as rf:
        tmp = rf.read()
        if kwargs is not None:
            for k, v in kwargs.items():
                tmp = tmp.replace('{{{}}}'.format(k), v)
        with open(filename, 'w') as wf:
            wf.write(tmp)