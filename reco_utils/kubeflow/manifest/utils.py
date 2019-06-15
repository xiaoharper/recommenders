# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
import os
import reco_utils.kubeflow.manifest as manifest


JOB_DIR = "jobs"

# TODO Maybe make API similar to Hyperdrive? e.g.:
# ps = RandomParameterSampling(
#     {
#         '--batch-size': choice(25, 50, 100),
#         '--first-layer-neurons': choice(10, 50, 200, 300, 500),
#         '--second-layer-neurons': choice(10, 50, 200, 500),
#         '--learning-rate': loguniform(-6, -1)
#     }
# )


def uniform(min_val, max_val):
    """Uniform hyperparameter spec

    Args:
        min_val (float)
        max_val (float)

    Returns:
        tuple: Hyperparameter spec
    """
    if min_val > max_val:
        raise ValueError("min_val should be less than max_val")

    return '_uniform', [min_val, max_val]


def _uniform(name, min_max):
    return manifest.UNIFORM_PARAM.format(name, 'double', min_max[0], min_max[1])


def choice(options):
    """Categorical hyperparameter spec

    Args:
        options (list)

    Returns:
        tuple: Hyperparameter spec
    """
    return '_choice', options


def _choice(name, list_val):
    items = "".join([manifest.CHOICE_ITEM.format(v) for v in list_val])
    return manifest.CHOICE_PARAM.format(name, items)


def _format_hyperparam(hyperparams):
    """Format hyperparameter spec by using `_uniform` and `_choice`"""
    return "".join([globals()[v[0]](k, v[1]) for k, v in hyperparams.items()])


def worker_manifest(
    worker_type=manifest.WorkerType.WORKER,
    image_name=None,
    entry_script=None,
    params=None,
    is_hypertune=False,
    storage_path=None,
    use_gpu=False,
):
    """Generate worker job manifest

    Args:
        worker_type (manifest.WorkerType): Type of the worker
        image_name (str): Name of the docker image
        entry_script (str): Path to the entry script in the image
        params (dict): Dictionary of script parameters.
            E.g. {'--log-dir': "/train/{{{{.WorkerID}}}}"}
        is_hypertune (bool): Whether the job is hyperparameter tuning or not
        storage_path (str): Mounted path of the persistent volume
        use_gpu (bool)

    Returns:
        str: Worker manifest yaml string
    """
    return _format_worker_spec(worker_type, image_name, entry_script, params, is_hypertune, storage_path, use_gpu)


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
    hyperparams=None,
    worker_spec=None,
):
    """Generate hyperparameter tuning StudyJob manifest file.

    Args:
        study_name (str): Study name. StudyJob name will be <study_name>(-<tag>)
        tag (str): Additional tag. E.g., if study_name = 'mnist-random' and tag = '1',
            StudyJob name will be 'mnist-random-1'
        search_type (manifest.SearchType): Hyperparameter search algorithm
        total_runs (int): Number of total trials
        concurrent_runs (int): Number of concurrent runs
        primary_metric (str): Primary evaluation metric to optimize on
        goal (manifest.Goal): Whether minimize or maximize the primary metric
        ideal_metric_value (float): The best possible value of the primary metric
        metrics (list): List of evaluation metrics to track
        hyperparams (dict): Dictionary of hyperparameters.
            E.g. {'--learning-rate': uniform(0.001, 0.05)}
        worker_spec (str): Worker manifest

    Returns:
        str: StudyJob name
        str: StudyJob filename
    """
    if not primary_metric:
        raise ValueError("Primary metric should be provided.")
    if total_runs < concurrent_runs:
        raise ValueError("Total runs should be equal or greater than concurrent runs.")

    if tag is None:
        studyjob_name = study_name
    else:
        studyjob_name = study_name + '-{}'.format(tag)

    os.makedirs(JOB_DIR, exist_ok=True)
    studyjob_file = os.path.join(JOB_DIR, '{}.yaml'.format(studyjob_name))

    _make_yaml_from_template(
        os.path.join(os.path.dirname(__file__), 'hypertune.template'),
        studyjob_file,
        **{
            'NAME': studyjob_name,
            'GOAL': goal,
            'PRIMARY_METRIC': primary_metric,
            'IDEAL_METRIC_VALUE': str(ideal_metric_value),
            'REQUEST_COUNT': str(math.ceil(total_runs / concurrent_runs)),
            'METRICS': _format_metrics(metrics),
            'HYPERPARAM': _format_hyperparam(hyperparams),
            'WORKER_SPEC': worker_spec,
            'SPEC': _format_search_spec(search_type, concurrent_runs),
        }
    )

    print("StudyJob spec has been generated. To start, run 'kubectl create -f {}'".format(studyjob_file))
    return studyjob_name, studyjob_file


def _format_search_spec(search_type, concurrent_runs):
    if search_type == manifest.SearchType.RANDOM:
        return manifest.RANDOM_SPEC.format(concurrent_runs)
    elif search_type == manifest.SearchType.BAYESIAN:
        # second param is 'burn-in'
        return manifest.BAYESIAN_SPEC.format(concurrent_runs, concurrent_runs)
    else:
        raise ValueError("Unknown search type {}.".format(search_type))


def _format_worker_spec(worker_type, image_name, entry_script, params, is_hypertune, storage_path, use_gpu):
    if worker_type == manifest.WorkerType.WORKER:
        resources = manifest.WORKER_GPU if use_gpu else ""
        hyperparam = manifest.WORKER_HYPERPARAM if is_hypertune else ""
        return manifest.WORKER_TEMPLATE.format(
            image_name,
            entry_script,
            "".join([manifest.WORKER_PARAM.format("{}={}".format(k, v)) for k, v in params.items()]) + hyperparam,
            "{{{{.StudyID}}}}",
            storage_path,
            resources,
        )
    elif worker_type == manifest.WorkerType.TF_WORKER:
        raise NotImplementedError("TF_WORKER spec is not supported yet.")
    else:
        raise ValueError("Unknown worker type {}.".format(worker_type))


# def generate_model_testing_yaml(study_name, study_id, model_id):
#     tfjob_name = '{}-test'.format(study_name)
#     tfjob_file = '{}.yaml'.format(tfjob_name)
#
#     generate_yaml_from_template(
#         os.path.join('manifest', 'template.test.yaml'),
#         tfjob_file,
#         **{
#             'NAME': tfjob_name,
#             'MODEL_DIR': "\"/tmp/tensorflow/{0}/{1}_model\"".format(study_id, model_id)
#         }
#     )
#     return tfjob_name, tfjob_file


def _format_metrics(metrics):
    return "".join([manifest.METRIC_ITEM.format(m) for m in metrics])


def _make_yaml_from_template(template, filename, **kwargs):
    """Make manifest yaml file by replacing keywords in template with values.
    """
    with open(template, 'r') as rf:
        tmp = rf.read()
        if kwargs is not None:
            for k, v in kwargs.items():
                tmp = tmp.replace('{{{}}}'.format(k), v)
        with open(filename, 'w') as wf:
            wf.write(tmp)
