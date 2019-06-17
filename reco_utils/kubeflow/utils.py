# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import json
import math
import os
import requests
import subprocess
import time
import yaml
from reco_utils.kubeflow import manifest


JOB_DIR = "jobs"
RESULT_DIR = "results"


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

    return "_uniform", [min_val, max_val]


def _uniform(name, min_max):
    return manifest.UNIFORM_PARAM.format(name, min_max[0], min_max[1])


def choice(options):
    """Categorical hyperparameter spec

    Args:
        options (list)

    Returns:
        tuple: Hyperparameter spec
    """
    return "_choice", options


def _choice(name, list_val):
    items = "".join([manifest.CHOICE_ITEM.format(v) for v in list_val])
    return manifest.CHOICE_PARAM.format(name, items)


def _format_hyperparams(hyperparams):
    """Format hyperparameter spec by using `_uniform` and `_choice`"""
    return "".join([globals()[v[0]](k, v[1]) for k, v in hyperparams.items()])


def make_worker_spec(
    name,
    tag=None,
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
        name (str): Job name. If tag is provided, the name will be 'name-tag'
        tag (str): Job name tag
        worker_type (manifest.WorkerType): Type of the worker
        image_name (str): Name of the docker image
        entry_script (str): Path to the entry script in the image
        params (dict): Dictionary of script parameters.
            E.g. {'--datastore': "/data"}
        is_hypertune (bool): Whether the job is hyperparameter tuning or not
        storage_path (str): Mounted path of the persistent volume
        use_gpu (bool)

    Returns:
        str: Job name
        str: Worker spec yaml string
    """

    if tag is None:
        job_name = name
    else:
        job_name = name + "-{}".format(tag)

    script_params = {k: v for k, v in params.items()}
    script_params['--output-dir'] = job_name

    return job_name, _format_worker_spec(
        worker_type,
        image_name,
        entry_script,
        script_params,
        is_hypertune,
        storage_path,
        use_gpu,
    )


def make_hypertune_manifest(
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
        search_type (manifest.SearchType): Hyperparameter search algorithm
        total_runs (int): Number of total trials
        concurrent_runs (int): Number of concurrent runs
        primary_metric (str): Primary evaluation metric to optimize on
        goal (manifest.Goal): Whether minimize or maximize the primary metric
        ideal_metric_value (float): The best possible value of the primary metric
        metrics (list): List of evaluation metrics to track
        hyperparams (dict): Dictionary of hyperparameters.
            E.g. {'--learning-rate': uniform(0.001, 0.05)}
        worker_spec (tuple): (Job name, Worker spec)

    Returns:
        str: StudyJob name
        str: StudyJob filename
    """
    if not primary_metric:
        raise ValueError("Primary metric should be provided.")
    if total_runs < concurrent_runs:
        raise ValueError("Total runs should be equal or greater than concurrent runs.")

    os.makedirs(JOB_DIR, exist_ok=True)
    studyjob_name, worker = worker_spec
    studyjob_file = os.path.join(JOB_DIR, "{}.yaml".format(studyjob_name))

    _make_yaml_from_template(
        os.path.join(os.path.dirname(__file__), "manifest", "hypertune.template"),
        studyjob_file,
        **{
            "NAME": studyjob_name,
            "GOAL": goal,
            "PRIMARY_METRIC": primary_metric,
            "IDEAL_METRIC_VALUE": str(ideal_metric_value),
            "REQUEST_COUNT": str(math.ceil(total_runs / concurrent_runs)),
            "METRICS": _format_metrics(metrics),
            "HYPERPARAMS": _format_hyperparams(hyperparams),
            "WORKER": worker,
            "SPEC": _format_search_spec(search_type, concurrent_runs),
        }
    )

    print(
        "StudyJob manifest has been generated. To start, run 'kubectl create -f {}'".format(
            studyjob_file
        )
    )
    return studyjob_name, studyjob_file


def _format_search_spec(search_type, concurrent_runs):
    if search_type == manifest.SearchType.RANDOM:
        return manifest.RANDOM_SPEC.format(concurrent_runs)
    elif search_type == manifest.SearchType.BAYESIAN:
        # second param is 'burn-in'
        return manifest.BAYESIAN_SPEC.format(concurrent_runs, concurrent_runs)
    else:
        raise ValueError("Unknown search type {}.".format(search_type))


def _format_worker_spec(
    worker_type, image_name, entry_script, script_params, is_hypertune, storage_path, use_gpu
):
    if worker_type == manifest.WorkerType.WORKER:
        resources = manifest.WORKER_GPU if use_gpu else ""
        hyperparam_parser = manifest.WORKER_HYPERPARAM_PARSER if is_hypertune else ""
        params = [
            manifest.WORKER_PARAM.format("--study-id={{.StudyID}}"),
            manifest.WORKER_PARAM.format("--trial-id={{.TrialID}}")
        ]
        for k, v in script_params.items():
            if isinstance(v, (list, tuple, set)):
                params.append(manifest.WORKER_PARAM.format(k))
                params.extend([manifest.WORKER_PARAM.format(i) for i in v])
            elif isinstance(v, str) and len(v) == 0:
                params.append(manifest.WORKER_PARAM.format(k))
            else:
                params.append(manifest.WORKER_PARAM.format("{}={}".format(k, v)))

        return manifest.WORKER_TEMPLATE.format(
            image_name,
            entry_script,
            "".join(params) + hyperparam_parser,
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
    with open(template, "r") as rf:
        tmp = rf.read()
        if kwargs is not None:
            for k, v in kwargs.items():
                tmp = tmp.replace("{{{}}}".format(k), v)
        with open(filename, "w") as wf:
            wf.write(tmp)


def _connect():
    """Check port-forwarding and re-tunneling if connection lost"""
    return subprocess.Popen("kubectl port-forward svc/vizier-core-rest 6790:80", shell=True)


def get_study_metrics(study_id, worker_ids, metrics_names=None, timeout_sec=10.0):
    """Get metrics

    Args:
        study_id (str):
        worker_ids (list of str):
        metrics_names (list of str):
        timeout_sec (float): Request timeout in sec

    Returns:
        (json): metrics
    """
    data = {
        'study_id': study_id,
        'worker_ids': worker_ids,
        'metrics_names': metrics_names
    }
    json_data = json.dumps(data)

    while timeout_sec > 0.0:
        try:
            response = _post("http://localhost:6791/api/Manager/GetMetrics", json_data)
            return response  # ['metrics_log_sets'][0]['metrics_logs'][0]['values'][0]['value']
        except requests.ConnectionError:
            _connect()
            timeout_sec -= 0.5
            time.sleep(0.5)

    raise TimeoutError("Connection timeout")


def _post(url, json_data):
    resp = requests.post(
        url, data=json_data, headers={'Content-type': 'application/json'}
    )
    resp.raise_for_status()

    return resp.json()


def get_study_result(studyjob_name, verbose=True, write_result=True):
    # Note, Parameter configs keys are duplicated in the result. Should not use when parse the result.
    study_result = subprocess.run(
        ['kubectl', 'describe', 'studyjob', studyjob_name],
        stdout=subprocess.PIPE
    ).stdout.decode('utf-8')
    if verbose:
        print(study_result, "\n\n")

    study_result = study_result.split("Status:", 1)[1].split("Events:", 1)[0]

    try:
        r = yaml.safe_load(study_result)
        print("Study name:", studyjob_name)
        print("Study id:", r['Studyid'])
        print("Duration:", (r['Completion Time'] - r['Start Time']).total_seconds())
        print("Best trial:", r['Best Trial Id'])
        print("Best worker:", r['Best Worker Id'])
        print("Best object value:", r['Best Objective Value'])
    except KeyError:
        print("Study is still running or not completed")
        return

    # Cache (backup) results as yaml file for later use
    if write_result:
        os.makedirs(RESULT_DIR, exist_ok=True)

        result_filename = os.path.join(RESULT_DIR, "{}-result.yaml".format(studyjob_name))
        with open(result_filename, 'w') as f:
            f.write(study_result)

        print("Result is saved to {}".format(result_filename))

    return r
