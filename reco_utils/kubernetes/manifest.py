# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
IMPORTANT!!!

DO NOT CHANGE THE FORMAT of the strings in this file NOR AUTO-FORMAT
"""


class Goal:
    MAXIMIZE = 'maximize'
    MINIMIZE = 'minimize'


class SearchType:
    RANDOM = 0
    BAYESIAN = 1


METRIC_ITEM = """
    - {}"""

RANDOM_SPEC = """
    suggestionAlgorithm: \"random\"
    requestNumber: {}"""

BAYESIAN_SPEC = """
    suggestionAlgorithm: \"bayesianoptimization\"
    suggestionParameters:
      -
          name: \"burn_in\"
          value: \"{1}\"
    requestNumber: {0}"""

RANGE_PARAM = """
    - name: {0}
      parametertype: {1}
      feasible:
        min: \"{2}\"
        max: \"{3}\""""

LIST_PARAM = """
    - name: {0}
      parametertype: categorical
      feasible:
        list: {1}"""

LIST_ITEM = """
        - \"{}\""""

# image, entry_script, param(TF_WORKER_PARAM, TF_WORKER_HYPERPARAM), volume_path="{{.StudyID}}", storage_path="/tmp/tensorflow", resources=""
TF_WORKER_TEMPLATE = """
        apiVersion: kubeflow.org/v1beta1
        kind: TFJob
        metadata:
          name: {{{{.WorkerID}}}}
          namespace: kubeflow
        spec:
          tfReplicaSpecs:
            Worker:
              replicas: 1
              restartPolicy: Never
              template:
                spec:
                  containers:
                    - name: tensorflow
                      image: {0}
                      imagePullPolicy: Always
                      command:
                        - \"python\"
                        - \"{1}\"{2}
                      volumeMounts:
                        - name: azurefile
                          subPath: {3}
                          mountPath: {4}{5}
                  volumes:
                    - name: azurefile
                      persistentVolumeClaim:
                        claimName: azurefile"""

TF_WORKER_GPU = """
                  resources:
                    limits:
                      nvidia.com/gpu: 1"""

TF_WORKER_PARAM = """
                        - \"{}\""""  # e.g. --log_dir=/train/{{.WorkerID}}

TF_WORKER_HYPERPARAM = """
                        {{{{- with .HyperParameters}}}}
                        {{{{- range .}}}}
                        - \"{{{{.Name}}}}={{{{.Value}}}}\"
                        {{{{- end}}}}
                        {{{{- end}}}}"""

# image, entry_script, param(WORKER_PARAM), volume_path="{{.StudyID}}", storage_path="/tmp/tensorflow", resources=""
WORKER_TEMPLATE = """
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: {{{{.WorkerID}}}}
          namespace: kubeflow
        spec:
          template:
            spec:
              containers:
              - name: {{{{.WorkerID}}}}
                image: {0}
                command:
                - \"python\"
                - \"{1}\"{2}
                volumeMounts:
                  - name: azurefile
                    subPath: {3}
                    mountPath: {4}{5}
              restartPolicy: Never
              volumes:
                - name: azurefile
                  persistentVolumeClaim:
                    claimName: azurefile"""

WORKER_PARAM = """
                - \"{}\""""

WORKER_HYPERPARAM = """
                {{{{- with .HyperParameters}}}}
                {{{{- range .}}}}
                - \"{{{{.Name}}}}={{{{.Value}}}}\"
                {{{{- end}}}}
                {{{{- end}}}}"""
