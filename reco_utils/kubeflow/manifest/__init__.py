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
    RANDOM = 'random'
    BAYESIAN = 'bayesianoptimization'


class WorkerType:
    WORKER = 'Job'
    TF_WORKER = 'TFJob'


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

UNIFORM_PARAM = """
    - name: {0}
      parametertype: double
      feasible:
        min: \"{1}\"
        max: \"{2}\""""

CHOICE_PARAM = """
    - name: {0}
      parametertype: categorical
      feasible:
        list: {1}"""

CHOICE_ITEM = """
        - \"{}\""""

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
                        - \"{}\""""

TF_WORKER_HYPERPARAM_PARSER = """
                        {{- with .HyperParameters}}
                        {{- range .}}
                        - \"{{.Name}}={{.Value}}\"
                        {{- end}}
                        {{- end}}"""

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

WORKER_HYPERPARAM_PARSER = """
                {{- with .HyperParameters}}
                {{- range .}}
                - \"{{.Name}}={{.Value}}\"
                {{- end}}
                {{- end}}"""

WORKER_GPU = """
                resources:
                  limits:
                    nvidia.com/gpu: 1"""
