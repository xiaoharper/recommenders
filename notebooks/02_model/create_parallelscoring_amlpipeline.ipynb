{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import json\n",
    "import sys\n",
    "from datetime import datetime\n",
    "from azureml.core.compute import AmlCompute\n",
    "from azureml.core.datastore import Datastore\n",
    "from azureml.data.data_reference import DataReference\n",
    "from azureml.pipeline.core import Pipeline, PipelineData\n",
    "from azureml.pipeline.steps import PythonScriptStep\n",
    "from azureml.core.runconfig import CondaDependencies, RunConfiguration\n",
    "from azureml.core import Workspace, Run, Experiment\n",
    "from azureml.core.authentication import ServicePrincipalAuthentication\n",
    "from azureml.pipeline.core.schedule import ScheduleRecurrence, Schedule\n",
    "from azureml.core import Experiment\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pipeline_config = \"pipeline_config.json\"\n",
    "with open(pipeline_config) as f:\n",
    "    j = json.loads(f.read())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# SP authentication\n",
    "sp_auth = ServicePrincipalAuthentication(\n",
    "    tenant_id=j[\"sp_tenant\"], username=j[\"sp_client\"], password=j[\"sp_secret\"]\n",
    ")\n",
    "\n",
    "# AML workspace\n",
    "aml_ws = Workspace.get(\n",
    "    name=j[\"aml_work_space\"],\n",
    "    auth=sp_auth,\n",
    "    subscription_id=str(j[\"subscription_id\"]),\n",
    "    resource_group=j[\"resource_group_name\"],\n",
    ")\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Pipeline inputs, models, and outputs\n",
    "inputs_ds = Datastore.register_azure_blob_container(\n",
    "    aml_ws,\n",
    "    datastore_name=\"inputs_ds\",\n",
    "    container_name=j[\"data_blob_container\"],\n",
    "    account_name=j[\"blob_account\"],\n",
    "    account_key=j[\"blob_key\"],\n",
    ")\n",
    "inputs_dir = DataReference(datastore=inputs_ds, data_reference_name=\"inputs\")\n",
    "\n",
    "models_ds = Datastore.register_azure_blob_container(\n",
    "    aml_ws,\n",
    "    datastore_name=\"models_ds\",\n",
    "    container_name=j[\"models_blob_container\"],\n",
    "    account_name=j[\"blob_account\"],\n",
    "    account_key=j[\"blob_key\"],\n",
    ")\n",
    "models_dir = DataReference(datastore=models_ds, data_reference_name=\"models\")\n",
    "\n",
    "outputs_ds = Datastore.register_azure_blob_container(\n",
    "    aml_ws,\n",
    "    datastore_name=\"outputs_ds\",\n",
    "    container_name=j[\"preds_blob_container\"],\n",
    "    account_name=j[\"blob_account\"],\n",
    "    account_key=j[\"blob_key\"],\n",
    ")\n",
    "outputs_dir = PipelineData(name=\"outputs\", datastore=outputs_ds, is_directory=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Run config\n",
    "conda_dependencies = CondaDependencies.create(\n",
    "    pip_packages=j[\"pip_packages\"],\n",
    "    conda_packages=j[\"conda_packages\"],\n",
    "    python_version=j[\"python_version\"]\n",
    ")\n",
    "run_config = RunConfiguration(conda_dependencies=conda_dependencies)\n",
    "run_config.environment.docker.enabled = True"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "MOVIELENS_DATA_SIZE = '10m'\n",
    "\n",
    "if MOVIELENS_DATA_SIZE == '10m':\n",
    "    MAX_ALL = 72000\n",
    "    NUM_PER_RUN = 10000\n",
    "#    compute_target = AmlCompute(aml_ws, j[\"cluster_name\"])    \n",
    "    compute_target = AmlCompute(aml_ws, \"top10-mvl-d4v2\")    \n",
    "else:\n",
    "    MAX_ALL = 140000\n",
    "    NUM_PER_RUN = 10000\n",
    "    # getting memory errors...\n",
    "    compute_target = AmlCompute(aml_ws, \"top10-mvl-d4v2\")    \n",
    "\n",
    "# AML compute target\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a pipeline step for a subset of data...\n",
    "\n",
    "steps = []\n",
    "CUR_MIN = 1\n",
    "CUR_MAX = CUR_MIN + NUM_PER_RUN\n",
    "\n",
    "## will say for 10m\n",
    "## if have copied the reco_utils dir to this dir...\n",
    "while CUR_MIN < MAX_ALL:\n",
    "    outputs_dir = PipelineData(name=\"outputs\", datastore=outputs_ds, is_directory=True)\n",
    "    cur_name = \"{}_{}_{}\".format(CUR_MIN, CUR_MAX, MOVIELENS_DATA_SIZE)\n",
    "    print(cur_name)\n",
    "    step = PythonScriptStep(\n",
    "        name=cur_name,\n",
    "        script_name=j[\"python_script_name\"],\n",
    "        arguments=[CUR_MIN, CUR_MAX, inputs_dir, models_dir, outputs_dir, '10', MOVIELENS_DATA_SIZE],\n",
    "        inputs=[models_dir, inputs_dir],\n",
    "        outputs=[outputs_dir],\n",
    "        source_directory=j[\"python_script_directory\"],\n",
    "        compute_target=compute_target,\n",
    "        runconfig=run_config,\n",
    "        allow_reuse=False,\n",
    "    )\n",
    "    steps.append(step)\n",
    "    CUR_MIN = CUR_MAX\n",
    "    CUR_MAX = CUR_MIN + NUM_PER_RUN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pipeline = Pipeline(workspace=aml_ws, steps=steps)\n",
    "pipeline.validate()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "exp_name = 'reco_score_%s' %(MOVIELENS_DATA_SIZE)\n",
    "print(exp_name)\n",
    "pipeline_run = Experiment(aml_ws, exp_name).submit(pipeline)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from azureml.widgets import RunDetails\n",
    "RunDetails(pipeline_run).show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python (recommenders)",
   "language": "python",
   "name": "recommenders"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
