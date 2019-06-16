FROM jupyter/scipy-notebook

USER root

ENV PYTHONPATH /app

COPY ./reco_utils /app/reco_utils/

RUN conda install --quiet --yes \
    "mock==2.0.0" \
    "gitpython>=2.1.8" \
    "numpy>=1.13.3" \
    "pytest>=3.6.4" \
    "scikit-surprise>=1.0.6"
    
RUN pip install \
    "idna==2.7" \
    "memory-profiler>=0.54.0" \
    "papermill>=0.15.0" \
    "tqdm==4.31.1"
    
# To use Conda environment w/ Kubernetes, maybe use "command: bash -c 'source activate reco && python ...'"
# ARG arg
# RUN python /app/generate_conda_file.py $arg --name environment
# RUN conda env create -f environment.yaml --name reco
# RUN echo "source activate reco" > ~/.bashrc
