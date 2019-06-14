FROM continuumio/miniconda3

ENV PYTHONPATH /app

COPY ./reco_utils /app/reco_utils/
COPY ./scripts/generate_conda_file.py /app/generate_conda_file.py

RUN apt-get update && \
    apt-get -y install gcc && \
    rm -rf /var/lib/apt/lists/*

ARG arg
RUN python /app/generate_conda_file.py $arg --name environment
RUN conda env create -f environment.yaml --name reco

RUN echo "source activate reco" > ~/.bashrc

