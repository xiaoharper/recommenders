FROM continuumio/miniconda3

ENV PYTHONPATH /app

COPY ./reco_utils /app/reco_utils/
COPY ./scripts/generate_conda_file.py /app/generate_conda_file.py

ARG arg
RUN python /app/generate_conda_file.py $arg --name environment
#RUN conda env create -f environment.yaml --name reco
#RUN conda activate reco

#RUN echo "conda activate reco" > ~/.bashrc
# ENV PATH /opt/conda/envs/env/bin:$PATH
