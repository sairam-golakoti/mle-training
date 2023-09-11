FROM conda/miniconda3:latest

WORKDIR /usr/src/app

COPY env1.yml .
COPY src/housing_price/. .

RUN conda update conda \
    && conda config --set restore_free_channel true
RUN conda env create -f env1.yml

ENTRYPOINT ["conda", "run", "-n", "mle-dev", "python"]

CMD ["ingest_data.py", "train.py", "score.py"]