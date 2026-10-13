FROM apache/airflow:3.1.0-python3.12

ARG AIRFLOW_VERSION=3.1.0
COPY --chown=airflow:root ml/requirements.txt /tmp/ml-requirements.txt

RUN pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.12.txt" \
    "apache-airflow==${AIRFLOW_VERSION}" \
    -r /tmp/ml-requirements.txt
