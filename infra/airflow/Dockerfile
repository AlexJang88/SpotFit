FROM apache/airflow:3.1.0-python3.12

ARG AIRFLOW_VERSION=3.1.0
COPY --chown=airflow:root requirements-server.txt /tmp/requirements-server.txt

RUN pip install --no-cache-dir \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-3.12.txt" \
    "apache-airflow==${AIRFLOW_VERSION}"

# Airflow's constraints apply to the Airflow install step only. Install project
# ML packages separately while pinning Airflow to prevent accidental upgrades.
RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    -r /tmp/requirements-server.txt
