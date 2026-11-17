FROM apache/airflow:3.1.0-python3.12

# Airflow orchestrates ingestion and readiness checks. ML training belongs in
# a separate worker image; the server's PyTorch/Transformers dependencies are
# intentionally not installed into the scheduler/webserver image.
