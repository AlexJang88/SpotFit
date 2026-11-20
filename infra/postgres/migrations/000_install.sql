-- Run with psql -X -v ON_ERROR_STOP=1 -f 000_install.sql against the SpotFit DB.
-- Do not point this file at the airflow/mlflow metadata databases.
\set ON_ERROR_STOP on
\encoding UTF8
BEGIN;
SET LOCAL TIME ZONE 'Asia/Seoul';
SET LOCAL lock_timeout = '15s';
SELECT pg_advisory_xact_lock(718261116);
CREATE SCHEMA IF NOT EXISTS spotfit;
CREATE TABLE IF NOT EXISTS spotfit.schema_migration (
    version text PRIMARY KEY,
    description text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now()
);
SELECT NOT EXISTS (SELECT 1 FROM spotfit.schema_migration WHERE version = '001_erd_v2') AS install_v1 \gset
\if :install_v1
\ir 001_schema.sql
INSERT INTO spotfit.schema_migration(version, description)
VALUES ('001_erd_v2', 'SpotFit ERD v2: 33 business tables, model type catalog, constraints');
\endif
\ir 002_reference_seed.sql
\ir 003_seoul_area_seed.sql
\ir 090_verify.sql
COMMIT;
\echo 'SpotFit schema and reference seed committed. No application DB adapter was changed.'
