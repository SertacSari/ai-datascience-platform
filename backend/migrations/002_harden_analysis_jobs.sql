BEGIN;

UPDATE analysis_jobs
SET config_json = '{}'::json
WHERE config_json IS NULL;

ALTER TABLE analysis_jobs
    ALTER COLUMN target_column SET NOT NULL,
    ALTER COLUMN config_json SET DEFAULT '{}'::json,
    ALTER COLUMN config_json SET NOT NULL;

ALTER TABLE analysis_jobs
    ADD CONSTRAINT ck_analysis_jobs_task_type
    CHECK (task_type IN ('classification', 'regression', 'forecasting'));

ALTER TABLE analysis_jobs
    ADD CONSTRAINT ck_analysis_jobs_status
    CHECK (status IN ('created', 'running', 'completed', 'failed'));

CREATE INDEX IF NOT EXISTS ix_analysis_jobs_user_created_id
    ON analysis_jobs (user_id, created_at DESC, id DESC);

COMMIT;
