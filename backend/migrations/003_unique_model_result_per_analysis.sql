BEGIN;

CREATE UNIQUE INDEX IF NOT EXISTS
    uq_model_results_analysis_id
ON
    model_results (analysis_id);

COMMIT;
