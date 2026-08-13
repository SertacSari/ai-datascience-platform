BEGIN;

CREATE UNIQUE INDEX IF NOT EXISTS
    uq_ai_explanations_analysis_id
ON
    ai_explanations (analysis_id);

COMMIT;
