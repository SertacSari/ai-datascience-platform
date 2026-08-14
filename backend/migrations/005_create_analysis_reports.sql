BEGIN;

CREATE TABLE IF NOT EXISTS analysis_reports (
    id SERIAL PRIMARY KEY,
    analysis_id INTEGER NOT NULL REFERENCES analysis_jobs(id),
    report_type VARCHAR(20) NOT NULL DEFAULT 'html',
    status VARCHAR(30) NOT NULL DEFAULT 'ready',
    file_name VARCHAR(255) NOT NULL,
    html_content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS
    uq_analysis_reports_analysis_id
ON
    analysis_reports (analysis_id);

CREATE INDEX IF NOT EXISTS
    ix_analysis_reports_id
ON
    analysis_reports (id);

COMMIT;
