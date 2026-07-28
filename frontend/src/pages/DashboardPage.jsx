import { useEffect, useState } from "react";
import {
  cleanDataset,
  createAnalysisJob,
  getCleaningReport,
  getDatasetPreview,
  listAnalysisJobs,
  uploadDataset
} from "../api/client";
import AiSummaryCard from "../components/AiSummaryCard";
import Badge from "../components/Badge";
import Dropzone from "../components/Dropzone";
import LineChart from "../components/LineChart";
import Segmented from "../components/Segmented";
import StatCard from "../components/StatCard";
import { emptyDashboardData, useSession } from "../context/SessionContext";
import { statRows, technicalRows } from "../lib/dashboardSamples";

function numericColumnsFromPreview(preview) {
  return (preview?.column_info || [])
    .filter((column) => /int|float|double|decimal|number/i.test(column.dtype))
    .map((column) => column.name);
}

function dashboardStats(dataset, cleaning, jobs) {
  if (!dataset) return statRows;

  const missingTotal = cleaning
    ? Object.values(cleaning.missing_values || {}).reduce((sum, value) => sum + value, 0)
    : 0;
  const issueCount = cleaning?.issues?.length || 0;

  return [
    ["Rows analyzed", dataset.row_count.toLocaleString(), `${dataset.column_count} columns`, "ok"],
    ["Missing values", missingTotal.toLocaleString(), "cleaning report", missingTotal ? "warn" : "ok"],
    ["Quality issues", issueCount, cleaning?.ready_for_ml ? "ready for ML" : "review", issueCount ? "warn" : "ok"],
    ["Analysis jobs", jobs.length, jobs[0]?.status || "none", jobs.some((job) => job.status === "failed") ? "err" : "ok"]
  ];
}

function technicalFromBackend(cleaning, preview) {
  if (!cleaning) return technicalRows;

  const columnTypes = cleaning.column_types || {};
  const numerical = Object.values(columnTypes).filter((type) => type === "numerical").length;
  const categorical = Object.values(columnTypes).filter((type) => type === "categorical").length;

  return [
    ["Duplicate rows", cleaning.duplicate_rows, cleaning.duplicate_rows ? "Review" : "Normal", cleaning.duplicate_rows ? "warn" : "ok"],
    ["Numerical columns", numerical, numerical ? "Normal" : "Review", numerical ? "ok" : "warn"],
    ["Categorical columns", categorical, "Normal", "ok"],
    ["Preview rows", preview?.preview?.length || 0, preview ? "Loaded" : "Review", preview ? "ok" : "warn"]
  ];
}

function DatasetPreviewCard({ dataset, preview }) {
  if (!dataset || !preview) return null;

  const visibleColumns = preview.columns.slice(0, 6);
  const hiddenColumnCount = Math.max(0, preview.columns.length - visibleColumns.length);
  const rows = preview.preview.slice(0, 5);

  return (
    <section className="card preview-table-card">
      <div className="card-head">
        <div>
          <p className="eyebrow">Dataset preview</p>
          <h2>{preview.file_name || dataset.file_name}</h2>
        </div>
        <Badge tone="neutral">
          {preview.row_count.toLocaleString()} rows / {preview.column_count} columns
        </Badge>
      </div>
      <div className="table-wrap">
        <table className="reports-table dataset-preview-table">
          <thead>
            <tr>
              {visibleColumns.map((column) => (
                <th key={column}>{column}</th>
              ))}
              {hiddenColumnCount ? <th>+{hiddenColumnCount} more</th> : null}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {visibleColumns.map((column) => (
                  <td key={column}>{String(row[column] ?? "")}</td>
                ))}
                {hiddenColumnCount ? <td className="muted">Hidden</td> : null}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export default function DashboardPage() {
  const {
    dashboardData,
    jobs,
    setDashboardData,
    setJobs
  } = useSession();
  const [range, setRange] = useState("1M");
  const [upload, setUpload] = useState({ status: "idle", file: "", progress: 0, message: "" });
  const [jobForm, setJobForm] = useState({ task_type: "regression", target_column: "", date_column: "" });
  const [cleanStatus, setCleanStatus] = useState({ type: "idle", message: "" });
  const [jobStatus, setJobStatus] = useState({ type: "idle", message: "" });
  const { cleanResult, cleaning, dataset, preview } = dashboardData;
  const stats = dashboardStats(dataset, cleaning, jobs);
  const technical = technicalFromBackend(cleaning, preview);
  const numericColumns = numericColumnsFromPreview(preview);

  useEffect(() => {
    listAnalysisJobs()
      .then(setJobs)
      .catch(() => setJobs([]));
  }, [setJobs]);

  async function startUpload(file) {
    if (!file) return;

    setUpload({ status: "uploading", file: file.name, progress: 30, message: "" });
    setDashboardData(emptyDashboardData());
    setCleanStatus({ type: "idle", message: "" });
    setJobStatus({ type: "idle", message: "" });

    try {
      const uploaded = await uploadDataset(file);
      setUpload((current) => ({ ...current, progress: 70 }));
      const [nextPreview, nextCleaning, nextJobs] = await Promise.all([
        getDatasetPreview(uploaded.id),
        getCleaningReport(uploaded.id),
        listAnalysisJobs()
      ]);
      const firstNumeric = numericColumnsFromPreview(nextPreview)[0];
      const fallbackTarget = firstNumeric || nextPreview.columns[0] || "";
      const firstDateColumn = nextPreview.columns.find((column) => column !== fallbackTarget) || "";

      setDashboardData({
        cleanResult: null,
        cleaning: nextCleaning,
        dataset: uploaded,
        preview: nextPreview
      });
      setJobs(nextJobs);
      setJobForm({
        task_type: firstNumeric ? "regression" : "classification",
        target_column: fallbackTarget,
        date_column: firstDateColumn
      });
      setUpload({
        status: "done",
        file: uploaded.file_name,
        progress: 100,
        message: `Uploaded ${uploaded.row_count.toLocaleString()} rows / ${uploaded.column_count} columns`
      });
    } catch (error) {
      setUpload({
        status: "error",
        file: file.name,
        progress: 0,
        message: error.message || "Upload failed"
      });
    }
  }

  async function handleCleanDataset() {
    if (!dataset) return;
    setCleanStatus({ type: "loading", message: "Cleaning dataset..." });
    try {
      const result = await cleanDataset(dataset.id);
      setDashboardData((current) => ({ ...current, cleanResult: result }));
      setCleanStatus({
        type: "success",
        message: `${result.message}. Quality metrics still describe the original upload.`
      });
    } catch (error) {
      setCleanStatus({ type: "error", message: error.message || "Cleaning failed" });
    }
  }

  async function handleCreateJob(event) {
    event.preventDefault();
    if (!dataset || !jobForm.target_column) return;

    setJobStatus({ type: "loading", message: "Creating analysis job..." });
    try {
      const created = await createAnalysisJob(
        {
          dataset_id: dataset.id,
          task_type: jobForm.task_type,
          target_column: jobForm.target_column,
          config_json: jobForm.task_type === "forecasting" ? { date_column: jobForm.date_column } : {}
        }
      );
      const nextJobs = await listAnalysisJobs();
      setJobs(nextJobs);
      setJobStatus({
        type: "success",
        message: `Created ${created.task_type} job #${created.id} (${created.status})`
      });
    } catch (error) {
      setJobStatus({ type: "error", message: error.message || "Could not create analysis job" });
    }
  }

  return (
    <main className="page-shell">
      <section className="dashboard-grid">
        <Dropzone onUpload={startUpload} upload={upload} />
        <section className="card kpi-strip">
          {stats.map(([label, value, delta, tone]) => (
            <StatCard delta={delta} key={label} label={label} tone={tone} value={value} />
          ))}
        </section>

        <section className="card chart-card">
          <div className="card-head">
            <div>
              <p className="eyebrow">Dataset overview</p>
              <h2>Profile placeholder</h2>
            </div>
            <Segmented onChange={setRange} value={range} values={["1W", "1M", "3M", "1Y"]} />
          </div>
          <LineChart label={`Sample profile chart for ${range}`} range={range} />
          <div className="legend-row">
            <span><i className="legend-dot solid" />Sample</span>
            <span><i className="legend-dot muted" />Reference</span>
          </div>
        </section>

        <AiSummaryCard cleaning={cleaning} dataset={dataset} preview={preview} />

        <section className="card technical-card">
          <div className="card-head">
            <div>
              <p className="eyebrow">Technical analysis</p>
              <h2>{dataset ? dataset.file_name : "Signal review"}</h2>
            </div>
          </div>
          <table className="mini-table">
            <tbody>
              {technical.map(([metric, value, signal, tone]) => (
                <tr key={metric}>
                  <td>{metric}</td>
                  <td>{value}</td>
                  <td><Badge tone={tone}>{signal}</Badge></td>
                </tr>
              ))}
            </tbody>
          </table>
          {dataset ? (
            <div className="backend-actions">
              <button className="button" disabled={cleanStatus.type === "loading"} onClick={handleCleanDataset} type="button">
                {cleanStatus.type === "loading" ? "Cleaning..." : "Clean dataset"}
              </button>
              {cleanResult ? (
                <span>
                  {cleanResult.cleaned_row_count.toLocaleString()} cleaned rows · removed {cleanResult.removed_duplicate_rows} duplicates
                </span>
              ) : null}
            </div>
          ) : null}
          {cleanStatus.message ? (
            <div aria-live="polite" className={`backend-status ${cleanStatus.type}`} role="status">
              {cleanStatus.message}
            </div>
          ) : null}
        </section>

        <section className="card backend-card">
          <div className="card-head">
            <div>
              <p className="eyebrow">Analysis request</p>
              <h2>Create analysis request</h2>
            </div>
            <Badge tone={jobs.length ? "ok" : "neutral"}>{jobs.length} jobs</Badge>
          </div>
          <form className="analysis-form" onSubmit={handleCreateJob}>
            <label>
              <span>Analysis type</span>
              <select
                disabled={!dataset}
                onChange={(event) => {
                  const taskType = event.target.value;
                  setJobForm((current) => {
                    const dateColumn = taskType === "forecasting" && (!current.date_column || current.date_column === current.target_column)
                      ? (preview?.columns || []).find((column) => column !== current.target_column) || ""
                      : current.date_column;
                    return { ...current, date_column: dateColumn, task_type: taskType };
                  });
                }}
                value={jobForm.task_type}
              >
                <option value="classification">Classification</option>
                <option value="regression">Regression</option>
                <option value="forecasting">Forecasting</option>
              </select>
            </label>
            <label>
              <span>Target column</span>
              <select
                disabled={!dataset}
                onChange={(event) => {
                  const targetColumn = event.target.value;
                  setJobForm((current) => ({
                    ...current,
                    date_column: current.date_column === targetColumn
                      ? (preview?.columns || []).find((column) => column !== targetColumn) || ""
                      : current.date_column,
                    target_column: targetColumn
                  }));
                }}
                value={jobForm.target_column}
              >
                {(preview?.columns || []).map((column) => (
                  <option key={column} value={column}>{column}</option>
                ))}
              </select>
            </label>
            {jobForm.task_type === "forecasting" ? (
              <label>
                <span>Date column for forecasting</span>
                <select
                  disabled={!dataset}
                  onChange={(event) => setJobForm((current) => ({ ...current, date_column: event.target.value }))}
                  value={jobForm.date_column}
                >
                  {(preview?.columns || []).filter((column) => column !== jobForm.target_column).map((column) => (
                    <option key={column} value={column}>{column}</option>
                  ))}
                </select>
              </label>
            ) : null}
            <button className="button primary" disabled={!dataset || jobStatus.type === "loading"} type="submit">
              Create job
            </button>
          </form>
          <p className="muted">
            {dataset
              ? "This creates the job only; model training comes in the next phase."
              : "Upload a dataset to enable backend analysis requests."}
          </p>
          {numericColumns.length ? <p className="muted">Numeric columns: {numericColumns.slice(0, 4).join(", ")}</p> : null}
          {jobStatus.message ? <div aria-live="polite" className={`backend-status ${jobStatus.type}`} role="status">{jobStatus.message}</div> : null}
          {jobs.length ? (
            <div className="job-list">
              {jobs.slice(0, 4).map((job) => (
                <div key={job.id}>
                  <strong>#{job.id} {job.task_type}</strong>
                  <span>{job.target_column}</span>
                  <Badge tone={job.status === "failed" ? "err" : job.status === "completed" ? "ok" : "warn"}>{job.status}</Badge>
                </div>
              ))}
            </div>
          ) : null}
        </section>

        <DatasetPreviewCard dataset={dataset} preview={preview} />
      </section>
    </main>
  );
}
