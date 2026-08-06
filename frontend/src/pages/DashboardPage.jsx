import { useEffect, useState } from "react";
import {
  cleanDataset,
  createAnalysisJob,
  getCleaningReport,
  getDatasetPreview,
  getAnalysisJobResult,
  listAnalysisJobs,
  runAnalysisJob,
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

function formatPercentageMetric(value) {
  if (typeof value !== "number") return value ?? "Not available";
  return `${(value * 100).toFixed(1)}%`;
}

function formatNumberMetric(value) {
  if (typeof value !== "number") return value ?? "Not available";
  return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

const CLASSIFICATION_METRIC_EXPLANATIONS = {
  accuracy: "Overall share of correct predictions.",
  precision: "When the model predicts a class, how often it is correct.",
  recall: "How many real cases of a class the model catches.",
  f1_score: "Balance between precision and recall."
};

const REGRESSION_METRIC_EXPLANATIONS = {
  mae: "Average absolute prediction error in target units.",
  rmse: "Error metric that penalizes large mistakes more.",
  r2_score: "How much target variation the model explains on the test split."
};

const QUALITY_TONES = {
  good: "ok",
  fair: "warn",
  weak: "warn"
};

function isReportMetricRow(value) {
  return (
    value &&
    typeof value === "object" &&
    ("precision" in value || "recall" in value || "f1-score" in value || "support" in value)
  );
}

function matrixLabels(matrix, classDistribution, reportRows) {
  const distributionLabels = Object.keys(classDistribution || {});
  if (distributionLabels.length === matrix.length) return distributionLabels;

  const reportLabels = reportRows
    .map(([label]) => label)
    .filter((label) => !/accuracy|avg/i.test(label));
  if (reportLabels.length === matrix.length) return reportLabels;

  return matrix.map((_, index) => `Class ${index + 1}`);
}

function formatQualityLevel(level) {
  if (!level) return "Review";
  return `${level.charAt(0).toUpperCase()}${level.slice(1)}`;
}

function modelNameFromResult(modelResult) {
  return modelResult?.model_name || modelResult?.metrics?.model_name || modelResult?.report_json?.model_name || "Saved model";
}

function ResultInterpretation({ fallbackSummary, interpretation }) {
  return (
    <div className="result-interpretation">
      <div className="result-interpretation-head">
        <strong>Model result</strong>
        <Badge tone={QUALITY_TONES[interpretation.quality_level] || "neutral"}>
          {formatQualityLevel(interpretation.quality_level)}
        </Badge>
      </div>
      <p>{interpretation.summary || fallbackSummary}</p>
    </div>
  );
}

function ResultWarnings({ warnings }) {
  return (
    <div className={`result-warning-panel ${warnings.length ? "has-warnings" : "clear"}`}>
      <strong>Warnings</strong>
      {warnings.length ? (
        <ul>
          {warnings.map((warning, index) => (
            <li key={`${warning.code || "warning"}-${index}`}>
              <Badge tone="warn">Review</Badge>
              <span>{warning.message || warning.code || "Review this model result before using it."}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p>No major result warnings were returned.</p>
      )}
    </div>
  );
}

function MetricSummaryGrid({ rows }) {
  return (
    <div className={`metric-summary-grid count-${rows.length}`}>
      {rows.map(([key, label, value, explanation]) => (
        <div className="metric-summary-item" key={key}>
          <span>{label}</span>
          <strong>{value}</strong>
          <p>{explanation}</p>
        </div>
      ))}
    </div>
  );
}

function isRunnableTask(taskType) {
  return taskType === "classification" || taskType === "regression";
}

function trainingErrorMessage(error) {
  const message = error?.message || "";

  if (error?.status === 409 || /Only created jobs can be run/i.test(message)) {
    return "This job has already been completed. This job cannot be run again from this phase. Refresh jobs and choose a created classification or regression job.";
  }

  if (error?.status === 401) {
    return "Your session expired. Please log in again.";
  }

  return message || "Training failed. Check that the dataset and target column are suitable for this model type.";
}

function ClassificationResultCard({ result }) {
  const modelResult = result?.model_result || result;
  if (!modelResult) return null;

  const { metrics = {}, report_json: reportJson = {} } = modelResult;
  const interpretation = reportJson.interpretation || {};
  const metricExplanations = {
    ...CLASSIFICATION_METRIC_EXPLANATIONS,
    ...(interpretation.metric_explanations || {})
  };
  const warnings = Array.isArray(interpretation.warnings) ? interpretation.warnings : [];
  const classDistribution = metrics.class_distribution || {};
  const confusionMatrix = Array.isArray(reportJson.confusion_matrix) ? reportJson.confusion_matrix : [];
  const classificationReport = reportJson.classification_report || {};
  const reportRows = Object.entries(classificationReport).filter(([, value]) => isReportMetricRow(value));
  const labels = matrixLabels(confusionMatrix, classDistribution, reportRows);
  const metricRows = [
    ["accuracy", "Accuracy", metrics.accuracy],
    ["precision", "Precision", metrics.precision],
    ["recall", "Recall", metrics.recall],
    ["f1_score", "F1 score", metrics.f1_score]
  ];

  return (
    <section className="card model-result-card">
      <div className="card-head">
        <div>
          <p className="eyebrow">Classification result</p>
          <h2>{result?.job?.id ? `Job #${result.job.id} completed` : "Completed job result"}</h2>
        </div>
        <Badge tone="ok">{modelNameFromResult(modelResult)}</Badge>
      </div>

      <ResultInterpretation
        fallbackSummary="Classification training completed. Review the metrics below before using the result."
        interpretation={interpretation}
      />

      <MetricSummaryGrid
        rows={metricRows.map(([key, label, value]) => [
          key,
          label,
          formatPercentageMetric(value),
          metricExplanations[key] || CLASSIFICATION_METRIC_EXPLANATIONS[key]
        ])}
      />

      <ResultWarnings warnings={warnings} />

      <div className="result-block">
        <strong>Class distribution</strong>
        {Object.keys(classDistribution).length ? (
          <table className="result-table">
            <tbody>
              {Object.entries(classDistribution).map(([label, count]) => (
                <tr key={label}>
                  <th scope="row">{label}</th>
                  <td>{count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="muted">No class distribution was returned.</p>
        )}
      </div>
      <div className="result-block">
        <strong>Confusion matrix</strong>
        {confusionMatrix.length ? (
          <div className="table-wrap">
            <table className="result-table matrix-table">
              <thead>
                <tr>
                  <th scope="col">Actual \ Predicted</th>
                  {labels.map((label) => (
                    <th key={label} scope="col">{label}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {confusionMatrix.map((row, rowIndex) => (
                  <tr key={labels[rowIndex] || rowIndex}>
                    <th scope="row">{labels[rowIndex] || `Class ${rowIndex + 1}`}</th>
                    {row.map((value, columnIndex) => (
                      <td key={`${rowIndex}-${columnIndex}`}>{value}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">No confusion matrix was returned.</p>
        )}
      </div>
      <div className="result-block">
        <strong>Classification report</strong>
        {reportRows.length ? (
          <div className="table-wrap">
            <table className="result-table report-table">
              <thead>
                <tr>
                  <th scope="col">Class</th>
                  <th scope="col">Precision</th>
                  <th scope="col">Recall</th>
                  <th scope="col">F1</th>
                  <th scope="col">Support</th>
                </tr>
              </thead>
              <tbody>
                {reportRows.map(([label, values]) => (
                  <tr key={label}>
                    <th scope="row">{label}</th>
                    <td>{formatPercentageMetric(values.precision)}</td>
                    <td>{formatPercentageMetric(values.recall)}</td>
                    <td>{formatPercentageMetric(values["f1-score"])}</td>
                    <td>{values.support ?? "Not available"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <pre>{JSON.stringify(classificationReport, null, 2)}</pre>
        )}
      </div>
    </section>
  );
}

function RegressionResultCard({ result }) {
  const modelResult = result?.model_result || result;
  if (!modelResult) return null;

  const { metrics = {}, report_json: reportJson = {} } = modelResult;
  const interpretation = reportJson.interpretation || {};
  const metricExplanations = {
    ...REGRESSION_METRIC_EXPLANATIONS,
    ...(interpretation.metric_explanations || {})
  };
  const warnings = Array.isArray(interpretation.warnings) ? interpretation.warnings : [];
  const predictionSample = Array.isArray(reportJson.prediction_sample) ? reportJson.prediction_sample : [];
  const numericFeatures = Array.isArray(reportJson.numeric_features) ? reportJson.numeric_features : [];
  const categoricalFeatures = Array.isArray(reportJson.categorical_features) ? reportJson.categorical_features : [];
  const metricRows = [
    ["mae", "MAE", formatNumberMetric(metrics.mae), metricExplanations.mae],
    ["rmse", "RMSE", formatNumberMetric(metrics.rmse), metricExplanations.rmse],
    ["r2_score", "R² score", formatPercentageMetric(metrics.r2_score), metricExplanations.r2_score]
  ];

  return (
    <section className="card model-result-card">
      <div className="card-head">
        <div>
          <p className="eyebrow">Regression result</p>
          <h2>{result?.job?.id ? `Job #${result.job.id} completed` : "Completed job result"}</h2>
        </div>
        <Badge tone="ok">{modelNameFromResult(modelResult)}</Badge>
      </div>

      <ResultInterpretation
        fallbackSummary="Regression training completed. Review the prediction errors and sample rows before using the result."
        interpretation={interpretation}
      />

      <MetricSummaryGrid rows={metricRows} />

      <ResultWarnings warnings={warnings} />

      <div className="result-block">
        <strong>Target summary</strong>
        <table className="result-table">
          <tbody>
            <tr>
              <th scope="row">Target mean</th>
              <td>{formatNumberMetric(metrics.target_mean)}</td>
            </tr>
            <tr>
              <th scope="row">Target min</th>
              <td>{formatNumberMetric(metrics.target_min)}</td>
            </tr>
            <tr>
              <th scope="row">Target max</th>
              <td>{formatNumberMetric(metrics.target_max)}</td>
            </tr>
            <tr>
              <th scope="row">Test size</th>
              <td>{formatPercentageMetric(metrics.test_size)}</td>
            </tr>
          </tbody>
        </table>
      </div>

      <div className="result-block">
        <strong>Prediction sample</strong>
        {predictionSample.length ? (
          <div className="table-wrap">
            <table className="result-table prediction-table">
              <thead>
                <tr>
                  <th scope="col">Actual</th>
                  <th scope="col">Predicted</th>
                  <th scope="col">Difference</th>
                </tr>
              </thead>
              <tbody>
                {predictionSample.map((row, index) => {
                  const hasDifference = typeof row.actual === "number" && typeof row.predicted === "number";
                  const difference = hasDifference ? row.predicted - row.actual : null;

                  return (
                    <tr key={`${row.actual ?? "actual"}-${row.predicted ?? "predicted"}-${index}`}>
                      <td>{formatNumberMetric(row.actual)}</td>
                      <td>{formatNumberMetric(row.predicted)}</td>
                      <td>{hasDifference ? formatNumberMetric(difference) : "Not available"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="muted">No prediction sample was returned.</p>
        )}
      </div>

      <div className="result-block">
        <strong>Model features</strong>
        <p className="muted">
          Numeric: {numericFeatures.length ? numericFeatures.slice(0, 6).join(", ") : "Not available"}
        </p>
        <p className="muted">
          Categorical: {categoricalFeatures.length ? categoricalFeatures.slice(0, 6).join(", ") : "Not available"}
        </p>
      </div>
    </section>
  );
}

function ModelResultCard({ result }) {
  if (!result) return null;
  const taskType = result.job?.task_type;
  const modelResult = result.model_result || result;

  if (taskType === "regression" || "mae" in (modelResult.metrics || {})) {
    return <RegressionResultCard result={result} />;
  }

  return <ClassificationResultCard result={result} />;
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
  const [runStatus, setRunStatus] = useState({ jobId: null, type: "idle", message: "" });
  const [modelResult, setModelResult] = useState(null);
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

  async function handleRunJob(job) {
    setRunStatus({ jobId: job.id, type: "loading", message: `Running ${job.task_type} job #${job.id}...` });
    setModelResult(null);
    try {
      const result = await runAnalysisJob(job.id);
      const nextJobs = await listAnalysisJobs();
      setJobs(nextJobs);
      setModelResult(result);
      setRunStatus({
        jobId: job.id,
        type: "success",
        message: `${job.task_type.charAt(0).toUpperCase()}${job.task_type.slice(1)} job #${job.id} completed.`
      });
    } catch (error) {
      const nextJobs = await listAnalysisJobs().catch(() => jobs);
      setJobs(nextJobs);
      setRunStatus({
        jobId: job.id,
        type: "error",
        message: trainingErrorMessage(error)
      });
    }
  }

  async function handleViewResult(job) {
    setRunStatus({ jobId: job.id, type: "loading", message: `Loading saved result for job #${job.id}...` });
    try {
      const savedModelResult = await getAnalysisJobResult(job.id);
      setModelResult({ job, model_result: savedModelResult });
      setRunStatus({
        jobId: job.id,
        type: "success",
        message: `Loaded saved ${job.task_type} result for job #${job.id}.`
      });
    } catch (error) {
      setRunStatus({
        jobId: job.id,
        type: "error",
        message: error.message || "Could not load the saved result for this job."
      });
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
              ? "Create the job first; classification and regression jobs can be run here."
              : "Upload a dataset to enable backend analysis requests."}
          </p>
          <p className="muted">
            Classification and regression training are available. Forecasting training comes later.
          </p>
          {numericColumns.length ? <p className="muted">Numeric columns: {numericColumns.slice(0, 4).join(", ")}</p> : null}
          {jobStatus.message ? <div aria-live="polite" className={`backend-status ${jobStatus.type}`} role="status">{jobStatus.message}</div> : null}
          {runStatus.message ? <div aria-live="polite" className={`backend-status ${runStatus.type}`} role="status">{runStatus.message}</div> : null}
          {jobs.length ? (
            <div className="job-list">
              {jobs.slice(0, 4).map((job) => (
                <div key={job.id}>
                  <strong>#{job.id} {job.task_type}</strong>
                  <span>{job.target_column}</span>
                  <Badge tone={job.status === "failed" ? "err" : job.status === "completed" ? "ok" : "warn"}>{job.status}</Badge>
                  {isRunnableTask(job.task_type) && job.status === "created" ? (
                    <button
                      className="button sm"
                      disabled={runStatus.type === "loading"}
                      onClick={() => handleRunJob(job)}
                      type="button"
                    >
                      {runStatus.type === "loading" && runStatus.jobId === job.id ? "Running..." : "Run job"}
                    </button>
                  ) : null}
                  {isRunnableTask(job.task_type) && job.status === "completed" ? (
                    <button
                      className="button sm"
                      disabled={runStatus.type === "loading"}
                      onClick={() => handleViewResult(job)}
                      type="button"
                    >
                      {runStatus.type === "loading" && runStatus.jobId === job.id ? "Loading..." : "View result"}
                    </button>
                  ) : null}
                </div>
              ))}
            </div>
          ) : null}
        </section>

        <ModelResultCard result={modelResult} />
        <DatasetPreviewCard dataset={dataset} preview={preview} />
      </section>
    </main>
  );
}
