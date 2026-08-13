import { useEffect, useState } from "react";
import {
  cleanDataset,
  createAnalysisJob,
  generateAiExplanation,
  getAiExplanation,
  getAnalysisRecommendation,
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
import StatCard from "../components/StatCard";
import { emptyDashboardData, useSession } from "../context/SessionContext";
import { statRows, technicalRows } from "../lib/dashboardSamples";

function numericColumnsFromPreview(preview) {
  return (preview?.column_info || [])
    .filter((column) => /int|float|double|decimal|number/i.test(column.dtype))
    .map((column) => column.name);
}

function isNumericPreviewColumn(preview, columnName) {
  return numericColumnsFromPreview(preview).includes(columnName);
}

function scoreTargetCandidate(columnName, dtype = "") {
  const normalized = String(columnName).toLowerCase();
  const normalizedType = String(dtype).toLowerCase();
  let score = 0;

  if (/(target|label|outcome|result|class)$/i.test(normalized)) score += 35;
  if (/(churn|churned|left|default|approved|fraud|status)$/i.test(normalized)) score += 32;
  if (/(sale_price|price|revenue|sales|amount|cost|value)$/i.test(normalized)) score += 30;
  if (/(score|rating|demand|quantity|total)$/i.test(normalized)) score += 18;
  if (/id$|^id$|uuid|index|row|code|zip|postal/i.test(normalized)) score -= 35;
  if (/int|float|double|decimal|number/i.test(normalizedType)) score += 6;

  return score;
}

function chooseDefaultTargetColumn(preview) {
  const columns = preview?.columns || [];
  const columnInfo = preview?.column_info || [];
  if (!columns.length) return "";

  const candidates = columns.map((name, index) => {
    const info = columnInfo.find((column) => column.name === name) || {};
    return {
      index,
      name,
      score: scoreTargetCandidate(name, info.dtype)
    };
  });

  const best = candidates.sort((a, b) => b.score - a.score || b.index - a.index)[0];
  if (best && best.score > 0) return best.name;

  return numericColumnsFromPreview(preview)[0] || columns[0] || "";
}

const VALID_TASK_TYPES = ["classification", "regression", "forecasting"];
const RECOMMENDATION_UNAVAILABLE_MESSAGE = "Recommendation is unavailable. You can still choose columns manually.";

function chooseDefaultDateColumn(preview, targetColumn) {
  return (preview?.columns || []).find((column) => column !== targetColumn) || "";
}

function defaultJobFormFromPreview(preview) {
  const targetColumn = chooseDefaultTargetColumn(preview);

  return {
    task_type: isNumericPreviewColumn(preview, targetColumn) ? "regression" : "classification",
    target_column: targetColumn,
    date_column: chooseDefaultDateColumn(preview, targetColumn)
  };
}

function jobFormFromRecommendation(recommendation, preview) {
  const fallback = defaultJobFormFromPreview(preview);
  const columns = preview?.columns || [];
  if (!recommendation || !columns.length) return fallback;

  const taskType = VALID_TASK_TYPES.includes(recommendation.recommended_task_type)
    ? recommendation.recommended_task_type
    : fallback.task_type;
  const targetColumn = columns.includes(recommendation.recommended_target_column)
    ? recommendation.recommended_target_column
    : fallback.target_column;
  const recommendedDateColumn = recommendation.recommended_date_column;
  const dateColumn = taskType === "forecasting" && columns.includes(recommendedDateColumn)
    ? recommendedDateColumn
    : taskType === "forecasting"
      ? chooseDefaultDateColumn(preview, targetColumn)
      : fallback.date_column;

  return {
    task_type: taskType,
    target_column: targetColumn,
    date_column: dateColumn === targetColumn ? chooseDefaultDateColumn(preview, targetColumn) : dateColumn
  };
}

function formatTaskLabel(taskType) {
  if (!taskType) return "Not available";
  return `${taskType.charAt(0).toUpperCase()}${taskType.slice(1)}`;
}

function recommendationText(item, fallback = "Review this recommendation before creating the job.") {
  if (typeof item === "string") return item;
  if (item && typeof item === "object") {
    return item.message || item.reason || item.code || fallback;
  }
  return fallback;
}

function recommendationTone(confidence) {
  if (confidence === "high") return "ok";
  if (confidence === "medium" || confidence === "low") return "warn";
  return "neutral";
}

function healthTone(score) {
  if (typeof score !== "number") return "neutral";
  if (score >= 70) return "ok";
  if (score >= 40) return "warn";
  return "err";
}

function severityTone(severity) {
  if (severity === "high") return "err";
  if (severity === "medium") return "warn";
  if (severity === "low") return "neutral";
  return "neutral";
}

function formatRoleLabel(role) {
  if (!role) return "Column";
  return role
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function compactGuidanceItems(items, limit) {
  return Array.isArray(items) ? items.slice(0, limit) : [];
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

function columnTypeCounts(preview) {
  return (preview?.column_info || []).reduce(
    (counts, column) => {
      if (/int|float|double|decimal|number/i.test(column.dtype)) counts.numeric += 1;
      else if (/date|time/i.test(column.dtype)) counts.datetime += 1;
      else counts.categorical += 1;
      return counts;
    },
    { categorical: 0, datetime: 0, numeric: 0 }
  );
}

function topMissingColumns(cleaning, preview) {
  const missingValues = cleaning?.missing_values || {};
  const fromCleaning = Object.entries(missingValues)
    .map(([name, count]) => ({ count, name }))
    .filter((column) => column.count > 0);

  const fromPreview = (preview?.column_info || [])
    .map((column) => ({ count: column.missing_count, name: column.name }))
    .filter((column) => column.count > 0);

  return (fromCleaning.length ? fromCleaning : fromPreview)
    .sort((a, b) => b.count - a.count)
    .slice(0, 4);
}

function prioritizedNumericColumns(preview, targetColumn) {
  const numericColumns = numericColumnsFromPreview(preview);
  if (!numericColumns.length) return [];

  if (targetColumn && numericColumns.includes(targetColumn)) {
    return [targetColumn, ...numericColumns.filter((column) => column !== targetColumn)];
  }

  return numericColumns;
}

function numericSummaryRows(preview, targetColumn) {
  const statistics = preview?.summary_statistics || {};

  return prioritizedNumericColumns(preview, targetColumn)
    .map((columnName) => {
      const stats = statistics[columnName] || {};
      return {
        max: stats.max,
        mean: stats.mean,
        min: stats.min,
        name: columnName
      };
    })
    .filter((column) => [column.min, column.mean, column.max].some((value) => typeof value === "number"))
    .slice(0, 3);
}

function previewDistribution(preview, targetColumn) {
  const numericColumn = isNumericPreviewColumn(preview, targetColumn)
    ? targetColumn
    : numericColumnsFromPreview(preview)[0];
  if (!numericColumn) return null;

  const values = (preview?.preview || [])
    .map((row) => Number(row[numericColumn]))
    .filter((value) => Number.isFinite(value));
  if (!values.length) return null;

  const max = Math.max(...values);
  const min = Math.min(...values);
  const range = max - min || 1;

  return {
    column: numericColumn,
    values: values.slice(0, 10).map((value) => ({
      height: Math.max(8, Math.round(((value - min) / range) * 58) + 8),
      value
    }))
  };
}

function DatasetInsightsCard({ cleaning, dataset, preview, targetColumn }) {
  const typeCounts = columnTypeCounts(preview);
  const missingColumns = topMissingColumns(cleaning, preview);
  const numericRows = numericSummaryRows(preview, targetColumn);
  const distribution = previewDistribution(preview, targetColumn);
  const duplicateRows = cleaning?.duplicate_rows ?? preview?.duplicate_rows ?? 0;
  const targetIsNumeric = isNumericPreviewColumn(preview, targetColumn);

  return (
    <section className="card insights-card">
      <div className="card-head">
        <div>
          <p className="eyebrow">Dataset insights</p>
          <h2>{dataset ? "Target-aware profile" : "Waiting for dataset"}</h2>
        </div>
        <Badge tone={dataset ? duplicateRows ? "warn" : "ok" : "neutral"}>
          {dataset ? `${dataset.column_count} columns` : "No data"}
        </Badge>
      </div>

      {dataset && preview ? (
        <>
          <div className="insight-summary-grid">
            <div>
              <span>Rows</span>
              <strong>{dataset.row_count.toLocaleString()}</strong>
              <small>Uploaded dataset</small>
            </div>
            <div>
              <span>Duplicates</span>
              <strong>{duplicateRows.toLocaleString()}</strong>
              <small>{duplicateRows ? "Review before modeling" : "No duplicate rows found"}</small>
            </div>
            <div>
              <span>Selected target</span>
              <strong>{targetColumn || "Not selected"}</strong>
              <small>{targetIsNumeric ? "Numeric target" : "Choose target in analysis request"}</small>
            </div>
            <div>
              <span>Column mix</span>
              <strong>{typeCounts.numeric} num / {typeCounts.categorical} cat</strong>
              <small>{typeCounts.datetime ? `${typeCounts.datetime} date-like` : "Based on backend dtypes"}</small>
            </div>
          </div>

          <div className="insight-split">
            <div className="insight-panel">
              <strong>Missing values by column</strong>
              {missingColumns.length ? (
                <div className="insight-bars">
                  {missingColumns.map((column) => {
                    const width = Math.max(8, Math.round((column.count / dataset.row_count) * 100));
                    return (
                      <div className="insight-bar-row" key={column.name}>
                        <span title={column.name}>{column.name}</span>
                        <div><i style={{ width: `${Math.min(width, 100)}%` }} /></div>
                        <em>{column.count.toLocaleString()}</em>
                      </div>
                    );
                  })}
                </div>
              ) : (
                <p className="muted">No missing values were returned for the uploaded dataset.</p>
              )}
            </div>

            <div className="insight-panel">
              <strong>Preview distribution</strong>
              {distribution ? (
                <>
                  <div className="mini-distribution" aria-label={`Preview distribution for ${distribution.column}`}>
                    {distribution.values.map((item, index) => (
                      <i key={`${item.value}-${index}`} style={{ height: `${item.height}px` }} title={formatNumberMetric(item.value)} />
                    ))}
                  </div>
                  <p className="muted">
                    {distribution.column}{distribution.column === targetColumn ? " target" : ""} from visible preview rows
                  </p>
                </>
              ) : (
                <p className="muted">Upload data with numeric preview values to show a small distribution.</p>
              )}
            </div>
          </div>

          <div className="numeric-summary-list">
            <strong>Important numeric columns</strong>
            {numericRows.length ? (
              numericRows.map((column) => (
                <div key={column.name}>
                  <span title={column.name}>{column.name}</span>
                  <small>Min {formatNumberMetric(column.min)}</small>
                  <small>Mean {formatNumberMetric(column.mean)}</small>
                  <small>Max {formatNumberMetric(column.max)}</small>
                </div>
              ))
            ) : (
              <p className="muted">No numeric summary statistics were returned.</p>
            )}
          </div>
        </>
      ) : (
        <div className="empty-insights">
          <strong>No dataset loaded</strong>
          <p>Upload a CSV or Excel file to see real column mix, missing values, duplicate rows, and preview-based numeric signals.</p>
        </div>
      )}
    </section>
  );
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

function formatPercentValue(value) {
  if (typeof value !== "number") return value ?? "Not available";
  return `${value.toFixed(1)}%`;
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

const FORECASTING_METRIC_EXPLANATIONS = {
  mae: "Average absolute forecast error in target units.",
  rmse: "Typical forecast error, with larger mistakes weighted more heavily.",
  mape: "Average percentage forecast error when actual values are non-zero.",
  r2_score: "How much target variation the model explains on the held-out time period."
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

function recommendedActionText(action) {
  if (typeof action === "string") return action;
  if (action && typeof action === "object") {
    return action.message || action.code || "Review this model result before using it.";
  }
  return "Review this model result before using it.";
}

function ResultRecommendedActions({ actions }) {
  const safeActions = Array.isArray(actions) ? actions : [];

  return (
    <div className={`result-warning-panel ${safeActions.length ? "has-warnings" : "clear"}`}>
      <strong>Recommended actions</strong>
      {safeActions.length ? (
        <ul>
          {safeActions.map((action, index) => (
            <li key={`${recommendedActionText(action)}-${index}`}>
              <Badge>Action</Badge>
              <span>{recommendedActionText(action)}</span>
            </li>
          ))}
        </ul>
      ) : (
        <p>No extra actions were recommended.</p>
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
  return taskType === "classification" || taskType === "regression" || taskType === "forecasting";
}

function trainingErrorMessage(error) {
  const message = error?.message || "";

  if (error?.status === 409 || /Only created jobs can be run/i.test(message)) {
    return "This job has already been completed. This job cannot be run again from this phase. Refresh jobs and choose a created model job.";
  }

  if (error?.status === 401) {
    return "Your session expired. Please log in again.";
  }

  return message || "Training failed. Check that the dataset and target column are suitable for this model type.";
}

function aiExplanationErrorMessage(error) {
  if (error?.status === 503) {
    return "AI explanation is currently unavailable. The rule-based summary is still available.";
  }

  if (error?.status === 401) {
    return "Your session expired. Please log in again.";
  }

  return error?.message || "Could not load the local AI explanation.";
}

function LocalAiExplanationSection({ aiExplanationState, jobId, onGenerate }) {
  if (!jobId) return null;

  const isCurrentJob = aiExplanationState?.jobId === jobId;
  const status = isCurrentJob ? aiExplanationState.status : "idle";
  const explanation = isCurrentJob ? aiExplanationState.explanation : null;
  const message = isCurrentJob ? aiExplanationState.message : "";
  const isLoading = status === "loading";

  return (
    <div className={`result-warning-panel ${status === "error" ? "has-warnings" : "clear"}`}>
      <div className="result-interpretation-head">
        <strong>Local AI explanation</strong>
        <span className="badge neutral llm-model-badge">{explanation?.llm_model || "Optional"}</span>
      </div>
      <p>This optional explanation is generated locally from the saved model result.</p>
      {explanation?.explanation_text ? (
        <p>{explanation.explanation_text}</p>
      ) : (
        <button className="button sm" disabled={isLoading} onClick={() => onGenerate(jobId)} type="button">
          {isLoading ? "Working..." : "Generate explanation"}
        </button>
      )}
      {message ? <p>{message}</p> : null}
    </div>
  );
}

function ClassificationResultCard({ aiExplanationState, onGenerateAiExplanation, result }) {
  const modelResult = result?.model_result || result;
  if (!modelResult) return null;

  const { metrics = {}, report_json: reportJson = {} } = modelResult;
  const interpretation = reportJson.interpretation || {};
  const metricExplanations = {
    ...CLASSIFICATION_METRIC_EXPLANATIONS,
    ...(interpretation.metric_explanations || {})
  };
  const warnings = Array.isArray(interpretation.warnings) ? interpretation.warnings : [];
  const recommendedActions = Array.isArray(interpretation.recommended_actions)
    ? interpretation.recommended_actions
    : [];
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

      <ResultRecommendedActions actions={recommendedActions} />

      <LocalAiExplanationSection
        aiExplanationState={aiExplanationState}
        jobId={result?.job?.id}
        onGenerate={onGenerateAiExplanation}
      />

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

function RegressionResultCard({ aiExplanationState, onGenerateAiExplanation, result }) {
  const modelResult = result?.model_result || result;
  if (!modelResult) return null;

  const { metrics = {}, report_json: reportJson = {} } = modelResult;
  const interpretation = reportJson.interpretation || {};
  const metricExplanations = {
    ...REGRESSION_METRIC_EXPLANATIONS,
    ...(interpretation.metric_explanations || {})
  };
  const warnings = Array.isArray(interpretation.warnings) ? interpretation.warnings : [];
  const recommendedActions = Array.isArray(interpretation.recommended_actions)
    ? interpretation.recommended_actions
    : [];
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

      <ResultRecommendedActions actions={recommendedActions} />

      <LocalAiExplanationSection
        aiExplanationState={aiExplanationState}
        jobId={result?.job?.id}
        onGenerate={onGenerateAiExplanation}
      />

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

function ForecastingResultCard({ aiExplanationState, onGenerateAiExplanation, result }) {
  const modelResult = result?.model_result || result;
  if (!modelResult) return null;

  const { metrics = {}, report_json: reportJson = {} } = modelResult;
  const interpretation = reportJson.interpretation || {};
  const metricExplanations = {
    ...FORECASTING_METRIC_EXPLANATIONS,
    ...(interpretation.metric_explanations || {})
  };
  const warnings = Array.isArray(interpretation.warnings) ? interpretation.warnings : [];
  const recommendedActions = Array.isArray(interpretation.recommended_actions)
    ? interpretation.recommended_actions
    : [];
  const predictionSample = Array.isArray(reportJson.prediction_sample) ? reportJson.prediction_sample : [];
  const numericFeatures = Array.isArray(reportJson.numeric_features) ? reportJson.numeric_features : [];
  const categoricalFeatures = Array.isArray(reportJson.categorical_features) ? reportJson.categorical_features : [];
  const metricRows = [
    ["mae", "MAE", formatNumberMetric(metrics.mae), metricExplanations.mae],
    ["rmse", "RMSE", formatNumberMetric(metrics.rmse), metricExplanations.rmse],
    ["mape", "MAPE", formatPercentValue(metrics.mape), metricExplanations.mape],
    ["r2_score", "R² score", formatPercentageMetric(metrics.r2_score), metricExplanations.r2_score]
  ];

  return (
    <section className="card model-result-card">
      <div className="card-head">
        <div>
          <p className="eyebrow">Forecasting result</p>
          <h2>{result?.job?.id ? `Job #${result.job.id} completed` : "Completed job result"}</h2>
        </div>
        <Badge tone="ok">{modelNameFromResult(modelResult)}</Badge>
      </div>

      <ResultInterpretation
        fallbackSummary="Forecasting training completed. Review the held-out time period and forecast errors before using the result."
        interpretation={interpretation}
      />

      <MetricSummaryGrid rows={metricRows} />

      <ResultWarnings warnings={warnings} />

      <ResultRecommendedActions actions={recommendedActions} />

      <LocalAiExplanationSection
        aiExplanationState={aiExplanationState}
        jobId={result?.job?.id}
        onGenerate={onGenerateAiExplanation}
      />

      <div className="result-block">
        <strong>Forecasting context</strong>
        <table className="result-table">
          <tbody>
            <tr>
              <th scope="row">Date column</th>
              <td>{reportJson.date_column || "Not available"}</td>
            </tr>
            <tr>
              <th scope="row">Target column</th>
              <td>{reportJson.target_column || result?.job?.target_column || "Not available"}</td>
            </tr>
            <tr>
              <th scope="row">Test start date</th>
              <td>{reportJson.test_start_date || "Not available"}</td>
            </tr>
            <tr>
              <th scope="row">Test end date</th>
              <td>{reportJson.test_end_date || "Not available"}</td>
            </tr>
            <tr>
              <th scope="row">Model name</th>
              <td>{modelNameFromResult(modelResult)}</td>
            </tr>
          </tbody>
        </table>
      </div>

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
                  <th scope="col">Date</th>
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
                    <tr key={`${row.date || "date"}-${row.actual ?? "actual"}-${row.predicted ?? "predicted"}-${index}`}>
                      <td>{row.date || "Not available"}</td>
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

function ModelResultCard({ aiExplanationState, onGenerateAiExplanation, result }) {
  if (!result) return null;
  const taskType = result.job?.task_type;
  const modelResult = result.model_result || result;
  const reportJson = modelResult.report_json || {};

  if (taskType === "forecasting" || "mape" in (modelResult.metrics || {}) || "date_column" in reportJson) {
    return (
      <ForecastingResultCard
        aiExplanationState={aiExplanationState}
        onGenerateAiExplanation={onGenerateAiExplanation}
        result={result}
      />
    );
  }

  if (taskType === "regression" || "mae" in (modelResult.metrics || {})) {
    return (
      <RegressionResultCard
        aiExplanationState={aiExplanationState}
        onGenerateAiExplanation={onGenerateAiExplanation}
        result={result}
      />
    );
  }

  return (
    <ClassificationResultCard
      aiExplanationState={aiExplanationState}
      onGenerateAiExplanation={onGenerateAiExplanation}
      result={result}
    />
  );
}

function AnalysisDatasetSource({ cleanResult, dataset }) {
  if (!dataset) return null;

  return (
    <div className="result-warning-panel clear">
      <strong>Dataset used for analysis</strong>
      <p>{dataset.file_name || "Uploaded dataset"}</p>
      <p>Source: {cleanResult ? "Cleaned dataset" : "Original uploaded file"}</p>
      <p>
        {cleanResult
          ? "Future analysis jobs will use the cleaned version."
          : "If you clean the dataset, future analysis jobs will use the cleaned version."}
      </p>
    </div>
  );
}

function RecommendationTargetExplanation({ explanation, recommendedTarget }) {
  const strengths = Array.isArray(explanation?.strengths) ? explanation.strengths : [];
  const risks = Array.isArray(explanation?.risks) ? explanation.risks : [];

  return (
    <div className="target-explanation-panel">
      <strong>Why this target</strong>
      <p>{explanation?.message || `${recommendedTarget || "The selected target"} is the recommended target for this setup.`}</p>
      <div className="target-explanation-grid">
        <div>
          <span>Strengths</span>
          {strengths.length ? (
            <ul>
              {strengths.slice(0, 3).map((strength, index) => (
                <li key={`${strength}-${index}`}>{strength}</li>
              ))}
            </ul>
          ) : (
            <p className="muted">No target strengths were returned.</p>
          )}
        </div>
        <div>
          <span>Risks</span>
          {risks.length ? (
            <ul>
              {risks.slice(0, 3).map((risk, index) => (
                <li key={`${risk}-${index}`}>{risk}</li>
              ))}
            </ul>
          ) : (
            <p className="muted">No major target risks were found.</p>
          )}
        </div>
      </div>
    </div>
  );
}

function RecommendationGuidanceList({ emptyText, items, quiet = false, title }) {
  return (
    <div className={`column-guidance-list ${quiet ? "quiet" : ""}`}>
      <strong>{title}</strong>
      {items.length ? (
        <ul>
          {items.map((item, index) => (
            <li key={`${item.column || item.role || "column"}-${index}`}>
              <span>
                <b title={item.column}>{item.column || "Column"}</b>
                <small>{item.message || "Review this column before creating the analysis job."}</small>
              </span>
              <Badge tone={severityTone(item.severity)}>{formatRoleLabel(item.role)}</Badge>
            </li>
          ))}
        </ul>
      ) : (
        <p className="muted">{emptyText}</p>
      )}
    </div>
  );
}

function AnalysisRecommendationCard({ onUseRecommendation, recommendationState }) {
  const { message, recommendation, status } = recommendationState;
  if (status === "idle" && !recommendation) return null;

  const reasons = Array.isArray(recommendation?.reasons) ? recommendation.reasons : [];
  const warnings = Array.isArray(recommendation?.warnings) ? recommendation.warnings : [];
  const alternatives = Array.isArray(recommendation?.alternatives) ? recommendation.alternatives : [];
  const columnGuidance = Array.isArray(recommendation?.column_guidance) ? recommendation.column_guidance : [];
  const columnCautions = compactGuidanceItems(
    columnGuidance.filter((item) => ["high", "medium"].includes(item.severity) && ![
      "recommended_target",
      "useful_feature",
      "possible_date_column"
    ].includes(item.role)),
    4
  );
  const usefulFeatures = compactGuidanceItems(
    columnGuidance.filter((item) => item.role === "useful_feature"),
    4
  );
  const dateCandidates = compactGuidanceItems(
    columnGuidance.filter((item) => item.role === "possible_date_column"),
    2
  );
  const hasRecommendation = Boolean(recommendation);
  const confidence = recommendation?.confidence || "manual";
  const healthScore = recommendation?.health_score;

  return (
    <div className={`recommendation-panel ${status === "error" ? "unavailable" : ""}`}>
      <div className="section-kicker">
        <strong>Setup guidance</strong>
        <Badge tone={hasRecommendation ? recommendationTone(confidence) : "neutral"}>
          {status === "loading" ? "Checking" : confidence}
        </Badge>
      </div>

      {status === "loading" ? (
        <p className="muted">Checking the uploaded dataset for a recommended analysis setup...</p>
      ) : hasRecommendation ? (
        <>
          <div className="recommendation-summary">
            <div>
              <span>Recommended</span>
              <strong>{formatTaskLabel(recommendation.recommended_task_type)}</strong>
            </div>
            <div>
              <span>Target</span>
              <strong>{recommendation.recommended_target_column || "Not available"}</strong>
            </div>
            {recommendation.recommended_date_column ? (
              <div>
                <span>Date</span>
                <strong>{recommendation.recommended_date_column}</strong>
              </div>
            ) : null}
            <div>
              <span>Health score</span>
              <strong>
                <Badge tone={healthTone(healthScore)}>
                  {typeof healthScore === "number" ? `${healthScore}/100` : "Review"}
                </Badge>
              </strong>
            </div>
          </div>

          <RecommendationTargetExplanation
            explanation={recommendation.target_explanation}
            recommendedTarget={recommendation.recommended_target_column}
          />

          <button className="button sm" onClick={onUseRecommendation} type="button">
            Use recommendation
          </button>

          <div className="recommendation-detail-grid">
            <div>
              <strong>Why this fits</strong>
              {reasons.length ? (
                <ul>
                  {reasons.map((reason, index) => (
                    <li key={`${recommendationText(reason)}-${index}`}>{recommendationText(reason)}</li>
                  ))}
                </ul>
              ) : (
                <p className="muted">No detailed reasons were returned.</p>
              )}
            </div>
            <div>
              <strong>{confidence === "low" ? "Review before running" : "Warnings"}</strong>
              {warnings.length ? (
                <ul>
                  {warnings.map((warning, index) => (
                    <li key={`${recommendationText(warning)}-${index}`}>{recommendationText(warning)}</li>
                  ))}
                </ul>
              ) : (
                <p className="muted">No setup warnings were returned.</p>
              )}
            </div>
          </div>

          <div className="column-guidance-grid">
            <RecommendationGuidanceList
              emptyText="No major column cautions were returned."
              items={columnCautions}
              title="Column cautions"
            />
            <RecommendationGuidanceList
              emptyText="No useful feature guidance was returned."
              items={usefulFeatures}
              quiet
              title="Useful features"
            />
            {recommendation.recommended_task_type === "forecasting" || dateCandidates.length ? (
              <RecommendationGuidanceList
                emptyText="No date candidates were returned."
                items={dateCandidates}
                quiet
                title="Date candidates"
              />
            ) : null}
          </div>

          {alternatives.length ? (
            <div className="recommendation-alternatives">
              <strong>Alternatives</strong>
              {alternatives.slice(0, 3).map((alternative, index) => (
                <p key={`${alternative.task_type || "alternative"}-${alternative.target_column || index}`}>
                  {formatTaskLabel(alternative.task_type)} · Target: {alternative.target_column || "Not available"}
                  {alternative.date_column ? ` · Date: ${alternative.date_column}` : ""}
                  {alternative.reason ? ` · ${alternative.reason}` : ""}
                </p>
              ))}
            </div>
          ) : null}
        </>
      ) : (
        <p className="muted">{message || RECOMMENDATION_UNAVAILABLE_MESSAGE}</p>
      )}
    </div>
  );
}

export default function DashboardPage() {
  const {
    dashboardData,
    jobs,
    setDashboardData,
    setJobs
  } = useSession();
  const [upload, setUpload] = useState({ status: "idle", file: "", progress: 0, message: "" });
  const [jobForm, setJobForm] = useState({ task_type: "regression", target_column: "", date_column: "" });
  const [cleanStatus, setCleanStatus] = useState({ type: "idle", message: "" });
  const [jobStatus, setJobStatus] = useState({ type: "idle", message: "" });
  const [runStatus, setRunStatus] = useState({ jobId: null, type: "idle", message: "" });
  const [modelResult, setModelResult] = useState(null);
  const [recommendationState, setRecommendationState] = useState({
    message: "",
    recommendation: null,
    status: "idle"
  });
  const [aiExplanationState, setAiExplanationState] = useState({
    explanation: null,
    jobId: null,
    message: "",
    status: "idle"
  });
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
    setRunStatus({ jobId: null, type: "idle", message: "" });
    setModelResult(null);
    setRecommendationState({ message: "", recommendation: null, status: "idle" });
    setAiExplanationState({ explanation: null, jobId: null, message: "", status: "idle" });

    try {
      const uploaded = await uploadDataset(file);
      setUpload((current) => ({ ...current, progress: 70 }));
      setRecommendationState({ message: "", recommendation: null, status: "loading" });
      const recommendationRequest = getAnalysisRecommendation(uploaded.id)
        .then((recommendation) => ({ recommendation }))
        .catch((error) => ({ error }));
      const [nextPreview, nextCleaning, nextJobs, recommendationResult] = await Promise.all([
        getDatasetPreview(uploaded.id),
        getCleaningReport(uploaded.id),
        listAnalysisJobs(),
        recommendationRequest
      ]);
      const nextRecommendation = recommendationResult.recommendation || null;

      setDashboardData({
        cleanResult: null,
        cleaning: nextCleaning,
        dataset: uploaded,
        preview: nextPreview
      });
      setJobs(nextJobs);
      setJobForm(nextRecommendation
        ? jobFormFromRecommendation(nextRecommendation, nextPreview)
        : defaultJobFormFromPreview(nextPreview));
      setRecommendationState(recommendationResult.error
        ? {
          message: RECOMMENDATION_UNAVAILABLE_MESSAGE,
          recommendation: null,
          status: "error"
        }
        : {
          message: "",
          recommendation: nextRecommendation,
          status: "success"
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
      setRecommendationState({ message: "", recommendation: null, status: "idle" });
    }
  }

  function handleUseRecommendation() {
    if (!recommendationState.recommendation || !preview) return;
    setJobForm(jobFormFromRecommendation(recommendationState.recommendation, preview));
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
    setAiExplanationState({ explanation: null, jobId: job.id, message: "", status: "idle" });
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
    setAiExplanationState({
      explanation: null,
      jobId: job.id,
      message: "Checking for saved local AI explanation...",
      status: "loading"
    });
    try {
      const savedModelResult = await getAnalysisJobResult(job.id);
      setModelResult({ job, model_result: savedModelResult });
      setRunStatus({
        jobId: job.id,
        type: "success",
        message: `Loaded saved ${job.task_type} result for job #${job.id}.`
      });

      try {
        const cachedExplanation = await getAiExplanation(job.id);
        setAiExplanationState({
          explanation: cachedExplanation,
          jobId: job.id,
          message: "Loaded saved local AI explanation.",
          status: "success"
        });
      } catch (error) {
        setAiExplanationState({
          explanation: null,
          jobId: job.id,
          message: error.status === 404 ? "" : aiExplanationErrorMessage(error),
          status: error.status === 404 ? "idle" : "error"
        });
      }
    } catch (error) {
      setAiExplanationState({ explanation: null, jobId: job.id, message: "", status: "idle" });
      setRunStatus({
        jobId: job.id,
        type: "error",
        message: error.message || "Could not load the saved result for this job."
      });
    }
  }

  async function handleGenerateAiExplanation(jobId) {
    setAiExplanationState((current) => ({
      explanation: current.jobId === jobId ? current.explanation : null,
      jobId,
      message: "Generating local AI explanation...",
      status: "loading"
    }));

    try {
      const explanation = await generateAiExplanation(jobId);
      setAiExplanationState({
        explanation,
        jobId,
        message: "Local AI explanation generated.",
        status: "success"
      });
    } catch (error) {
      setAiExplanationState((current) => ({
        explanation: current.jobId === jobId ? current.explanation : null,
        jobId,
        message: aiExplanationErrorMessage(error),
        status: "error"
      }));
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

        <DatasetInsightsCard
          cleaning={cleaning}
          dataset={dataset}
          preview={preview}
          targetColumn={jobForm.target_column}
        />

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
          <AnalysisDatasetSource cleanResult={cleanResult} dataset={dataset} />
          <AnalysisRecommendationCard
            onUseRecommendation={handleUseRecommendation}
            recommendationState={recommendationState}
          />
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
          <div className="analysis-helper">
            <p>
              {dataset
                ? "Create the request, then run it from the recent jobs list."
                : "Upload a dataset to enable analysis requests."}
            </p>
            <p>Classification, regression, and forecasting training are available.</p>
            {numericColumns.length ? <p>Numeric columns: {numericColumns.slice(0, 4).join(", ")}</p> : null}
          </div>
          {jobStatus.message ? <div aria-live="polite" className={`backend-status ${jobStatus.type}`} role="status">{jobStatus.message}</div> : null}
          {runStatus.message ? <div aria-live="polite" className={`backend-status ${runStatus.type}`} role="status">{runStatus.message}</div> : null}
          {jobs.length ? (
            <section className="job-queue" aria-label="Recent analysis jobs">
              <div className="section-kicker">
                <strong>Recent jobs</strong>
                <span>{jobs.slice(0, 4).length} shown</span>
              </div>
              <div className="job-list">
                {jobs.slice(0, 4).map((job) => (
                  <div className="job-row" key={job.id}>
                    <span className="job-details">
                      <strong>#{job.id} {job.task_type}</strong>
                      <small className="job-file" title={job.dataset_file_name || "Uploaded dataset"}>{job.dataset_file_name || "Uploaded dataset"}</small>
                      <small className="job-target">Target: {job.target_column}</small>
                      <small className="job-source">Source: {job.dataset_source_label || "Original uploaded file"}</small>
                    </span>
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
            </section>
          ) : null}
        </section>

        <ModelResultCard
          aiExplanationState={aiExplanationState}
          onGenerateAiExplanation={handleGenerateAiExplanation}
          result={modelResult}
        />
        <DatasetPreviewCard dataset={dataset} preview={preview} />
      </section>
    </main>
  );
}
