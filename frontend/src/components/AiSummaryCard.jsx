import Badge from "./Badge";

export default function AiSummaryCard({ cleaning, dataset, preview }) {
  const issueCount = cleaning?.issues?.length || 0;
  const duplicateRows = cleaning?.duplicate_rows || 0;
  const columns = preview?.columns?.slice(0, 3).join(", ");

  return (
    <section className="card ai-card">
      <div className="card-head">
        <div>
          <p className="eyebrow">Dataset summary</p>
          <h2>{dataset ? "Cleaning summary" : "Upload summary"}</h2>
        </div>
        <Badge tone={dataset ? cleaning?.ready_for_ml ? "ok" : "warn" : "type"}>
          {dataset ? "Backend" : "Waiting"}
        </Badge>
      </div>
      {dataset ? (
        <>
          <p>
            <strong>{dataset.file_name}</strong> contains <strong>{dataset.row_count.toLocaleString()} rows</strong>{" "}
            across <strong>{dataset.column_count} columns</strong>.
          </p>
          <p>
            Cleaning report found <strong>{issueCount} issue groups</strong> and{" "}
            <strong>{duplicateRows} duplicate rows</strong>.{" "}
            {cleaning?.ready_for_ml ? "The dataset is ready for ML checks." : "Review or clean it before modeling."}
          </p>
        </>
      ) : (
        <>
          <p>
            Upload a dataset to load backend preview details and cleaning checks.
          </p>
          <p>
            Classification, regression, and forecasting jobs can run here. Final reports come later.
          </p>
        </>
      )}
      <div className="chip-row">
        <span>{dataset ? "real upload" : "waiting for upload"}</span>
        <span>{columns || "preview columns"}</span>
        <span>{cleaning?.ready_for_ml ? "ready for ML" : "review needed"}</span>
      </div>
    </section>
  );
}
