import Badge from "./Badge";

export default function ReportRow({ armed, downloaded, onArm, onDelete, onDownload, onPreview, onToggle, report, selected }) {
  return (
    <tr>
      <td><input aria-label={`Select ${report.name}`} checked={selected} onChange={onToggle} type="checkbox" /></td>
      <td>
        <div className="report-name">
          <strong>{report.name}</strong>
          <span>{report.summary.slice(0, 72)}...</span>
        </div>
      </td>
      <td><Badge tone="type">{report.type}</Badge></td>
      <td>{report.date}</td>
      <td>{report.size}</td>
      <td><Badge tone={report.status === "Ready" ? "ok" : report.status === "Processing" ? "warn" : "err"}>{report.status}</Badge></td>
      <td>
        <div className="row-actions">
          <button aria-label={`Preview ${report.name}`} className="icon-button" onClick={onPreview} type="button">PV</button>
          <button aria-label={`Download ${report.name}`} className="icon-button primary" disabled={report.status !== "Ready"} onClick={onDownload} type="button">
            {downloaded ? "OK" : "DL"}
          </button>
          <button aria-label={`Delete ${report.name}`} className={`icon-button ${armed ? "danger" : ""}`} data-delete-action onClick={armed ? onDelete : onArm} type="button">
            {armed ? "YES" : "DEL"}
          </button>
        </div>
      </td>
    </tr>
  );
}
