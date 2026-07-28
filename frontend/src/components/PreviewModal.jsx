import { useEffect, useRef } from "react";

export default function PreviewModal({ onClose, report }) {
  const closeRef = useRef(null);

  useEffect(() => {
    closeRef.current?.focus();
  }, []);

  return (
    <div className="modal-scrim" onMouseDown={onClose}>
      <section aria-modal="true" className="preview-modal" onMouseDown={(event) => event.stopPropagation()} role="dialog">
        <div className="card-head">
          <div>
            <p className="eyebrow">Report preview</p>
            <h2>{report.name}</h2>
          </div>
          <button className="icon-button" onClick={onClose} ref={closeRef} type="button">X</button>
        </div>
        <dl className="preview-meta">
          <div><dt>Type</dt><dd>{report.type}</dd></div>
          <div><dt>Created</dt><dd>{report.date}</dd></div>
          <div><dt>Status</dt><dd>{report.status}</dd></div>
          <div><dt>Size</dt><dd>{report.size}</dd></div>
        </dl>
        <p>{report.summary}</p>
        <div className="modal-actions">
          <button className="button" onClick={onClose} type="button">Close</button>
          <button className="button primary" disabled type="button">Download unavailable</button>
        </div>
      </section>
    </div>
  );
}
