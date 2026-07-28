import { useRef } from "react";

export default function Dropzone({ onUpload, upload }) {
  const inputRef = useRef(null);

  function handleFiles(files) {
    const file = files?.[0];
    if (file) onUpload(file);
    if (inputRef.current) inputRef.current.value = "";
  }

  return (
    <section
      className="card dropzone"
      onClick={() => inputRef.current?.click()}
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        event.preventDefault();
        handleFiles(event.dataTransfer.files);
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          inputRef.current?.click();
        }
      }}
      role="button"
      tabIndex="0"
    >
      <input accept=".csv,.xlsx,.xls" className="sr-only" onChange={(event) => handleFiles(event.target.files)} ref={inputRef} type="file" />
      <div className="drop-icon">CSV/XLS</div>
      <h2>Drop a CSV or Excel file here</h2>
      <p>CSV or Excel. Recommended max size: 20 MB.</p>
      {upload.status === "uploading" ? (
        <div className="upload-status">
          <span>{upload.file}</span>
          <div className="progress"><i style={{ width: `${upload.progress}%` }} /></div>
          <strong>{upload.progress}%</strong>
        </div>
      ) : null}
      {upload.status === "done" ? <div aria-live="polite" className="success-chip" role="status">{upload.message}</div> : null}
      {upload.status === "error" ? <div className="backend-status error" role="alert">{upload.message}</div> : null}
    </section>
  );
}
