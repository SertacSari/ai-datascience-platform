import { useEffect, useMemo, useRef, useState } from "react";
import {
  API_BASE_URL,
  cleanDataset,
  createAnalysisJob,
  getCleaningReport,
  getDatasetPreview,
  getMe,
  listAnalysisJobs,
  login,
  register,
  uploadDataset
} from "./src/api";

const TOKEN_KEY = "datavista_access_token";

function emptyDashboardData() {
  return { cleanResult: null, cleaning: null, dataset: null, preview: null };
}

function readStoredToken() {
  return localStorage.getItem(TOKEN_KEY) || sessionStorage.getItem(TOKEN_KEY);
}

function clearStoredToken() {
  localStorage.removeItem(TOKEN_KEY);
  sessionStorage.removeItem(TOKEN_KEY);
}

function storeToken(accessToken, remember) {
  clearStoredToken();
  const storage = remember ? localStorage : sessionStorage;
  storage.setItem(TOKEN_KEY, accessToken);
}

function reportSizeToMb(size) {
  const match = /^([\d.]+)\s*(KB|MB)$/i.exec(size);
  if (!match) return 0;
  const value = Number(match[1]);
  return match[2].toUpperCase() === "KB" ? value / 1024 : value;
}

function formatArchiveSize(reports) {
  const total = reports.reduce((sum, report) => sum + reportSizeToMb(report.size), 0);
  return total >= 1 ? `${total.toFixed(1)} MB` : `${Math.round(total * 1024)} KB`;
}

const reportSeed = [
  ["Q3 Tech Sector Analysis.pdf", "Analysis", "2026-07-09", "2.4 MB", "Ready"],
  ["Forex Daily Summary.pdf", "Forex", "2026-07-08", "1.1 MB", "Ready"],
  ["Custom CSV Data Report.pdf", "Custom", "2026-07-07", "3.5 MB", "Ready"],
  ["Historical Trend Deep Dive.pdf", "Historical", "2026-07-06", "4.8 MB", "Processing"],
  ["Q2 Technical Momentum Report.pdf", "Technical", "2026-07-04", "1.8 MB", "Ready"],
  ["Retail Revenue Summary.pdf", "Summary", "2026-07-03", "920 KB", "Ready"],
  ["Emerging Markets Analysis.pdf", "Analysis", "2026-07-02", "4.2 MB", "Processing"],
  ["Volatility Index Report.pdf", "Technical", "2026-07-01", "2.1 MB", "Failed"],
  ["Customer Churn Model Notes.pdf", "Analysis", "2026-06-29", "2.9 MB", "Ready"],
  ["House Price Regression Summary.pdf", "Summary", "2026-06-27", "1.7 MB", "Ready"],
  ["Coffee Shop Sales Forecast.pdf", "Custom", "2026-06-26", "3.2 MB", "Ready"],
  ["Macro Signals Weekly.pdf", "Forex", "2026-06-24", "1.4 MB", "Ready"],
  ["Dataset Cleaning Audit.pdf", "Technical", "2026-06-22", "875 KB", "Ready"],
  ["Inventory Demand Review.pdf", "Historical", "2026-06-20", "3.8 MB", "Ready"],
  ["North Region Sales Analysis.pdf", "Analysis", "2026-06-18", "2.6 MB", "Processing"],
  ["Daily Sales Forecasting.pdf", "Summary", "2026-06-16", "1.9 MB", "Ready"],
  ["Campaign Response Report.pdf", "Custom", "2026-06-14", "3.1 MB", "Ready"],
  ["Margin Risk Technical Report.pdf", "Technical", "2026-06-12", "2.2 MB", "Ready"],
  ["Currency Exposure Snapshot.pdf", "Forex", "2026-06-10", "1.2 MB", "Ready"],
  ["Quarterly Archive Pack.pdf", "Historical", "2026-06-08", "5.6 MB", "Ready"],
  ["Anomaly Detection Notes.pdf", "Analysis", "2026-06-05", "2.0 MB", "Failed"],
  ["Executive Summary Pack.pdf", "Summary", "2026-06-02", "1.5 MB", "Ready"],
  ["Uploaded CSV Profiling Report.pdf", "Custom", "2026-05-30", "3.4 MB", "Ready"]
].map(([name, type, date, size, status], index) => ({
  id: index + 1,
  name,
  type,
  date,
  size,
  status,
  summary:
    "The report identifies stable baseline performance with visible seasonality, a small outlier cluster, and a recommendation to segment the next pass by date range and source column."
}));

const statRows = [
  ["Rows previewed", "128,440", "sample", "ok"],
  ["Columns profiled", "42", "sample", "ok"],
  ["Quality checks", "4", "ready", "ok"],
  ["Jobs created", "0", "upload first", "warn"]
];

const technicalRows = [
  ["Duplicate rows", "0", "Sample", "ok"],
  ["Missing values", "1.7%", "Sample", "ok"],
  ["Column types", "Mixed", "Sample", "ok"],
  ["Preview rows", "0", "Upload", "warn"]
];

const rangePoints = { "1W": 7, "1M": 30, "3M": 90, "1Y": 52 };

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

function App() {
  const [view, setView] = useState("login");
  const [user, setUser] = useState(null);
  const [token, setToken] = useState(readStoredToken);
  const [checkingSession, setCheckingSession] = useState(Boolean(token));
  const [reports, setReports] = useState(reportSeed);
  const [theme, setTheme] = useState("system");
  const [dashboardData, setDashboardData] = useState(() => emptyDashboardData());

  useEffect(() => {
    if (theme === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    const onUnauthorized = () => {
      clearStoredToken();
      setToken(null);
      setUser(null);
      setView("login");
    };

    window.addEventListener("datavista:unauthorized", onUnauthorized);
    return () => window.removeEventListener("datavista:unauthorized", onUnauthorized);
  }, []);

  useEffect(() => {
    if (!token) {
      setCheckingSession(false);
      return undefined;
    }

    let cancelled = false;
    setCheckingSession(true);
    getMe(token)
      .then((profile) => {
        if (!cancelled) {
          setUser(profile);
          setView("dashboard");
        }
      })
      .catch(() => {
        if (!cancelled) {
          clearStoredToken();
          setToken(null);
          setUser(null);
          setView("login");
        }
      })
      .finally(() => {
        if (!cancelled) setCheckingSession(false);
      });

    return () => {
      cancelled = true;
    };
  }, [token]);

  function signIn(accessToken, profile, remember) {
    storeToken(accessToken, remember);
    setToken(accessToken);
    setUser(profile);
    setView("dashboard");
  }

  function logOut() {
    clearStoredToken();
    setDashboardData(emptyDashboardData());
    setToken(null);
    setUser(null);
    setView("login");
  }

  if (checkingSession) {
    return (
      <main className="loading-screen">
        <Brand />
        <strong>Checking BasitAnaliz session...</strong>
      </main>
    );
  }

  if (view === "login" || !token || !user) {
    return <LoginPage onSignIn={signIn} />;
  }

  return (
    <AppShell
      activeView={view}
      onLogOut={logOut}
      onNavigate={setView}
      onTheme={() => setTheme((current) => current === "system" ? "light" : current === "light" ? "dark" : "system")}
      reports={reports}
      theme={theme}
      user={user}
    >
      {view === "reports" ? (
        <ReportsPage reports={reports} setReports={setReports} onNavigate={setView} />
      ) : (
        <DashboardPage
          dashboardData={dashboardData}
          reports={reports}
          setDashboardData={setDashboardData}
          token={token}
          onNavigate={setView}
        />
      )}
    </AppShell>
  );
}

function LoginPage({ onSignIn }) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [remember, setRemember] = useState(true);
  const [status, setStatus] = useState({ type: "idle", message: "" });
  const [errors, setErrors] = useState({});
  const passwordRef = useRef(null);

  async function submit(event) {
    event.preventDefault();
    const nextErrors = {};
    if (mode === "register" && username.trim().length < 3) {
      nextErrors.username = "Username must be at least 3 characters.";
    }
    if (!email.trim()) nextErrors.email = "Email is required.";
    if (!password.trim()) nextErrors.password = "Password is required.";
    if (mode === "register" && password.trim().length < 6) {
      nextErrors.password = "Password must be at least 6 characters.";
    }
    setErrors(nextErrors);
    if (Object.keys(nextErrors).length) return;

    setStatus({ type: "loading", message: "" });
    try {
      if (mode === "register") {
        await register({
          email: email.trim(),
          password,
          username: username.trim()
        });
      }

      const tokenResponse = await login(email.trim(), password);
      const profile = await getMe(tokenResponse.access_token);
      onSignIn(tokenResponse.access_token, profile, remember);
    } catch (error) {
      setStatus({
        type: "error",
        message: error.message || "Could not sign in. Check both fields and try again."
      });
      passwordRef.current?.focus();
    }
  }

  return (
    <main className="login-screen">
      <section className="login-left">
        <Brand />
        <div className="login-copy">
          <p className="eyebrow">Terminal ledger analytics</p>
          <h1>Upload datasets. Check quality. Create analysis jobs.</h1>
          <p>
            BasitAnaliz keeps upload, quality checks, and job setup in one compact
            workspace.
          </p>
        </div>
        <section className="card preview-card">
          <div className="card-head">
            <div>
              <p className="eyebrow">Dataset workflow</p>
              <h2>Quality preview</h2>
            </div>
            <Badge tone="neutral">Sample</Badge>
          </div>
          <BarChart label="Sample dataset quality preview" />
        </section>
        <div className="feature-list">
          <Feature title="Analysis Requests" text="Create classification, regression, and forecasting jobs." />
          <Feature title="Quality Checks" text="Review missing values, duplicates, and column signals." />
          <Feature title="CSV Upload" text="Fast dataset profiling with clear quality feedback." />
        </div>
      </section>

      <section className="login-right">
        <form className="login-card card" onSubmit={submit}>
          <div>
            <p className="eyebrow">Secure workspace</p>
            <h2>{mode === "register" ? "Create account" : "Sign in"}</h2>
            <p className="muted">Connected to {API_BASE_URL}</p>
          </div>
          {status.type === "error" ? (
            <div className="alert" role="alert">
              {status.message}
            </div>
          ) : null}
          {mode === "register" ? (
            <label className="field">
              <span>Username</span>
              <input
                aria-invalid={Boolean(errors.username)}
                autoComplete="username"
                onChange={(event) => setUsername(event.target.value)}
                placeholder="analyst"
                type="text"
                value={username}
              />
              {errors.username ? <small>{errors.username}</small> : null}
            </label>
          ) : null}
          <label className="field">
            <span>Email</span>
            <input
              aria-invalid={Boolean(errors.email)}
              autoComplete="email"
              onChange={(event) => setEmail(event.target.value)}
              placeholder="analyst@example.com"
              type="email"
              value={email}
            />
            {errors.email ? <small>{errors.email}</small> : null}
          </label>
          <label className="field">
            <span>Password</span>
            <div className="password-field">
              <input
                aria-invalid={Boolean(errors.password)}
                autoComplete={mode === "register" ? "new-password" : "current-password"}
                onChange={(event) => setPassword(event.target.value)}
                placeholder="Enter password"
                ref={passwordRef}
                type={showPassword ? "text" : "password"}
                value={password}
              />
              <button
                aria-pressed={showPassword}
                onClick={() => setShowPassword((value) => !value)}
                type="button"
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>
            {errors.password ? <small>{errors.password}</small> : null}
          </label>
          <div className="form-row">
            <label className="check-row">
              <input checked={remember} onChange={(event) => setRemember(event.target.checked)} type="checkbox" />
              <span>Remember me</span>
            </label>
            <button className="link-button" type="button">Forgot password?</button>
          </div>
          <button className="button primary" disabled={status.type === "loading"} type="submit">
            {status.type === "loading"
              ? mode === "register"
                ? "Creating account..."
                : "Signing in..."
              : mode === "register"
                ? "Create account"
                : "Sign In to BasitAnaliz"}
          </button>
          <button
            className="button"
            onClick={() => {
              setMode((current) => current === "login" ? "register" : "login");
              setErrors({});
              setStatus({ type: "idle", message: "" });
            }}
            type="button"
          >
            {mode === "register" ? "Already registered? Sign in" : "Need an account? Register"}
          </button>
          <p className="footnote">Uses FastAPI auth endpoints. Register once, then sign in.</p>
        </form>
      </section>
    </main>
  );
}

function AppShell({ activeView, children, onLogOut, onNavigate, onTheme, theme, user }) {
  return (
    <div className="app-shell">
      <header className="top-nav">
        <button className="brand-button" onClick={() => onNavigate("dashboard")} type="button">
          <Brand />
        </button>
        <nav className="nav-links" aria-label="Primary">
          <button className={activeView === "dashboard" ? "active" : ""} onClick={() => onNavigate("dashboard")} type="button">
            Dashboard
          </button>
        </nav>
        <div className="top-actions">
          <button className="button sm" onClick={onTheme} type="button">Theme: {theme}</button>
          <div className="avatar-chip" title={user?.email}>
            <span>{((user?.username || user?.name)?.[0] || "A").toUpperCase()}</span>
            <strong>{user?.username || user?.name}</strong>
          </div>
          <button className="button sm" onClick={onLogOut} type="button">Log out</button>
        </div>
      </header>
      {children}
    </div>
  );
}

function DashboardPage({ dashboardData, onNavigate, reports, setDashboardData, token }) {
  const [range, setRange] = useState("1M");
  const [upload, setUpload] = useState({ status: "idle", file: "", progress: 0, message: "" });
  const [jobs, setJobs] = useState([]);
  const [jobForm, setJobForm] = useState({ task_type: "regression", target_column: "", date_column: "" });
  const [cleanStatus, setCleanStatus] = useState({ type: "idle", message: "" });
  const [jobStatus, setJobStatus] = useState({ type: "idle", message: "" });
  const { cleanResult, cleaning, dataset, preview } = dashboardData;
  const stats = dashboardStats(dataset, cleaning, jobs);
  const technical = technicalFromBackend(cleaning, preview);
  const numericColumns = numericColumnsFromPreview(preview);

  useEffect(() => {
    listAnalysisJobs(token)
      .then(setJobs)
      .catch(() => setJobs([]));
  }, [token]);

  async function startUpload(file) {
    if (!file) return;

    setUpload({ status: "uploading", file: file.name, progress: 30, message: "" });
    setDashboardData(emptyDashboardData());
    setCleanStatus({ type: "idle", message: "" });
    setJobStatus({ type: "idle", message: "" });

    try {
      const uploaded = await uploadDataset(file, token);
      setUpload((current) => ({ ...current, progress: 70 }));
      const [nextPreview, nextCleaning, nextJobs] = await Promise.all([
        getDatasetPreview(uploaded.id, token),
        getCleaningReport(uploaded.id, token),
        listAnalysisJobs(token)
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
      const result = await cleanDataset(dataset.id, token);
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
        },
        token
      );
      const nextJobs = await listAnalysisJobs(token);
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

function ReportsPage({ onNavigate, reports, setReports }) {
  const [filters, setFilters] = useState({ query: "", from: "", to: "", type: "All", sort: "newest" });
  const [page, setPage] = useState(1);
  const [selected, setSelected] = useState(new Set());
  const [preview, setPreview] = useState(null);
  const [armedDelete, setArmedDelete] = useState(null);
  const [downloaded, setDownloaded] = useState(null);
  const selectAllRef = useRef(null);
  const pageSize = 8;

  const filtered = useMemo(() => {
    const query = filters.query.trim().toLowerCase();
    return reports
      .filter((report) => {
        return (
          (!query || report.name.toLowerCase().includes(query) || report.type.toLowerCase().includes(query)) &&
          (filters.type === "All" || report.type === filters.type) &&
          (!filters.from || report.date >= filters.from) &&
          (!filters.to || report.date <= filters.to)
        );
      })
      .sort((a, b) => {
        if (filters.sort === "oldest") return a.date.localeCompare(b.date);
        if (filters.sort === "name") return a.name.localeCompare(b.name);
        if (filters.sort === "type") return a.type.localeCompare(b.type);
        return b.date.localeCompare(a.date);
      });
  }, [filters, reports]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));
  const safePage = Math.min(page, totalPages);
  const start = (safePage - 1) * pageSize;
  const pageRows = filtered.slice(start, start + pageSize);
  const pageIds = pageRows.map((report) => report.id);
  const allPageSelected = pageIds.length > 0 && pageIds.every((id) => selected.has(id));
  const somePageSelected = pageIds.some((id) => selected.has(id));

  useEffect(() => {
    if (selectAllRef.current) selectAllRef.current.indeterminate = somePageSelected && !allPageSelected;
  }, [allPageSelected, somePageSelected]);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.key === "Escape") {
        setPreview(null);
        setArmedDelete(null);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  function updateFilter(key, value) {
    setFilters((current) => ({ ...current, [key]: value }));
    setPage(1);
  }

  function clearFilters() {
    setFilters({ query: "", from: "", to: "", type: "All", sort: "newest" });
    setPage(1);
  }

  function toggleOne(id) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  function togglePage() {
    setSelected((current) => {
      const next = new Set(current);
      if (allPageSelected) pageIds.forEach((id) => next.delete(id));
      else pageIds.forEach((id) => next.add(id));
      return next;
    });
  }

  function deleteReports(ids) {
    setReports((current) => current.filter((report) => !ids.includes(report.id)));
    setSelected((current) => {
      const next = new Set(current);
      ids.forEach((id) => next.delete(id));
      return next;
    });
    setArmedDelete(null);
  }

  function download(report) {
    if (report.status !== "Ready") return;
    setDownloaded(report.id);
    window.setTimeout(() => setDownloaded(null), 1000);
  }

  const summary = {
    total: reports.length,
    ready: reports.filter((report) => report.status === "Ready").length,
    processing: reports.filter((report) => report.status === "Processing").length,
    size: formatArchiveSize(reports)
  };

  return (
    <main
      className="page-shell"
      onPointerDown={(event) => {
        if (!event.target.closest("[data-delete-action]")) setArmedDelete(null);
      }}
    >
      <div className="reports-head">
        <button className="button" onClick={() => onNavigate("dashboard")} type="button">Dashboard</button>
        <div>
          <p className="eyebrow">Advanced archive</p>
          <h1>Sample Report Archive</h1>
        </div>
        <Badge tone="type">Sample reports</Badge>
      </div>

      <section className="summary-grid">
        <StatCard label="Total reports" value={summary.total} delta="All records" tone="neutral" />
        <StatCard label="Ready" value={summary.ready} delta="Downloadable" tone="ok" />
        <StatCard label="Processing" value={summary.processing} delta="In queue" tone="warn" />
        <StatCard label="Archive size" value={summary.size} delta="PDF storage" tone="neutral" />
      </section>

      <section className="card filter-card">
        <label>
          <span>Keyword</span>
          <input onChange={(event) => updateFilter("query", event.target.value)} placeholder="Search by file name or keyword..." type="search" value={filters.query} />
        </label>
        <label>
          <span>From</span>
          <input onChange={(event) => updateFilter("from", event.target.value)} type="date" value={filters.from} />
        </label>
        <label>
          <span>To</span>
          <input onChange={(event) => updateFilter("to", event.target.value)} type="date" value={filters.to} />
        </label>
        <label>
          <span>Report type</span>
          <select onChange={(event) => updateFilter("type", event.target.value)} value={filters.type}>
            <option>All</option>
            <option>Analysis</option>
            <option>Summary</option>
            <option>Custom</option>
            <option>Historical</option>
            <option>Forex</option>
            <option>Technical</option>
          </select>
        </label>
        <label>
          <span>Sort</span>
          <select onChange={(event) => updateFilter("sort", event.target.value)} value={filters.sort}>
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
            <option value="name">Name A-Z</option>
            <option value="type">Type A-Z</option>
          </select>
        </label>
        <button className="button" onClick={clearFilters} type="button">Clear</button>
      </section>

      {selected.size ? (
        <section className="bulk-bar">
          <strong>{selected.size} selected</strong>
          <button type="button">Download selected</button>
          <button onClick={() => deleteReports(Array.from(selected))} type="button">Delete selected</button>
          <button onClick={() => setSelected(new Set())} type="button">Clear</button>
        </section>
      ) : null}

      <section className="card table-card">
        <div className="table-wrap">
          <table className="reports-table">
            <thead>
              <tr>
                <th>
                  <input aria-label="Select current page" checked={allPageSelected} onChange={togglePage} ref={selectAllRef} type="checkbox" />
                </th>
                <th>Report</th>
                <th>Type</th>
                <th>Created</th>
                <th>Size</th>
                <th>Status</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {pageRows.length ? pageRows.map((report) => (
                <ReportRow
                  armed={armedDelete === report.id}
                  downloaded={downloaded === report.id}
                  key={report.id}
                  onArm={() => setArmedDelete(report.id)}
                  onDelete={() => deleteReports([report.id])}
                  onDownload={() => download(report)}
                  onPreview={() => setPreview(report)}
                  onToggle={() => toggleOne(report.id)}
                  report={report}
                  selected={selected.has(report.id)}
                />
              )) : (
                <tr>
                  <td className="empty-cell" colSpan="7">
                    <strong>No reports match these filters.</strong>
                    <button className="link-button" onClick={clearFilters} type="button">Clear filters</button>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <Pager currentPage={safePage} pageSize={pageSize} total={filtered.length} totalPages={totalPages} onPage={setPage} />
      </section>

      {preview ? <PreviewModal onClose={() => setPreview(null)} report={preview} /> : null}
    </main>
  );
}

function ReportRow({ armed, downloaded, onArm, onDelete, onDownload, onPreview, onToggle, report, selected }) {
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

function Dropzone({ onUpload, upload }) {
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

function AiSummaryCard({ cleaning, dataset, preview }) {
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
            This view creates analysis jobs only. Model training and generated results come later.
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

function PreviewModal({ onClose, report }) {
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

function StatCard({ delta, label, tone, value }) {
  return (
    <div className="stat">
      <span>{label}</span>
      <strong>{value}</strong>
      <em className={tone}>{delta}</em>
    </div>
  );
}

function Segmented({ onChange, value, values }) {
  return (
    <div className="segmented" role="group" aria-label="Time range">
      {values.map((item) => (
        <button
          aria-pressed={value === item}
          className={value === item ? "active" : ""}
          key={item}
          onClick={() => onChange(item)}
          type="button"
        >
          {item}
        </button>
      ))}
    </div>
  );
}

function Pager({ currentPage, onPage, pageSize, total, totalPages }) {
  const first = total === 0 ? 0 : (currentPage - 1) * pageSize + 1;
  const last = Math.min(currentPage * pageSize, total);
  return (
    <div className="pager">
      <span>{first}-{last} of {total} reports</span>
      <div>
        <button disabled={currentPage === 1} onClick={() => onPage(currentPage - 1)} type="button">&lt;</button>
        {Array.from({ length: totalPages }, (_, index) => index + 1).map((pageNumber) => (
          <button className={pageNumber === currentPage ? "active" : ""} key={pageNumber} onClick={() => onPage(pageNumber)} type="button">
            {pageNumber}
          </button>
        ))}
        <button disabled={currentPage === totalPages} onClick={() => onPage(currentPage + 1)} type="button">&gt;</button>
      </div>
    </div>
  );
}

function Feature({ text, title }) {
  return (
    <div className="feature">
      <span aria-hidden="true">+</span>
      <div>
        <strong>{title}</strong>
        <p>{text}</p>
      </div>
    </div>
  );
}

function Badge({ children, tone = "neutral" }) {
  return <span className={`badge ${tone}`}>{children}</span>;
}

function Brand() {
  return (
    <span className="brand">
      <span className="brand-mark">BA</span>
      <span>BasitAnaliz</span>
    </span>
  );
}

function LineChart({ label, range }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return undefined;
    const draw = () => drawLine(canvas, range);
    draw();
    const observer = new ResizeObserver(draw);
    if (canvas.parentElement) observer.observe(canvas.parentElement);
    const themeObserver = new MutationObserver(draw);
    themeObserver.observe(document.documentElement, { attributeFilter: ["data-theme"], attributes: true });
    return () => {
      observer.disconnect();
      themeObserver.disconnect();
    };
  }, [range]);

  return <canvas aria-label={label} className="chart-canvas" ref={ref} role="img" />;
}

function BarChart({ label }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return undefined;
    const draw = () => drawBars(canvas);
    draw();
    const observer = new ResizeObserver(draw);
    if (canvas.parentElement) observer.observe(canvas.parentElement);
    const themeObserver = new MutationObserver(draw);
    themeObserver.observe(document.documentElement, { attributeFilter: ["data-theme"], attributes: true });
    return () => {
      observer.disconnect();
      themeObserver.disconnect();
    };
  }, []);

  return <canvas aria-label={label} className="bar-canvas" ref={ref} role="img" />;
}

function prepCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * ratio));
  canvas.height = Math.max(1, Math.floor(rect.height * ratio));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return { ctx, height: rect.height, width: rect.width };
}

function token(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function alphaColor(color, alpha) {
  const hex = color.replace("#", "").trim();
  if (!/^[\da-f]{6}$/i.test(hex)) return color;
  const parts = [0, 2, 4].map((start) => parseInt(hex.slice(start, start + 2), 16));
  return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`;
}

function drawLine(canvas, range) {
  const { ctx, height, width } = prepCanvas(canvas);
  const primary = token("--primary");
  const border = token("--border");
  const muted = token("--muted");
  const card = token("--card");
  const count = rangePoints[range];
  const data = Array.from({ length: count }, (_, index) => {
    return 62 + Math.sin(index / 3.4) * 16 + Math.cos(index / 8) * 9 + index * 0.18;
  });
  const min = Math.min(...data) - 8;
  const max = Math.max(...data) + 8;
  const pad = 26;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = card;
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i += 1) {
    const y = pad + ((height - pad * 2) / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
  }

  const xy = (value, index) => [
    pad + (index / (data.length - 1)) * (width - pad * 2),
    height - pad - ((value - min) / (max - min)) * (height - pad * 2)
  ];

  ctx.beginPath();
  data.forEach((value, index) => {
    const [x, y] = xy(value, index);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(width - pad, height - pad);
  ctx.lineTo(pad, height - pad);
  ctx.closePath();
  ctx.fillStyle = alphaColor(primary, 0.1);
  ctx.fill();

  ctx.beginPath();
  data.forEach((value, index) => {
    const [x, y] = xy(value, index);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = primary;
  ctx.lineWidth = 2.5;
  ctx.stroke();

  const last = xy(data[data.length - 1], data.length - 1);
  ctx.fillStyle = primary;
  ctx.beginPath();
  ctx.arc(last[0], last[1], 4, 0, Math.PI * 2);
  ctx.fill();

  ctx.setLineDash([5, 5]);
  ctx.beginPath();
  ctx.moveTo(last[0], last[1]);
  ctx.lineTo(width - 10, Math.max(pad, last[1] - 18));
  ctx.strokeStyle = muted;
  ctx.lineWidth = 1.5;
  ctx.stroke();
  ctx.setLineDash([]);
}

function drawBars(canvas) {
  const { ctx, height, width } = prepCanvas(canvas);
  const primary = token("--primary");
  const border = token("--border");
  const muted = token("--muted");
  const values = [28, 35, 32, 44, 48, 46, 58, 62, 59, 68, 72, 81];
  const pad = 18;
  const gap = 6;
  const barWidth = (width - pad * 2) / values.length - gap;

  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = border;
  ctx.lineWidth = 1;
  for (let i = 0; i < 4; i += 1) {
    const y = pad + ((height - pad * 2) / 3) * i;
    ctx.beginPath();
    ctx.moveTo(pad, y);
    ctx.lineTo(width - pad, y);
    ctx.stroke();
  }

  values.forEach((value, index) => {
    const x = pad + index * (barWidth + gap);
    const barHeight = (value / 90) * (height - pad * 2);
    ctx.globalAlpha = index > 8 ? 1 : 0.38;
    ctx.fillStyle = index > 8 ? primary : muted;
    ctx.fillRect(x, height - pad - barHeight, barWidth, barHeight);
  });
  ctx.globalAlpha = 1;
}

export default App;
