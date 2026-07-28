import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { getMe, login, register } from "../api/client";
import Badge from "../components/Badge";
import BarChart from "../components/BarChart";
import Brand from "../components/Brand";
import Feature from "../components/Feature";
import { useSession } from "../context/SessionContext";

export default function LoginPage() {
  const navigate = useNavigate();
  const { signIn } = useSession();
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
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

      await login(email.trim(), password);
      const profile = await getMe();
      signIn(profile);
      navigate("/dashboard", { replace: true });
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
            <p className="muted">Use your project account to continue.</p>
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
