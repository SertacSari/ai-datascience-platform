import { useNavigate } from "react-router-dom";
import Brand from "./Brand";

export default function AppShell({ activeView, children, onLogOut, onTheme, theme, user }) {
  const navigate = useNavigate();

  return (
    <div className="app-shell">
      <header className="top-nav">
        <button className="brand-button" onClick={() => navigate("/dashboard")} type="button">
          <Brand />
        </button>
        <nav className="nav-links" aria-label="Primary">
          <button className={activeView === "dashboard" ? "active" : ""} onClick={() => navigate("/dashboard")} type="button">
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
