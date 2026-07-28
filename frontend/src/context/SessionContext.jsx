import { createContext, useContext, useEffect, useMemo, useState } from "react";
import { clearLegacyStoredToken, getMe, logout } from "../api/client";

const SessionContext = createContext(null);

export function emptyDashboardData() {
  return { cleanResult: null, cleaning: null, dataset: null, preview: null };
}

export function clearStoredToken() {
  clearLegacyStoredToken();
}

export function SessionProvider({ children }) {
  const [viewReady, setViewReady] = useState(false);
  const [user, setUser] = useState(null);
  const [checkingSession, setCheckingSession] = useState(true);
  const [theme, setTheme] = useState("system");
  const [dashboardData, setDashboardData] = useState(() => emptyDashboardData());
  const [jobs, setJobs] = useState([]);

  useEffect(() => {
    if (theme === "system") document.documentElement.removeAttribute("data-theme");
    else document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    let cancelled = false;
    setCheckingSession(true);
    clearStoredToken();
    getMe()
      .then((profile) => {
        if (!cancelled) setUser(profile);
      })
      .catch(() => {
        if (!cancelled) {
          setUser(null);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setCheckingSession(false);
          setViewReady(true);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  function signIn(profile) {
    clearStoredToken();
    setUser(profile);
  }

  async function logOut() {
    try {
      await logout();
    } catch {
      // Clear client state even if the server cookie is already absent or expired.
    }
    clearStoredToken();
    setDashboardData(emptyDashboardData());
    setJobs([]);
    setUser(null);
  }

  const value = useMemo(() => ({
    checkingSession,
    dashboardData,
    jobs,
    logOut,
    setDashboardData,
    setJobs,
    setTheme,
    signIn,
    theme,
    user,
    viewReady
  }), [checkingSession, dashboardData, jobs, theme, user, viewReady]);

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}

export function useSession() {
  const context = useContext(SessionContext);
  if (!context) throw new Error("useSession must be used within SessionProvider");
  return context;
}
