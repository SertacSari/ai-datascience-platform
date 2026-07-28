import { useEffect } from "react";
import {
  BrowserRouter,
  Navigate,
  Outlet,
  Route,
  Routes,
  useLocation,
  useNavigate
} from "react-router-dom";
import AppShell from "./components/AppShell";
import Brand from "./components/Brand";
import { SessionProvider, useSession } from "./context/SessionContext";
import DashboardPage from "./pages/DashboardPage";
import LoginPage from "./pages/LoginPage";

function UnauthorizedRedirect() {
  const navigate = useNavigate();
  const { logOut } = useSession();

  useEffect(() => {
    const onUnauthorized = () => {
      logOut();
      navigate("/login", { replace: true });
    };

    window.addEventListener("datavista:unauthorized", onUnauthorized);
    return () => window.removeEventListener("datavista:unauthorized", onUnauthorized);
  }, [logOut, navigate]);

  return null;
}

function LoadingScreen() {
  return (
    <main className="loading-screen">
      <Brand />
      <strong>Checking BasitAnaliz session...</strong>
    </main>
  );
}

function ProtectedLayout() {
  const location = useLocation();
  const { checkingSession, logOut, setTheme, theme, user, viewReady } = useSession();

  if (checkingSession || !viewReady) return <LoadingScreen />;
  if (!user) return <Navigate to="/login" replace state={{ from: location }} />;

  return (
    <AppShell
      activeView="dashboard"
      onLogOut={logOut}
      onTheme={() => setTheme((current) => current === "system" ? "light" : current === "light" ? "dark" : "system")}
      theme={theme}
      user={user}
    >
      <Outlet />
    </AppShell>
  );
}

function LoginRoute() {
  const { checkingSession, user, viewReady } = useSession();

  if (checkingSession || !viewReady) return <LoadingScreen />;
  if (user) return <Navigate to="/dashboard" replace />;
  return <LoginPage />;
}

function AppRoutes() {
  return (
    <>
      <UnauthorizedRedirect />
      <Routes>
        <Route path="/login" element={<LoginRoute />} />
        <Route element={<ProtectedLayout />}>
          <Route path="/dashboard" element={<DashboardPage />} />
        </Route>
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </>
  );
}

function App() {
  return (
    <BrowserRouter>
      <SessionProvider>
        <AppRoutes />
      </SessionProvider>
    </BrowserRouter>
  );
}

export default App;
