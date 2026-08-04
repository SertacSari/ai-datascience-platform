const defaultApiBaseUrl =
  typeof window === "undefined"
    ? "http://127.0.0.1:8000"
    : `http://${window.location.hostname}:8000`;
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || defaultApiBaseUrl;
const LEGACY_TOKEN_KEY = "datavista_access_token";

export function clearLegacyStoredToken() {
  localStorage.removeItem(LEGACY_TOKEN_KEY);
  sessionStorage.removeItem(LEGACY_TOKEN_KEY);
}

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    if (response.status === 401 && typeof window !== "undefined") {
      clearLegacyStoredToken();
      window.dispatchEvent(new CustomEvent("datavista:unauthorized"));
    }

    const detail =
      typeof body === "object" && body !== null && "detail" in body
        ? body.detail
        : body || "Request failed";
    const error = new Error(Array.isArray(detail) ? detail.map((item) => item.msg).join(", ") : detail);
    error.status = response.status;
    throw error;
  }

  return body;
}

export async function login(email, password) {
  const form = new URLSearchParams();
  form.set("username", email);
  form.set("password", password);

  return parseResponse(
    await fetch(`${API_BASE_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      credentials: "include",
      body: form
    })
  );
}

export async function register({ email, password, username }) {
  return parseResponse(
    await fetch(`${API_BASE_URL}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify({ email, password, username })
    })
  );
}

export async function logout() {
  return parseResponse(
    await fetch(`${API_BASE_URL}/auth/logout`, {
      method: "POST",
      credentials: "include"
    })
  );
}

export async function getMe() {
  return parseResponse(
    await fetch(`${API_BASE_URL}/auth/me`, {
      credentials: "include"
    })
  );
}

export async function uploadDataset(file) {
  const form = new FormData();
  form.append("file", file);

  return parseResponse(
    await fetch(`${API_BASE_URL}/datasets/upload`, {
      method: "POST",
      credentials: "include",
      body: form
    })
  );
}

export async function getDatasetPreview(datasetId) {
  return parseResponse(
    await fetch(`${API_BASE_URL}/datasets/${datasetId}/preview`, {
      credentials: "include"
    })
  );
}

export async function getCleaningReport(datasetId) {
  return parseResponse(
    await fetch(`${API_BASE_URL}/datasets/${datasetId}/cleaning-report`, {
      credentials: "include"
    })
  );
}

export async function cleanDataset(datasetId) {
  return parseResponse(
    await fetch(`${API_BASE_URL}/datasets/${datasetId}/clean`, {
      method: "POST",
      credentials: "include"
    })
  );
}

export async function createAnalysisJob(payload) {
  return parseResponse(
    await fetch(`${API_BASE_URL}/analysis/jobs`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      credentials: "include",
      body: JSON.stringify(payload)
    })
  );
}

export async function runAnalysisJob(jobId) {
  return parseResponse(
    await fetch(`${API_BASE_URL}/analysis/jobs/${jobId}/run`, {
      method: "POST",
      credentials: "include"
    })
  );
}

export async function getAnalysisJobResult(jobId) {
  return parseResponse(
    await fetch(`${API_BASE_URL}/analysis/jobs/${jobId}/result`, {
      credentials: "include"
    })
  );
}

export async function listAnalysisJobs() {
  return parseResponse(
    await fetch(`${API_BASE_URL}/analysis/jobs?limit=20&offset=0`, {
      credentials: "include"
    })
  );
}

export { API_BASE_URL };
