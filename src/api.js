const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000";

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : await response.text();

  if (!response.ok) {
    if (response.status === 401 && typeof window !== "undefined") {
      window.dispatchEvent(new CustomEvent("datavista:unauthorized"));
    }

    const detail =
      typeof body === "object" && body !== null && "detail" in body
        ? body.detail
        : body || "Request failed";
    throw new Error(Array.isArray(detail) ? detail.map((item) => item.msg).join(", ") : detail);
  }

  return body;
}

function authHeaders(token) {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

export async function login(email, password) {
  const form = new URLSearchParams();
  form.set("username", email);
  form.set("password", password);

  return parseResponse(
    await fetch(`${API_BASE_URL}/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: form
    })
  );
}

export async function register({ email, password, username }) {
  return parseResponse(
    await fetch(`${API_BASE_URL}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password, username })
    })
  );
}

export async function getMe(token) {
  return parseResponse(
    await fetch(`${API_BASE_URL}/auth/me`, {
      headers: authHeaders(token)
    })
  );
}

export async function uploadDataset(file, token) {
  const form = new FormData();
  form.append("file", file);

  return parseResponse(
    await fetch(`${API_BASE_URL}/datasets/upload`, {
      method: "POST",
      headers: authHeaders(token),
      body: form
    })
  );
}

export async function getDatasetPreview(datasetId, token) {
  return parseResponse(
    await fetch(`${API_BASE_URL}/datasets/${datasetId}/preview`, {
      headers: authHeaders(token)
    })
  );
}

export async function getCleaningReport(datasetId, token) {
  return parseResponse(
    await fetch(`${API_BASE_URL}/datasets/${datasetId}/cleaning-report`, {
      headers: authHeaders(token)
    })
  );
}

export async function cleanDataset(datasetId, token) {
  return parseResponse(
    await fetch(`${API_BASE_URL}/datasets/${datasetId}/clean`, {
      method: "POST",
      headers: authHeaders(token)
    })
  );
}

export async function createAnalysisJob(payload, token) {
  return parseResponse(
    await fetch(`${API_BASE_URL}/analysis/jobs`, {
      method: "POST",
      headers: { ...authHeaders(token), "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
  );
}

export async function listAnalysisJobs(token) {
  return parseResponse(
    await fetch(`${API_BASE_URL}/analysis/jobs?limit=20&offset=0`, {
      headers: authHeaders(token)
    })
  );
}

export { API_BASE_URL };
