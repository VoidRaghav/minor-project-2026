async function request(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return body;
}

async function upload(path, form) {
  const response = await fetch(`/api${path}`, { method: "POST", body: form });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : `Request failed (${response.status})`;
    throw new Error(detail);
  }
  return body;
}

export const api = {
  visionDomains: () => request("/vision/domains"),
  visionDomain: (id) => request(`/vision/domains/${id}`),
  visionSampleUrl: (id, name) => `/api/vision/domains/${id}/samples/${encodeURIComponent(name)}`,
  visionSample: async (id, name) => {
    const response = await fetch(`/api/vision/domains/${id}/samples/${encodeURIComponent(name)}`);
    if (!response.ok) throw new Error(`Could not load sample (${response.status})`);
    return response.blob();
  },
  visionPredict: (id, blob, filename) => {
    const form = new FormData();
    form.append("file", blob, filename);
    return upload(`/vision/domains/${id}/predict`, form);
  },
};
