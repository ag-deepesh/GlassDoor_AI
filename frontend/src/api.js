// Thin fetch client for the FastAPI backend (api/main.py). SSE is parsed by
// hand here rather than via the browser's EventSource, because EventSource
// can't send a POST body -- both /kbs/build and /query need one (files/config).
export const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export function kbImageUrl(kbName, imageId) {
  return `${API_BASE}/kbs/${encodeURIComponent(kbName)}/images/${encodeURIComponent(imageId)}`;
}

async function streamSSE(response, onEvent) {
  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`${response.status} ${response.statusText}: ${text}`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const events = buffer.split("\n\n");
    buffer = events.pop(); // last chunk may be a partial event -- keep it for next read
    for (const raw of events) {
      const line = raw.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      try {
        onEvent(JSON.parse(line.slice("data: ".length)));
      } catch {
        // malformed chunk boundary -- drop this one event, stream continues
      }
    }
  }
}

// Drop blank key fields rather than sending them as empty strings -- an
// empty string is a present-but-invalid key to the provider SDKs (a
// confusing auth error), whereas an absent key hits our own clean
// "missing API key" error path server-side.
function cleanKeys(keys) {
  return Object.fromEntries(Object.entries(keys || {}).filter(([, v]) => v));
}

// Extracts FastAPI's HTTPException detail message when present, so the UI
// can show "Missing gemini API key -- ..." instead of a bare status code.
async function errorMessage(res, fallback) {
  const text = await res.text().catch(() => "");
  try {
    const detail = JSON.parse(text)?.detail;
    if (detail) return typeof detail === "string" ? detail : JSON.stringify(detail);
  } catch {
    // not JSON -- fall through to the raw text/fallback below
  }
  return text || fallback;
}

export async function getRegistry() {
  const res = await fetch(`${API_BASE}/registry`);
  if (!res.ok) throw new Error(`Failed to load registry: ${res.status}`);
  return res.json();
}

// Which providers already have a usable key on the server (from its .env) --
// lets the UI skip asking the user for a key it doesn't actually need.
export async function getAvailableProviders() {
  const res = await fetch(`${API_BASE}/config/providers`);
  if (!res.ok) throw new Error(`Failed to load provider config: ${res.status}`);
  return res.json();
}

export async function getKbs() {
  const res = await fetch(`${API_BASE}/kbs`);
  if (!res.ok) throw new Error(`Failed to load KBs: ${res.status}`);
  return res.json();
}

export async function deleteKb(name) {
  const res = await fetch(`${API_BASE}/kbs/${encodeURIComponent(name)}`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Failed to delete KB '${name}': ${res.status}`);
  return res.json();
}

export async function buildKb({ name, files, config, apiKeys }, onEvent) {
  const form = new FormData();
  form.append("name", name);
  for (const f of files) form.append("files", f);
  form.append("api_keys", JSON.stringify(cleanKeys(apiKeys)));
  for (const [k, v] of Object.entries(config || {})) form.append(k, v);

  const res = await fetch(`${API_BASE}/kbs/build`, { method: "POST", body: form });
  await streamSSE(res, onEvent);
}

export async function runQuery(payload, onEvent) {
  const res = await fetch(`${API_BASE}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, api_keys: cleanKeys(payload.api_keys) }),
  });
  await streamSSE(res, onEvent);
}

export async function rewritePrompt({ prompt, provider, apiKey, model }) {
  const res = await fetch(`${API_BASE}/llm/rewrite-prompt`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, provider, api_key: apiKey, model }),
  });
  if (!res.ok) throw new Error(await errorMessage(res, `Rewrite failed: ${res.status}`));
  return res.json();
}

export async function suggestOption({ stage, options, context, provider, apiKey, model }) {
  const res = await fetch(`${API_BASE}/llm/suggest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stage, options, context, provider, api_key: apiKey, model }),
  });
  if (!res.ok) throw new Error(await errorMessage(res, `Suggest failed: ${res.status}`));
  return res.json();
}

export async function diagnoseDeep({ query, stageReports, provider, apiKey }) {
  const res = await fetch(`${API_BASE}/diagnose/deep`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, stage_reports: stageReports, provider, api_key: apiKey }),
  });
  if (!res.ok) throw new Error(await errorMessage(res, `Deep-dive failed: ${res.status}`));
  return res.json();
}
