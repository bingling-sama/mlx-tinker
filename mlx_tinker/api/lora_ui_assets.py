"""No-build UI assets for the built-in LoRA web UI."""

LORA_UI_HTML = """<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>MLX-Tinker LoRAs</title>
    <link rel="stylesheet" href="/ui/assets/loras.css" />
  </head>
  <body>
    <div class="page-shell">
      <header class="page-header">
        <div>
          <p class="eyebrow">MLX-Tinker</p>
          <h1>LoRA Control Room</h1>
          <p class="subhead">
            Inspect live LoRAs, browse exported adapters, and download bundles without leaving the
            running server.
          </p>
        </div>
        <button id="refresh-button" class="ghost-button" type="button">Refresh</button>
      </header>

      <section id="summary-grid" class="summary-grid" aria-live="polite"></section>

      <section class="panel filters-panel">
        <div class="filters-row">
          <label>
            <span>Base Model</span>
            <select id="base-filter"></select>
          </label>
          <label>
            <span>Rank</span>
            <select id="rank-filter"></select>
          </label>
          <label>
            <span>State</span>
            <select id="state-filter">
              <option value="all">All</option>
              <option value="exported">Exported</option>
              <option value="live">Live only</option>
            </select>
          </label>
          <label>
            <span>Sort</span>
            <select id="sort-filter">
              <option value="recent">Recent activity</option>
              <option value="name">Name</option>
              <option value="rank">Rank</option>
              <option value="size">Size</option>
            </select>
          </label>
        </div>
      </section>

      <section class="layout">
        <main class="content-stack">
          <section class="panel">
            <div class="section-heading">
              <div>
                <p class="section-label">Live</p>
                <h2>In-memory models</h2>
              </div>
            </div>
            <div id="live-list" class="card-grid"></div>
          </section>

          <section class="panel">
            <div class="section-heading">
              <div>
                <p class="section-label">Exports</p>
                <h2>Saved adapters</h2>
              </div>
            </div>
            <div id="artifact-list" class="card-grid"></div>
          </section>
        </main>

        <aside class="panel detail-panel">
          <div class="section-heading">
            <div>
              <p class="section-label">Detail</p>
              <h2 id="detail-title">Choose a LoRA</h2>
            </div>
          </div>
          <div id="detail-body" class="detail-body empty-state">
            Select any live or exported LoRA to inspect its config, training stats, and artifact
            contents.
          </div>
        </aside>
      </section>
    </div>

    <script src="/ui/assets/loras.js" defer></script>
  </body>
</html>
"""

LORA_UI_CSS = """:root {
  --bg: #f6efe6;
  --panel: rgba(255, 250, 242, 0.94);
  --panel-strong: #fffdf8;
  --ink: #1f1d1a;
  --muted: #675f56;
  --line: rgba(72, 53, 28, 0.12);
  --accent: #b85724;
  --accent-soft: rgba(184, 87, 36, 0.12);
  --good: #2e6f4e;
  --good-soft: rgba(46, 111, 78, 0.12);
  --shadow: 0 18px 40px rgba(74, 49, 18, 0.09);
  --radius: 22px;
  --mono: "SFMono-Regular", "SF Mono", Menlo, Consolas, monospace;
  --sans: "Iowan Old Style", "Palatino Linotype", Georgia, serif;
}

* {
  box-sizing: border-box;
}

body {
  margin: 0;
  background:
    radial-gradient(circle at top left, rgba(184, 87, 36, 0.14), transparent 28%),
    radial-gradient(circle at top right, rgba(46, 111, 78, 0.10), transparent 24%),
    linear-gradient(180deg, #fbf6ee 0%, var(--bg) 46%, #efe5d7 100%);
  color: var(--ink);
  font-family: var(--sans);
  min-height: 100vh;
}

button,
select {
  font: inherit;
}

.page-shell {
  max-width: 1520px;
  margin: 0 auto;
  padding: 28px;
}

.page-header,
.filters-row,
.section-heading,
.card-head,
.stat-row,
.chip-row,
.detail-grid,
.detail-actions,
.future-row,
.file-row,
.session-row {
  display: flex;
  gap: 12px;
}

.page-header,
.section-heading,
.card-head,
.stat-row,
.future-row,
.file-row,
.session-row {
  align-items: center;
  justify-content: space-between;
}

.eyebrow,
.section-label {
  margin: 0 0 8px;
  text-transform: uppercase;
  letter-spacing: 0.22em;
  font: 600 0.7rem/1.2 var(--mono);
  color: var(--accent);
}

.page-header h1,
.section-heading h2,
.detail-panel h2 {
  margin: 0;
}

.page-header h1 {
  font-size: clamp(2.4rem, 4vw, 4.2rem);
  line-height: 0.95;
}

.subhead {
  max-width: 760px;
  margin: 14px 0 0;
  color: var(--muted);
  font-size: 1rem;
}

.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 20px;
  backdrop-filter: blur(8px);
}

.summary-grid,
.card-grid,
.layout {
  display: grid;
  gap: 18px;
}

.summary-grid {
  grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  margin: 22px 0 18px;
}

.summary-card {
  background: var(--panel-strong);
  border: 1px solid var(--line);
  border-radius: 18px;
  padding: 16px;
}

.summary-card p {
  margin: 0;
  color: var(--muted);
  font-size: 0.92rem;
}

.summary-card strong {
  display: block;
  margin-top: 6px;
  font-size: 1.7rem;
  line-height: 1;
}

.filters-row {
  align-items: end;
  flex-wrap: wrap;
}

.filters-row label {
  min-width: 170px;
  flex: 1 1 170px;
}

.filters-row span {
  display: block;
  margin-bottom: 8px;
  color: var(--muted);
  font: 600 0.78rem/1.3 var(--mono);
  text-transform: uppercase;
  letter-spacing: 0.08em;
}

select,
.ghost-button,
.action-button {
  appearance: none;
  border-radius: 14px;
  border: 1px solid var(--line);
  background: var(--panel-strong);
  color: var(--ink);
  padding: 11px 14px;
}

.ghost-button,
.action-button {
  cursor: pointer;
}

.ghost-button:hover,
.action-button:hover {
  border-color: rgba(184, 87, 36, 0.45);
}

.layout {
  grid-template-columns: minmax(0, 2fr) minmax(320px, 0.95fr);
  align-items: start;
}

.content-stack {
  display: grid;
  gap: 18px;
}

.card-grid {
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
}

.lora-card {
  border: 1px solid var(--line);
  border-radius: 18px;
  background: var(--panel-strong);
  padding: 16px;
}

.lora-card h3 {
  margin: 0;
  font-size: 1.05rem;
}

.small-text,
.mono-text,
.detail-label,
.empty-state,
.meta-list {
  color: var(--muted);
}

.small-text {
  font-size: 0.9rem;
}

.mono-text,
.detail-label {
  font-family: var(--mono);
  font-size: 0.8rem;
}

.mono-text {
  word-break: break-all;
}

.chip-row {
  flex-wrap: wrap;
  margin: 12px 0;
}

.chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  border-radius: 999px;
  padding: 6px 10px;
  background: var(--accent-soft);
  color: var(--accent);
  font: 600 0.75rem/1 var(--mono);
}

.chip.good {
  background: var(--good-soft);
  color: var(--good);
}

.meta-list {
  display: grid;
  gap: 8px;
  margin: 0;
}

.meta-list strong {
  color: var(--ink);
}

.card-actions,
.detail-actions {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 16px;
}

.action-button.primary {
  background: var(--accent);
  color: #fff8f1;
  border-color: var(--accent);
}

.action-button.secondary {
  background: transparent;
}

.detail-body {
  display: grid;
  gap: 16px;
}

.detail-grid {
  flex-wrap: wrap;
}

.detail-metric {
  flex: 1 1 140px;
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 12px;
  background: var(--panel-strong);
}

.detail-metric strong {
  display: block;
  margin-top: 6px;
  font-size: 1.1rem;
}

.detail-block {
  border-top: 1px solid var(--line);
  padding-top: 14px;
}

.detail-block h3 {
  margin: 0 0 10px;
}

.future-list,
.file-list,
.session-list {
  display: grid;
  gap: 8px;
}

.future-row,
.file-row,
.session-row {
  padding: 10px 12px;
  border-radius: 14px;
  border: 1px solid var(--line);
  background: var(--panel-strong);
}

.empty-state {
  border: 1px dashed rgba(72, 53, 28, 0.2);
  border-radius: 16px;
  padding: 18px;
  text-align: center;
  background: rgba(255, 253, 248, 0.72);
}

.loading {
  opacity: 0.65;
  pointer-events: none;
}

@media (max-width: 980px) {
  .layout {
    grid-template-columns: 1fr;
  }

  .detail-panel {
    order: -1;
  }
}

@media (max-width: 640px) {
  .page-shell {
    padding: 18px;
  }

  .page-header {
    align-items: start;
    flex-direction: column;
  }
}
"""

LORA_UI_JS = """const state = {
  items: [],
  filteredArtifacts: [],
  filteredLive: [],
};

const els = {
  summaryGrid: document.getElementById("summary-grid"),
  liveList: document.getElementById("live-list"),
  artifactList: document.getElementById("artifact-list"),
  detailTitle: document.getElementById("detail-title"),
  detailBody: document.getElementById("detail-body"),
  refreshButton: document.getElementById("refresh-button"),
  baseFilter: document.getElementById("base-filter"),
  rankFilter: document.getElementById("rank-filter"),
  stateFilter: document.getElementById("state-filter"),
  sortFilter: document.getElementById("sort-filter"),
};

function boot() {
  els.refreshButton.addEventListener("click", () => loadCatalog());
  for (const key of ["baseFilter", "rankFilter", "stateFilter", "sortFilter"]) {
    els[key].addEventListener("change", renderLists);
  }
  loadCatalog();
}

async function loadCatalog() {
  setLoading(true);
  try {
    const response = await fetch("/api/v1/loras");
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
    const payload = await response.json();
    state.items = payload.items || [];
    renderSummary(payload.summary || {});
    populateFilters(state.items);
    renderLists();
  } catch (error) {
    renderError(error);
  } finally {
    setLoading(false);
  }
}

function setLoading(isLoading) {
  document.body.classList.toggle("loading", isLoading);
}

function renderSummary(summary) {
  const cards = [
    ["Exported", summary.total_exported_loras || 0],
    ["Live", summary.live_loras || 0],
    ["Base models", summary.unique_base_models || 0],
    ["Adapter size", formatBytes(summary.total_adapter_disk_usage_bytes || 0)],
    ["Optim steps", summary.total_optim_steps || 0],
    ["Last activity", formatDate(summary.last_activity_at)],
  ];
  els.summaryGrid.innerHTML = cards
    .map(
      ([label, value]) => `
        <article class="summary-card">
          <p>${escapeHtml(String(label))}</p>
          <strong>${escapeHtml(String(value))}</strong>
        </article>
      `,
    )
    .join("");
}

function populateFilters(items) {
  const baseModels = Array.from(new Set(items.map((item) => item.base_model).filter(Boolean))).sort();
  const ranks = Array.from(
    new Set(
      items
        .map((item) => item.lora_config && item.lora_config.rank)
        .filter((rank) => rank !== undefined && rank !== null),
    ),
  ).sort((a, b) => Number(a) - Number(b));

  applyOptions(els.baseFilter, "All base models", baseModels);
  applyOptions(els.rankFilter, "All ranks", ranks);
}

function applyOptions(select, label, values) {
  const current = select.value;
  select.innerHTML = [`<option value="all">${escapeHtml(label)}</option>`]
    .concat(values.map((value) => `<option value="${escapeAttr(String(value))}">${escapeHtml(String(value))}</option>`))
    .join("");
  if ([...select.options].some((option) => option.value === current)) {
    select.value = current;
  }
}

function renderLists() {
  const filtered = filterAndSortItems(state.items);
  state.filteredLive = filtered.filter((item) => item.is_live && !item.is_exported);
  state.filteredArtifacts = filtered.filter((item) => item.is_exported);
  renderCollection(els.liveList, state.filteredLive, "No live in-memory LoRAs found.");
  renderCollection(els.artifactList, state.filteredArtifacts, "No exported LoRAs matched the current filters.");
}

function filterAndSortItems(items) {
  const baseValue = els.baseFilter.value;
  const rankValue = els.rankFilter.value;
  const stateValue = els.stateFilter.value;
  const sortValue = els.sortFilter.value;

  return [...items]
    .filter((item) => {
      if (baseValue !== "all" && item.base_model !== baseValue) {
        return false;
      }
      const itemRank = item.lora_config && item.lora_config.rank;
      if (rankValue !== "all" && String(itemRank) !== rankValue) {
        return false;
      }
      if (stateValue === "live" && !(item.is_live && !item.is_exported)) {
        return false;
      }
      if (stateValue === "exported" && !item.is_exported) {
        return false;
      }
      return true;
    })
    .sort((left, right) => compareItems(left, right, sortValue));
}

function compareItems(left, right, sortValue) {
  if (sortValue === "name") {
    return left.display_name.localeCompare(right.display_name);
  }
  if (sortValue === "rank") {
    return Number(right.lora_config?.rank || 0) - Number(left.lora_config?.rank || 0);
  }
  if (sortValue === "size") {
    return Number(right.size_bytes || 0) - Number(left.size_bytes || 0);
  }
  const leftDate = Date.parse(left.stats?.last_activity_at || left.created_at || 0);
  const rightDate = Date.parse(right.stats?.last_activity_at || right.created_at || 0);
  return rightDate - leftDate;
}

function renderCollection(target, items, emptyMessage) {
  if (!items.length) {
    target.innerHTML = `<div class="empty-state">${escapeHtml(emptyMessage)}</div>`;
    return;
  }

  target.innerHTML = items.map(renderCard).join("");
  target.querySelectorAll("[data-action='detail']").forEach((button) => {
    button.addEventListener("click", () => loadDetail(button.dataset.id));
  });
  target.querySelectorAll("[data-action='copy']").forEach((button) => {
    button.addEventListener("click", () => copyText(button.dataset.value));
  });
  target.querySelectorAll("[data-action='download']").forEach((button) => {
    button.addEventListener("click", () => {
      window.location.href = `/api/v1/loras/${encodeURIComponent(button.dataset.id)}/download`;
    });
  });
  target.querySelectorAll("[data-action='export-download']").forEach((button) => {
    button.addEventListener("click", async () => {
      await exportLiveAndDownload(button.dataset.modelId, button);
    });
  });
}

function renderCard(item) {
  const stats = item.stats || {};
  const chips = [
    item.is_live && !item.is_exported ? '<span class="chip good">live</span>' : "",
    item.is_exported ? '<span class="chip">exported</span>' : "",
    item.lora_config?.rank ? `<span class="chip">rank ${escapeHtml(String(item.lora_config.rank))}</span>` : "",
    item.status ? `<span class="chip">${escapeHtml(item.status)}</span>` : "",
  ]
    .filter(Boolean)
    .join("");

  const primaryAction = item.is_live && !item.is_exported
    ? `<button class="action-button primary" data-action="export-download" data-model-id="${escapeAttr(item.openai_model_id)}" type="button">Export + Download</button>`
    : item.downloadable
      ? `<button class="action-button primary" data-action="download" data-id="${escapeAttr(item.id)}" type="button">Download zip</button>`
      : "";

  return `
    <article class="lora-card">
      <div class="card-head">
        <h3>${escapeHtml(item.display_name)}</h3>
        <span class="mono-text">${escapeHtml(formatDate(stats.last_activity_at || item.created_at))}</span>
      </div>
      <div class="chip-row">${chips}</div>
      <div class="meta-list">
        <div><strong>Base:</strong> ${escapeHtml(item.base_model)}</div>
        <div><strong>Size:</strong> ${escapeHtml(formatBytes(item.size_bytes || 0))}</div>
        <div><strong>OpenAI model id:</strong> <span class="mono-text">${escapeHtml(item.openai_model_id)}</span></div>
        <div><strong>Optim steps:</strong> ${escapeHtml(String(stats.optim_step_count || 0))}</div>
        <div><strong>Last loss:</strong> ${escapeHtml(formatNumber(stats.last_loss))}</div>
      </div>
      <div class="card-actions">
        ${primaryAction}
        <button class="action-button secondary" data-action="detail" data-id="${escapeAttr(item.id)}" type="button">Details</button>
        <button class="action-button secondary" data-action="copy" data-value="${escapeAttr(item.openai_model_id)}" type="button">Copy model id</button>
      </div>
    </article>
  `;
}

async function loadDetail(id) {
  els.detailTitle.textContent = "Loading...";
  els.detailBody.innerHTML = '<div class="empty-state">Loading detail...</div>';
  try {
    const response = await fetch(`/api/v1/loras/${encodeURIComponent(id)}`);
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
    const detail = await response.json();
    renderDetail(detail);
  } catch (error) {
    els.detailTitle.textContent = "Unavailable";
    els.detailBody.innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  }
}

function renderDetail(detail) {
  const item = detail.item;
  const stats = item.stats || {};
  els.detailTitle.textContent = item.display_name;
  els.detailBody.innerHTML = `
    <div class="detail-grid">
      ${renderMetric("Forward/backward", stats.forward_backward_count || 0)}
      ${renderMetric("Optim steps", stats.optim_step_count || 0)}
      ${renderMetric("Samples", stats.sample_count || 0)}
      ${renderMetric("Exports", stats.sampler_export_count || 0)}
      ${renderMetric("Last loss", formatNumber(stats.last_loss))}
      ${renderMetric("Avg loss", formatNumber(stats.avg_loss))}
      ${renderMetric("Tokens", formatInteger(stats.total_tokens || 0))}
      ${renderMetric("Avg grad", formatNumber(stats.avg_grad_norm))}
    </div>

    <div class="detail-actions">
      <button class="action-button secondary" data-detail-copy="${escapeAttr(item.openai_model_id)}" type="button">Copy model id</button>
      ${
        item.is_live && !item.is_exported
          ? `<button class="action-button primary" data-detail-export="${escapeAttr(item.openai_model_id)}" type="button">Export + Download</button>`
          : item.downloadable
            ? `<button class="action-button primary" data-detail-download="${escapeAttr(item.id)}" type="button">Download zip</button>`
            : ""
      }
    </div>

    <div class="detail-block">
      <h3>Artifact</h3>
      <div class="meta-list">
        <div><strong>Status:</strong> ${escapeHtml(item.status)}</div>
        <div><strong>Created:</strong> ${escapeHtml(formatDate(item.created_at))}</div>
        <div><strong>Relative path:</strong> <span class="mono-text">${escapeHtml(item.relative_path || "n/a")}</span></div>
        <div><strong>Base model:</strong> ${escapeHtml(item.base_model)}</div>
      </div>
    </div>

    <div class="detail-block">
      <h3>LoRA config</h3>
      <pre class="mono-text">${escapeHtml(JSON.stringify(item.lora_config || {}, null, 2))}</pre>
    </div>

    <div class="detail-block">
      <h3>Session</h3>
      ${renderSessions(detail.session, detail.sampling_sessions || [])}
    </div>

    <div class="detail-block">
      <h3>Recent futures</h3>
      ${renderFutures(detail.recent_futures || [])}
    </div>

    <div class="detail-block">
      <h3>Files</h3>
      ${renderFiles(detail.files || [])}
    </div>
  `;

  const copyButton = els.detailBody.querySelector("[data-detail-copy]");
  if (copyButton) {
    copyButton.addEventListener("click", () => copyText(copyButton.dataset.detailCopy));
  }
  const downloadButton = els.detailBody.querySelector("[data-detail-download]");
  if (downloadButton) {
    downloadButton.addEventListener("click", () => {
      window.location.href = `/api/v1/loras/${encodeURIComponent(downloadButton.dataset.detailDownload)}/download`;
    });
  }
  const exportButton = els.detailBody.querySelector("[data-detail-export]");
  if (exportButton) {
    exportButton.addEventListener("click", async () => {
      await exportLiveAndDownload(exportButton.dataset.detailExport, exportButton);
    });
  }
}

function renderMetric(label, value) {
  return `
    <div class="detail-metric">
      <span class="detail-label">${escapeHtml(String(label))}</span>
      <strong>${escapeHtml(String(value))}</strong>
    </div>
  `;
}

function renderSessions(session, samplingSessions) {
  const rows = [];
  if (session) {
    rows.push(`
      <div class="session-row">
        <strong>${escapeHtml(session.session_id)}</strong>
        <span class="small-text">${escapeHtml(formatDate(session.last_heartbeat_at || session.created_at))}</span>
      </div>
    `);
  }
  for (const sampling of samplingSessions) {
    rows.push(`
      <div class="session-row">
        <strong>${escapeHtml(sampling.sampling_session_id)}</strong>
        <span class="small-text">${escapeHtml(formatDate(sampling.created_at))}</span>
      </div>
    `);
  }
  return rows.length ? `<div class="session-list">${rows.join("")}</div>` : '<div class="empty-state">No session records.</div>';
}

function renderFutures(futures) {
  if (!futures.length) {
    return '<div class="empty-state">No recent futures.</div>';
  }
  return `
    <div class="future-list">
      ${futures
        .map(
          (future) => `
            <div class="future-row">
              <strong>${escapeHtml(future.request_type)}</strong>
              <span class="small-text">${escapeHtml(formatDate(future.completed_at || future.created_at))}</span>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderFiles(files) {
  if (!files.length) {
    return '<div class="empty-state">No persisted files for this item.</div>';
  }
  return `
    <div class="file-list">
      ${files
        .map(
          (file) => `
            <div class="file-row">
              <strong>${escapeHtml(file.name)}</strong>
              <span class="small-text">${escapeHtml(formatBytes(file.size_bytes))}</span>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

async function exportLiveAndDownload(modelId, button) {
  const original = button.textContent;
  button.disabled = true;
  button.textContent = "Exporting...";
  try {
    const response = await fetch(`/api/v1/loras/live/${encodeURIComponent(modelId)}/export`, {
      method: "POST",
    });
    if (!response.ok) {
      throw new Error(`Export failed: ${response.status}`);
    }
    const item = await response.json();
    await loadCatalog();
    window.location.href = `/api/v1/loras/${encodeURIComponent(item.id)}/download`;
  } catch (error) {
    window.alert(error.message);
  } finally {
    button.disabled = false;
    button.textContent = original;
  }
}

function renderError(error) {
  const message = error && error.message ? error.message : "Unknown error";
  els.summaryGrid.innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
  els.liveList.innerHTML = "";
  els.artifactList.innerHTML = "";
}

async function copyText(value) {
  try {
    await navigator.clipboard.writeText(value);
  } catch (_error) {
    window.prompt("Copy model id", value);
  }
}

function formatBytes(value) {
  const size = Number(value || 0);
  if (!size) {
    return "0 B";
  }
  const units = ["B", "KB", "MB", "GB", "TB"];
  const power = Math.min(Math.floor(Math.log(size) / Math.log(1024)), units.length - 1);
  const scaled = size / 1024 ** power;
  return `${scaled >= 10 ? scaled.toFixed(0) : scaled.toFixed(1)} ${units[power]}`;
}

function formatDate(value) {
  if (!value) {
    return "n/a";
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.valueOf())) {
    return value;
  }
  return parsed.toLocaleString();
}

function formatNumber(value) {
  if (value === null || value === undefined || value === "") {
    return "n/a";
  }
  return Number(value).toFixed(3);
}

function formatInteger(value) {
  return Number(value || 0).toLocaleString();
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function escapeAttr(value) {
  return escapeHtml(value);
}

boot();
"""
