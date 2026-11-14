import { appUrl } from "./app-path.mjs?v=compact-2";

export function displayNumber(value, unit = "") {
  return typeof value === "number" && Number.isFinite(value) ? `${value.toLocaleString("zh-CN", { maximumSignificantDigits: 6 })}${unit ? ` ${unit}` : ""}` : "不可用";
}

export function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function table(columns, rows) {
  const frame = element("div", undefined, "table-scroll");
  const node = element("table"); const head = element("thead"); const tr = element("tr");
  columns.forEach(column => tr.append(element("th", column))); head.append(tr); node.append(head);
  const body = element("tbody");
  rows.forEach(row => { const tr = element("tr"); row.forEach(value => tr.append(element("td", value))); body.append(tr); });
  node.append(body); frame.append(node); return frame;
}

export function renderNetworkPreview(container, result) {
  const optical = result.isl_links || [];
  const nodes = new Map(result.topology.nodes.map(node => [node.id, node.satellite_id]));
  const rows = optical.length ? optical.map(link => [
    `${link.source_satellite} → ${link.target_satellite}`, displayNumber(link.distance_km), displayNumber(link.bandwidth_gbps), displayNumber(link.latency_ns), link.feasible ? "可用" : link.limiting_reason || "不可用",
  ]) : result.topology.links.filter(link => link.level === "inter_satellite").map(link => [
    `${nodes.get(link.source)} → ${nodes.get(link.target)}`, displayNumber(link.distance_km), displayNumber(link.bandwidth_gbps), displayNumber(link.latency_ns), link.enabled ? "可用" : "停用",
  ]);
  container.replaceChildren(rows.length ? table(["星间链路", "距离 / km", "带宽 / Gbps", "传播时延 / ns", "状态"], rows) : element("p", "无星间链路", "upload-status"));
}

function pairs(value, path = "") {
  return Object.entries(value || {}).flatMap(([key, item]) => {
    const name = path ? `${path}.${key}` : key;
    return item !== null && typeof item === "object" && !Array.isArray(item) ? pairs(item, name) : [[name, typeof item === "number" ? displayNumber(item) : item == null ? "不可用" : String(item)]];
  });
}

export async function renderResults(container, jobId, result, api) {
  container.replaceChildren(element("p", jobId, "muted"));
  const metrics = result.metrics || {};
  const cards = element("div", undefined, "metric-grid");
  for (const [label, value, unit] of [
    ["仿真总时间", metrics.simulation?.wall_time_ns == null ? null : metrics.simulation.wall_time_ns / 1e9, "s"],
    ["调度步骤", metrics.simulation?.scheduler_steps, ""], ["Kernel 调用", metrics.simulation?.kernel_calls, ""], ["传输尝试", metrics.simulation?.flow_count, ""],
  ]) { const card = element("div"); card.append(element("small", label), element("strong", displayNumber(value, unit))); cards.append(card); }
  container.append(cards);
  const tabs = element("div", undefined, "result-tabs"); tabs.setAttribute("role", "tablist"); tabs.setAttribute("aria-label", "仿真结果分类");
  const viewport = element("div"); container.append(tabs, viewport);
  const panels = [];
  function addTab(title) {
    const index = panels.length, button = element("button", title), panel = element("section", undefined, "result-page");
    button.type = "button"; button.id = `result-tab-${index}`; panel.id = `result-page-${index}`;
    button.setAttribute("role", "tab"); button.setAttribute("aria-controls", panel.id); panel.setAttribute("role", "tabpanel"); panel.setAttribute("aria-labelledby", button.id);
    function select() { panels.forEach(p => { const selected = p.panel === panel; p.panel.hidden = !selected; p.button.classList.toggle("is-selected", selected); p.button.setAttribute("aria-selected", String(selected)); p.button.tabIndex = selected ? 0 : -1; }); }
    button.onclick = select;
    button.onkeydown = event => { if (["ArrowLeft", "ArrowRight"].includes(event.key)) { event.preventDefault(); const next = panels[(index + (event.key === "ArrowRight" ? 1 : -1) + panels.length) % panels.length].button; next.click(); next.focus(); } };
    tabs.append(button); viewport.append(panel); panels.push({button, panel}); if (!index) select(); else { panel.hidden = true; button.tabIndex = -1; button.setAttribute("aria-selected", "false"); }
    return panel;
  }
  const sections = [
    ["请求性能", metrics.serving, "时间单位为秒；TTFT / TPOT 保持引擎定义。PD 场景按终端产出 pool 统计。吞吐量分母为仿真时钟零点至最终完成，不是网页等待时间。"],
    ["计算执行", metrics.simulation, "wall_time_ns 是仿真时间（ns）。Kernel 资源耗时可能重叠，不能直接相加作为关键路径。"],
    ["网络通信", metrics.network_by_level, "V1 Runtime Comm 的原始运行统计。传输尝试计数来自 admit 事件；不把整条路径服务时间解释成独立逐跳完成时间。"],
    ["Host / KV", { host: metrics.simulation?.timing_mode === "device_only" ? "未启用（device_only）" : metrics.host, kv_offload: Object.keys(metrics.kv_offload || {}).length ? metrics.kv_offload : "未启用或没有记录" }, "主机与设备可能重叠，服务耗时不能直接相加作为总延迟。空 KV 数据不代表零耗时测量。"],
  ];
  for (const [title, value, note] of sections) {
    const section = addTab(title);
    const definition = element("details", undefined, "metric-definition");
    definition.append(element("summary", "指标说明"), element("p", note, "muted"));
    section.append(element("h3", title), definition);
    const summary = { ...value }; delete summary.requests;
    const rows = pairs(summary);
    section.append(rows.length ? table(["指标（原始字段保留单位）", "值"], rows) : element("p", "未启用或没有可用数据"));
    if (title === "请求性能") {
      const requests = Object.values(value?.requests || {}); let offset = 0;
      if (requests.length) section.append(requestLatencyChart(requests));
      const page = element("div"); const previous = element("button", "上一页"); const next = element("button", "下一页"); const status = element("span");
      function render() {
        page.replaceChildren(table(["请求 ID", "TTFT (s)", "TPOT (s)", "延迟 (s)"], requests.slice(offset, offset + 100).map(r => [r.request_id, displayNumber(r.ttft_s), displayNumber(r.tpot_s), displayNumber(r.latency_s)])));
        previous.disabled = offset === 0; next.disabled = offset + 100 >= requests.length;
        status.textContent = `${requests.length} 个请求 · ${requests.length ? offset + 1 : 0}–${Math.min(offset + 100, requests.length)}`;
      }
      previous.onclick = () => { offset -= 100; render(); }; next.onclick = () => { offset += 100; render(); }; render();
      previous.className = next.className = "small-action";
      section.append(page, previous, status, next);
    }
  }
  if (result.warnings?.length) container.append(element("pre", result.warnings.join("\n"), "warning"));
  const artifacts = addTab("产物与明细");
  artifacts.append(element("h3", "版本与完整结果"), element("pre", JSON.stringify({status:result.status, simulator:result.simulator, network_backend:result.network_backend, versions:result.versions}, null, 2)));
  const extra = Object.fromEntries(Object.entries(metrics).filter(([key]) => !["serving", "simulation", "network_by_level", "host", "kv_offload"].includes(key)));
  if (Object.keys(extra).length) artifacts.append(table(["其他指标", "值"], pairs(extra)));
  await renderArtifacts(artifacts, jobId, api);
}

function requestLatencyChart(requests) {
  const chart = element("div", undefined, "request-chart");
  const samples = requests.filter(r => Number.isFinite(r.latency_s)).slice(0, 100);
  chart.append(element("p", `请求延迟 / s · 前 ${samples.length} 个有效请求`, "muted"));
  if (!samples.length) return chart;
  const ns = "http://www.w3.org/2000/svg", svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", "0 0 800 150"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "请求端到端延迟柱状图；精确值见下方表格");
  const maximum = Math.max(...samples.map(r => r.latency_s), 1e-12), width = 760 / samples.length;
  samples.forEach((r, i) => {
    const bar = document.createElementNS(ns, "rect"); const height = r.latency_s / maximum * 115;
    for (const [key, value] of Object.entries({x:20+i*width,y:130-height,width:Math.max(1,width-4),height,fill:"var(--violet)"})) bar.setAttribute(key,String(value));
    const title = document.createElementNS(ns, "title"); title.textContent = `${r.request_id}: ${r.latency_s} s`; bar.append(title); svg.append(bar);
  });
  const label = document.createElementNS(ns,"text"); label.setAttribute("x","20"); label.setAttribute("y","12"); label.setAttribute("fill","var(--muted)"); label.setAttribute("font-size","10"); label.textContent = `最大 ${displayNumber(maximum, "s")}`; svg.append(label);
  chart.append(svg); return chart;
}

export async function renderArtifacts(container, jobId, api) {
  const details = element("section", undefined, "result-section");
  details.append(element("h3", "完整产物与执行明细"), element("p", "按需读取，每页最多 100 条；所有文件可逐项下载。时间线中的 rank 可通过 placement.json 映射至卫星与星内卡。", "muted"));
  container.append(details);
  try {
    const manifest = await api(`/api/simulations/${jobId}/artifacts`);
    const select = element("select", undefined, "result-select"); select.setAttribute("aria-label", "选择结果产物");
    for (const item of manifest.items) { const option = element("option", item.key); option.value = item.key; select.append(option); }
    if (!manifest.items.length) { details.append(element("p", "此历史任务没有可下载的产物清单")); return; }
    const download = element("a", "下载当前文件", "small-action");
    const previous = element("button", "上一页"); const next = element("button", "下一页"); const page = element("div"); const status = element("p", "", "muted"); let offset = 0; let version = 0;
    previous.className = next.className = "small-action";
    async function load() {
      const requestVersion = ++version;
      const url = `/api/simulations/${jobId}/artifact?key=${encodeURIComponent(select.value)}`;
      download.href = appUrl(`${url}&download=true`); previous.disabled = next.disabled = true; page.textContent = "正在读取…";
      try {
        const data = await api(`${url}&offset=${offset}&limit=100`);
        if (requestVersion !== version) return;
        if (data.items) {
          const columns = [...new Set(data.items.flatMap(item => Object.keys(item)))];
          page.replaceChildren(table(columns, data.items.map(item => columns.map(key => typeof item[key] === "object" && item[key] !== null ? JSON.stringify(item[key]) : item[key] ?? "不可用"))));
          status.textContent = `从第 ${offset + 1} 条起 · 本页 ${data.items.length} 条${data.preview_only ? ' · 嵌套长数据为预览，完整内容请下载' : ''}`;
        } else { page.replaceChildren(element("pre", JSON.stringify(data.value, null, 2))); status.textContent = data.preview_only ? "JSON 预览 · 完整内容请下载" : "完整 JSON"; }
        previous.disabled = offset === 0; next.disabled = !data.has_more;
      } catch (error) { if (requestVersion === version) page.textContent = error.message; }
    }
    select.onchange = () => { offset = 0; load(); }; previous.onclick = () => { offset -= 100; load(); }; next.onclick = () => { offset += 100; load(); };
    details.append(select, download, previous, next, status, page); await load();
  } catch (error) { details.append(element("p", `产物读取失败：${error.message}`, "error")); }
}
