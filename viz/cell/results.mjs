import { renderDashboard } from "./result-dashboard.mjs?v=mode-results-1";
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
  rows.forEach(row => { const tr = element("tr"); row.forEach((value, index) => {
    const cell = element("td", value);
    if (/^(request(_id)?|id)$/i.test(columns[index])) cell.className = "identifier";
    tr.append(cell);
  }); body.append(tr); });
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

export async function renderResults(container, jobId, result, api) {
  return renderDashboard(container, jobId, result, api, renderArtifacts);
}

export async function renderArtifacts(container, jobId, api) {
  const details = element("section", undefined, "result-section");
  details.append(element("h3", "完整产物与执行明细"), element("p", "按需读取，每页最多 100 条；所有文件可逐项下载。时间线中的 rank 可通过 placement.json 映射至卫星与星内卡。", "muted"));
  container.append(details);
  try {
    const manifest = await api(`/api/simulations/${jobId}/artifacts`);
    const select = element("select", undefined, "result-select"); select.setAttribute("aria-label", "选择结果产物");
    for (const item of manifest.items) {
      const option=element("option", item.key==='execution.log' ? '原始诊断日志（execution.log）' : item.key);
      option.value=item.key; select.append(option);
    }
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
          status.textContent = `从第 ${offset + 1} 条起 本页 ${data.items.length} 条${data.preview_only ? ' 嵌套长数据为预览，完整内容请下载' : ''}`;
        } else if (select.value.endsWith(".log") && typeof data.value === "string") {
          const raw = element("pre", data.value);
          raw.dataset.raw = "";
          page.replaceChildren(raw);
          status.textContent = "原始诊断日志预览，完整内容请下载。";
        } else { page.replaceChildren(element("pre", JSON.stringify(data.value, null, 2))); status.textContent = data.preview_only ? "JSON 预览 完整内容请下载" : "完整 JSON"; }
        previous.disabled = offset === 0; next.disabled = !data.has_more;
      } catch (error) { if (requestVersion === version) page.textContent = error.message; }
    }
    select.onchange = () => { offset = 0; load(); }; previous.onclick = () => { offset -= 100; load(); }; next.onclick = () => { offset += 100; load(); };
    details.append(select, download, previous, next, status, page); await load();
  } catch (error) { details.append(element("p", `产物读取失败：${error.message}`, "error")); }
}
