import { appUrl } from "./app-path.mjs?v=compact-2";
import {formatTaskLog} from "../assets/progress_log.js";

const $ = (selector) => document.querySelector(selector);
const statuses = {
  received: "已接收", queued: "排队中", running: "运行中", success: "成功",
  failed: "失败", rejected: "校验拒绝", cancelled: "已取消", interrupted: "已中断",
};
const state = { offset: 0, limit: 25, total: 0, token: "", selected: null };
let listController;
let detailController;

function node(tag, text, className) {
  const element = document.createElement(tag);
  if (text !== undefined) element.textContent = text;
  if (className) element.className = className;
  return element;
}

function date(value) {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? value : parsed.toLocaleString("zh-CN", { hour12: false });
}

function badge(status) {
  const element = node("span", statuses[status] || status || "未知", "status");
  element.dataset.status = status;
  return element;
}

async function api(path, signal) {
  const response = await fetch(appUrl(path), {
    signal, cache: "no-store",
    headers: state.token ? { Authorization: `Bearer ${state.token}` } : {},
  });
  const payload = await response.json().catch(() => ({}));
  if (response.status === 403) {
    $("#access-form").hidden = false;
    throw new Error("请使用管理员口令查看记录，或在服务器本机打开此页面。");
  }
  if (!response.ok) {
    throw new Error(typeof payload.detail === "string" ? payload.detail : `读取失败（${response.status}），请刷新重试。`);
  }
  return payload;
}

function pagination(loading = false) {
  $("#previous").disabled = loading || state.offset === 0;
  $("#next").disabled = loading || state.offset + state.limit >= state.total;
}

async function loadRecords() {
  listController?.abort();
  const controller = new AbortController();
  listController = controller;
  pagination(true);
  $("#message").className = "";
  $("#message").textContent = "正在读取使用记录…";
  const query = new URLSearchParams({
    query: $("#search").value.trim(), status: $("#status-filter").value,
    offset: state.offset, limit: state.limit,
  });
  try {
    const result = await api(`/api/usage?${query}`, controller.signal);
    if (controller.signal.aborted) return;
    state.total = result.total;
    if (state.offset >= state.total && state.offset > 0) {
      state.offset = Math.max(0, Math.floor((state.total - 1) / state.limit) * state.limit);
      return loadRecords();
    }
    $("#access-form").hidden = true;
    $("#records").replaceChildren();
    for (const record of result.items) {
      const row = node("tr");
      const submitted = node("td", date(record.created_at));
      submitted.append(node("span", record.id, "record-id"));
      const status = node("td");
      status.append(badge(record.status));
      const action = node("td");
      const button = node("button", "查看详情 ↗", "view-detail");
      button.type = "button";
      button.setAttribute("aria-label", `查看 ${record.id} 的详情`);
      button.addEventListener("click", () => showDetail(record.id));
      action.append(button);
      row.append(submitted, node("td", record.ip || "未获取", "ip"),
        node("td", record.model || "未指定"), node("td", record.satellite_count ?? "—"), status, action);
      $("#records").append(row);
    }
    const filtered = $("#search").value.trim() || $("#status-filter").value;
    $("#message").textContent = result.total
      ? `按提交时间倒序排列${result.skipped ? ` ${result.skipped} 个记录文件无法读取，已跳过` : ""}`
      : `${filtered ? "没有符合条件的记录，请调整筛选条件。" : "还没有使用记录。返回仿真页面提交一次任务后，点击刷新即可查看。"}${result.skipped ? ` ${result.skipped} 个文件无法读取。` : ""}`;
    $("#count").textContent = result.total
      ? `第 ${state.offset + 1}–${state.offset + result.items.length} 条，共 ${result.total} 条`
      : "共 0 条记录";
  } catch (error) {
    if (error.name === "AbortError") return;
    state.total = 0;
    $("#records").replaceChildren();
    $("#count").textContent = "记录未加载";
    $("#message").className = "error";
    $("#message").textContent = error.message;
  } finally {
    if (listController === controller) pagination();
  }
}

function section(title, value, open = false) {
  const details = node("details");
  details.open = open;
  details.append(node("summary", title));
  details.append(node("pre", typeof value === "string" ? value : JSON.stringify(value ?? null, null, 2)));
  return details;
}

async function showDetail(id) {
  detailController?.abort();
  const controller = new AbortController();
  detailController = controller;
  state.selected = null;
  $("#download").disabled = true;
  $("#download-message").textContent = "";
  $("#detail-content").replaceChildren(node("p", "正在读取完整记录…", "muted"));
  if (!$("#detail-dialog").open) $("#detail-dialog").showModal();
  try {
    const record = await api(`/api/usage/${encodeURIComponent(id)}`, controller.signal);
    if (controller.signal.aborted) return;
    state.selected = record;
    const submitted = record.submitted_configuration || {};
    const effective = record.effective_configuration || {};
    // V1 effective configuration has physical/cells, while the editor keeps
    // hardware/deployment. Label these sections as the submitted editor values.
    const config = effective.hardware ? effective : submitted;
    const grid = node("div", undefined, "summary-grid");
    const fields = [
      ["来源 IP", record.client?.ip || record.ip || "未获取"],
      ["模型", config.model || submitted.model || "未指定"],
      ["运行状态", statuses[record.status] || record.status],
      ["提交时间", date(record.created_at)], ["开始时间", date(record.started_at)],
      ["结束时间", date(record.finished_at)],
      ["卫星数量", config.hardware?.satellite_count ?? "—"],
      ["每星卡数", config.hardware?.accelerators_per_satellite ?? "—"],
      ["Worker pools", config.deployment?.worker_pools?.map(pool => `${pool.name}: ${pool.role}`).join("; ") || "—"],
    ];
    for (const [label, value] of fields) {
      const cell = node("div", undefined, "summary-cell");
      cell.append(node("span", label), node("strong", value));
      grid.append(cell);
    }
    const content = $("#detail-content");
    content.replaceChildren(node("p", record.id, "detail-id"), grid);
    content.append(node("h3", "执行日志"), node("pre", formatTaskLog(record)));
    if (!record.effective_configuration) content.append(node("p", "此提交未生成有效仿真配置，下方保留了原始提交及错误信息。", "muted"));
    content.append(
      section("提交硬件配置 物理清单 / 星内互联", config.hardware, true),
      section("V1 生效部署 物理设备 / Cell", {physical:effective.physical,cells:effective.cells}),
      section("卫星配置 编队 / 卫星 ID / 部署映射", {
        constellation: config.constellation,
        deployment: config.deployment,
      }),
      section("模型与负载 模型 / 并行维度 / Prompt / Decode", {
        model: config.model ?? submitted.model, requests: config.requests,
        scheduler: config.scheduler, runtime_inputs: config.runtime_inputs,
      }),
      section("提交链路配置 拓扑 / FSO / 传输", {
        topology: config.topology ?? submitted.links, fso: config.fso ?? submitted.fso,
        network: config.network, flowsim: config.flowsim,
      }),
    );
    for (const [key, source] of Object.entries(record.configuration_sources || {})) {
      let contents = source.content;
      try { contents = JSON.parse(contents); } catch { /* TLE stays plain text. */ }
      content.append(section(`配置快照 ${{ model: "完整模型结构", tle: "TLE 轨道数据", components: "组件版本" }[key] || key}`, contents));
    }
    content.append(
      section("原始提交 完整配置", record.submitted_configuration),
      section("后端实际配置 完整配置", record.effective_configuration),
      section("客户端与运行产物", { client: record.client, artifacts: record.artifacts, warnings: record.warnings }),
      section("完整记录 JSON", record),
    );
    $("#download").disabled = false;
  } catch (error) {
    if (error.name !== "AbortError") $("#detail-content").replaceChildren(node("p", error.message, "run-error"));
  }
}

$("#filters").addEventListener("submit", (event) => {
  event.preventDefault(); state.offset = 0; loadRecords();
});
$("#status-filter").addEventListener("change", () => { state.offset = 0; loadRecords(); });
$("#refresh").addEventListener("click", loadRecords);
$("#previous").addEventListener("click", () => { state.offset = Math.max(0, state.offset - state.limit); loadRecords(); });
$("#next").addEventListener("click", () => { state.offset += state.limit; loadRecords(); });
$("#access-form").addEventListener("submit", (event) => {
  event.preventDefault(); state.token = $("#access-token").value;
  $("#access-token").value = ""; state.offset = 0; loadRecords();
});
$("#close-detail").addEventListener("click", () => $("#detail-dialog").close());
$("#detail-dialog").addEventListener("close", () => {
  detailController?.abort(); state.selected = null;
});
$("#download").addEventListener("click", () => {
  if (!state.selected) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(state.selected, null, 2)], { type: "application/json;charset=utf-8" }));
  const anchor = node("a");
  anchor.href = url; anchor.download = `${state.selected.id}.json`;
  document.body.append(anchor); anchor.click(); anchor.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  $("#download-message").textContent = "已导出完整记录，包含来源 IP 和全部配置。";
});
loadRecords();
