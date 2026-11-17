import { AsterMindScene, sceneState, syncSceneConfiguration } from "./scene.js?v=profile-1";
import { parseConfiguration } from "./config-form.mjs?v=validation-1";
import { buildWorkspace, renderWorkspaceForms } from "./workspace-ui.mjs?v=profile-1";
import { stages, stageNames, importResourceHardware, validationTarget } from "./config-layout.mjs?v=compact-2";
import { editDependencies, bindRequestIdentity, newWebConfiguration, networkInputs } from "./config-edit.mjs?v=validation-1";
import { renderResults, renderArtifacts, renderNetworkPreview, element } from "./results.mjs?v=compact-2";
import { appUrl, appStorageKey } from "./app-path.mjs?v=compact-2";
import { sceneData } from "./scene-data.mjs?v=compact-2";
import { configurationErrors, validationError } from "./config-validation.mjs?v=validation-1";

const $ = selector => document.querySelector(selector);
buildWorkspace();
const statuses = { queued: "排队", running: "运行中", success: "成功", failed: "失败", interrupted: "已中断", cancelled: "已取消", rejected: "校验失败" };
const phases = { queued: "等待执行", configuration: "解析配置", topology: "构建网络", serving: "请求调度与计算 / 通信仿真", completed: "完成" };
let config, capabilities, scene, dirty = false, busy = false, jobId = null, pollVersion = 0, previewVersion = 0, refreshForms = false;
let previewTimer, previewAbort, previewKey = "", networkPreview = null;
const drafts = new Map();

function draftChanged() {
  if(drafts.size) {
    ++previewVersion; previewKey=''; clearTimeout(previewTimer); previewAbort?.abort();
    $('#scene-links').textContent='请修正输入 · 预览未更新';
  } else if(config) { message(''); scheduleNetworkPreview(); }
}

function scheduleNetworkPreview() {
  if(drafts.size || ['formation','network'].some(name=>configurationErrors(config,capabilities.schema,name).length)) {
    ++previewVersion; previewKey=''; clearTimeout(previewTimer); previewAbort?.abort();
    $('#scene-links').textContent='请修正输入 · 预览未更新'; return;
  }
  const payload = networkInputs(config), key = JSON.stringify(payload);
  if (key === previewKey) return;
  previewKey = key; networkPreview = null;
  $("#scene-links").textContent = "正在核对链路…";
  const version = ++previewVersion;
  clearTimeout(previewTimer); previewAbort?.abort();
  $("#topology-preview").replaceChildren();
  $("#preview-note").textContent = "正在计算链路…";
  previewTimer = setTimeout(async () => {
    if (dirty || version !== previewVersion) return;
    previewAbort = new AbortController();
    try {
      const result = await api("/api/topology/preview", {method:"POST", headers:{"Content-Type":"application/json"}, body:key, signal:previewAbort.signal});
      if (dirty || version !== previewVersion) return;
      networkPreview = result;
      const edges = sceneData(config,result).edges, reached = new Set([0]);
      for (let pass=0;pass<config.hardware.satellite_ids.length;pass++) for (const [a,b] of edges) {
        if (reached.has(a)) reached.add(b); if (reached.has(b)) reached.add(a);
      }
      $("#scene-links").textContent = `${edges.length} 条链路 · ${reached.size === config.hardware.satellite_ids.length ? "全部接入" : "存在未接入卫星"}`;
      if (scene) syncSceneConfiguration(scene, config, result);
      $("#preview-note").textContent = "";
      renderNetworkPreview($("#topology-preview"), result);
    } catch (error) {
      if (version === previewVersion && error.name !== "AbortError") {
        $("#scene-links").textContent = "链路检查失败";
        $("#preview-note").textContent = `链路预测未完成：${error.message}`;
      }
    }
  }, 400);
}

let messageTimer;
function message(text, error = false) {
  clearTimeout(messageTimer);
  $("#web-message").textContent = document.body.dataset.stage === "home" && !error ? "" : text;
  $("#web-message").classList.toggle("is-error", error);
  if (text && !error) messageTimer = setTimeout(() => { $("#web-message").replaceChildren(); }, 4000);
}
function showError(error) {
  message(error.message, true);
  const target = validationTarget(error.detail);
  if (!target) return;
  if(target.stage===document.body.dataset.stage) {
    focusField(target); return;
  }
  const action = element("button", `前往${stageNames[stages.indexOf(target.stage)]}`, "small-action");
  action.onclick = () => {
    stage(target.stage,null,true); focusField(target);
  };
  $("#web-message").append(document.createTextNode("\n"), action);
}
function focusField(target) {
    const candidates = [...document.querySelectorAll(`.config-panel[data-panel="${target.stage}"] [data-field-path]`)];
    const field = candidates.find(e => JSON.stringify(target.path) === e.dataset.fieldPath) || candidates.find(e => target.path.every((part, i) => JSON.parse(e.dataset.fieldPath)[i] === part));
    if (!field) return;
    for (let parent = field.parentElement; parent; parent = parent.parentElement) if (parent.tagName === "DETAILS") parent.open = true;
    field.classList.add("field-error"); field.scrollIntoView({block:"center"}); field.querySelector("input,select,button")?.focus();
}
function stored(key, fallback) { try { return JSON.parse(localStorage.getItem(appStorageKey(key))) ?? fallback; } catch { return fallback; } }
function save(key, value) { try { localStorage.setItem(appStorageKey(key), JSON.stringify(value)); } catch { message("浏览器存储不可用，刷新前请保存任务链接和配置。", true); } }

async function api(path, options = {}) {
  const response = await fetch(appUrl(path), { cache: "no-store", ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map(e => `${(e.loc || []).join(".")}: ${e.msg}`).join("\n") : data.detail;
    const error = new Error(detail || `请求失败 (${response.status})`); error.detail = data.detail; throw error;
  }
  return data;
}
const post = (path, payload) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });

function validateFormInputs(name = null) {
  if(dirty) throw new Error('请先应用或放弃 JSON 编辑');
  const pending=[...drafts].map(([key,value])=>({loc:JSON.parse(key),msg:value.error}));
  if(pending.length) throw validationError(pending);
  const selector=name ? `.config-panel[data-panel="${name}"] .basic-fields` : '.basic-fields';
  for (const input of document.querySelectorAll(`${selector} input[data-config-path], ${selector} select[data-config-path]`)) {
    if (!input.checkValidity()) { const error = new Error(`输入格式或范围无效：${input.labels?.[0]?.textContent || input.id}`); error.detail = [{loc:JSON.parse(input.dataset.configPath || "[]"),msg:error.message}]; throw error; }
  }
  const error=validationError(configurationErrors(config,capabilities.schema,name));
  if(error) throw error;
}

function lockFields() {
  document.querySelectorAll('[data-action=start-config]').forEach(control => { control.disabled = !config; });
  document.querySelectorAll('.chapter-rail button, .cell-overview, #show-history, .orbit-node-action').forEach(control => { control.disabled = !config; });
  document.querySelectorAll(".basic-fields").forEach(field => { field.disabled = dirty || busy; });
  document.querySelectorAll("input[type=file], #run-simulation, #export-config, #bind-request-identity").forEach(control => { control.disabled = dirty || busy || !config; });
  $("#advanced-config").disabled = busy;
  $("#apply-json").disabled = $("#discard-json").disabled = busy;
}

function changed(path) {
  scheduleNetworkPreview();
  const layout = config.topology.source === "orbit" ? "轨道快照" : config.constellation?.formation?.layout === "chain" ? "链式编队" : "环形编队";
  $("#scene-summary").textContent = `${layout} · ${config.hardware.satellite_ids.length} 颗卫星`;
  $("#advanced-config").value = JSON.stringify(config, null, 2);
  save("astermind-config", config);
  if (scene) syncSceneConfiguration(scene, config, networkPreview);
  if (refreshForms) { refreshForms = false; renderForms(); }
}

function renderForms() {
  renderWorkspaceForms(config, capabilities, changed, (path, value, previous) => {
    refreshForms = editDependencies(config, path, value, capabilities, previous);
  }, text => message(text, true), {drafts,onDraftChange:draftChanged});
  $("#advanced-config").value = JSON.stringify(config, null, 2); lockFields();
  scheduleNetworkPreview();
}

function stage(name, anchor = null, revealError = false) {
  const current=document.body.dataset.stage;
  const advancing=stages.includes(name) && stages.indexOf(name)>stages.indexOf(current);
  if(config && !revealError && current==='home' && drafts.size) {
    const [path,draft]=drafts.entries().next().value;
    const error=validationError([{loc:JSON.parse(path),msg:draft.error}]);
    const owner=validationTarget(error.detail)?.stage;
    if(stages.indexOf(name)>stages.indexOf(owner)) { showError(error); return false; }
  }
  if(config && !revealError && stages.includes(current) && current!=='analysis' && advancing) {
    try { validateFormInputs(current); } catch(error) { showError(error); return false; }
  }
  message("");
  const sceneStage = {network:"terminal", workload:"inference", deployment:"hardware", runtime:"inference"}[name] || name;
  if (scene && !["home", "analysis"].includes(sceneStage) && ["home", "analysis"].includes(sceneState.stage)) scene.prepareConfigAnchor(anchor);
  sceneState.stage = sceneStage; sceneState.linkBuildStart = null;
  document.body.dataset.stage = name; scene?.setStage(sceneStage);
  if (name !== "analysis") $("#job-status").hidden = true;
  document.querySelectorAll(".config-panel").forEach(panel => { panel.hidden = panel.dataset.panel !== name; panel.classList.toggle("is-active", !panel.hidden); });
  $("#analysis-shell").hidden = name !== "analysis";
  document.querySelectorAll("[data-stage-target]").forEach(node => { node.classList.toggle("is-current", node.dataset.stageTarget === name); if (node.closest("nav")) node.setAttribute("aria-current", node.dataset.stageTarget === name ? "step" : "false"); });
  $("#chapter-progress").style.transform = `scaleY(${Math.max(0, stages.indexOf(name)) / (stages.length - 1)})`;
  if (config && !dirty) renderForms();
  return true;
}

function selectNode(index) {
  if (!config || dirty || busy || config.topology.source === "file") return;
  if (sceneState.linkBuildStart === null) {
    sceneState.linkBuildStart = index;
    $("#link-selection").textContent = `已选 ${config.hardware.satellite_ids[index]}，请选择另一颗卫星`;
    scene?.setHardwareSatelliteSelection(index); return;
  }
  const source = config.hardware.satellite_ids[sceneState.linkBuildStart], target = config.hardware.satellite_ids[index];
  const links = config.topology.inter_satellite.links;
  if (source !== target && !links.some(link => [link.source, link.target].includes(source) && [link.source, link.target].includes(target))) links.push({ source, target, mode:"fixed", bandwidth_gbps:400 });
  sceneState.linkBuildStart = null; $("#link-selection").textContent = ""; renderForms(); changed();
}

function bindScene() {
  try {
    scene = new AsterMindScene($("#scene"), {
      onHover: cluster => { $("#cluster-peek").classList.toggle("is-visible", !!cluster); if (cluster) $("#peek-name").textContent = cluster.name || "轨道示意节点"; },
      onHoverStation: () => {}, onSelect: cluster => { if (config) stage("formation", cluster); },
      onLinkSelect: (_, edge) => {
        if(!stage("network")) return;
        const ids = edge?.map(index => config.hardware.satellite_ids[index]);
        const index = config.topology.inter_satellite.links.findIndex(link => ids?.includes(link.source) && ids.includes(link.target));
        const row = [...document.querySelectorAll("[data-detail-path]")].find(node => node.dataset.detailPath === JSON.stringify(["topology","inter_satellite","links",index]));
        if (row) { row.open = true; row.scrollIntoView({block:"center"}); }
      }, onFormationNodeSelect: () => stage("formation"),
      onHardwareNodeSelect: () => {}, onTerminalNodeSelect: selectNode,
      onLinkRemove: ([a, b]) => {
        if (!config || dirty || busy || config.topology.source === "file") return;
        const ids = [config.hardware.satellite_ids[a], config.hardware.satellite_ids[b]];
        config.topology.inter_satellite.links = config.topology.inter_satellite.links.filter(l => !(ids.includes(l.source) && ids.includes(l.target)));
        renderForms(); changed();
      },
    });
  } catch { message("三维场景不可用；配置表单、拓扑数据及仿真功能仍可使用。", true); }
}

async function guarded(action, checkForm = false) {
  if (busy) return;
  try {
    if (checkForm) validateFormInputs();
    busy = true; lockFields(); await action();
  } catch (error) { showError(error); }
  finally { busy = false; lockFields(); }
}

function bindUpload(id, kind, apply) {
  $(id).addEventListener("change", async event => {
    const file = event.target.files[0]; if (!file) return;
    await guarded(async () => {
      if (file.size > 16 * 1024 * 1024) throw new Error("文件超过 16 MiB 上限");
      message(`正在上传 ${file.name}…`);
      const resource = await api(`/api/uploads/${kind}`, { method: "POST", headers: { "Content-Type": "application/octet-stream" }, body: file });
      apply(resource); renderForms(); changed();
      message(`已导入 ${file.name}`);
    });
    event.target.value = "";
  });
}

function remember(id) {
  const history = stored("astermind-jobs", []).filter(item => typeof item === "string" && /^AM-\d{8}-[0-9a-f]{32}$/.test(item));
  save("astermind-jobs", [id, ...history.filter(item => item !== id)].slice(0, 100));
  save("astermind-active-job", id); historyReplace(id);
}
function historyReplace(id) { window.history.replaceState(null, "", `#job=${encodeURIComponent(id)}`); }

async function watch(id, open = false) {
  const version = ++pollVersion; jobId = id; remember(id);
  if (open) stage("analysis");
  $("#job-status").hidden = false;
  while (version === pollVersion) {
    try {
      const status = await api(`/api/simulations/${id}`);
      if (version !== pollVersion) return;
      $("#job-progress").textContent = `${statuses[status.status] || status.status} · ${phases[status.phase] || status.phase || ""} · ${status.progress ?? 0}%`;
      if (status.status === "success") {
        const result = await api(`/api/simulations/${id}/result`);
        if (version !== pollVersion) return;
        await renderResults($("#simulation-results"), id, result, api); message(""); return;
      }
      if (!["queued", "running", "received"].includes(status.status)) {
        $("#simulation-results").replaceChildren(element("h2", statuses[status.status] || status.status), element("pre", status.error || "任务未完成，请检查配置后重新提交。"));
        await renderArtifacts($("#simulation-results"), id, api); return;
      }
      $("#simulation-results").replaceChildren(element("h2", "仿真进行中"), element("p", "可以关闭网页；后台任务继续执行。通过历史记录或当前页面链接恢复查看。"));
    } catch (error) { message(`任务连接中断，将自动重试：${error.message}`, true); }
    await new Promise(resolve => setTimeout(resolve, 2000));
  }
}

async function historyView() {
  if(!stage("analysis")) return; const list = $("#history-list"); list.replaceChildren(element("h2", "本浏览器任务历史"), element("p", "任务数据保存在服务器；本浏览器记录最近 100 个任务链接。请收藏链接以便跨浏览器访问。", "muted"));
  const ids = stored("astermind-jobs", []).filter(id => /^AM-\d{8}-[0-9a-f]{32}$/.test(id));
  if (!ids.length) list.append(element("p", "尚无历史任务。"));
  for (const id of ids) {
    const row = element("div", undefined, "history-row"); const open = element("button", id); open.onclick = () => watch(id, true);
    const reuse = element("button", "载入配置"); reuse.onclick = () => guarded(async () => {
      if (dirty) throw new Error("请先应用或放弃高级 JSON 编辑");
      config = importResourceHardware(await api(`/api/simulations/${id}/configuration`)); renderForms(); changed(); stage("formation");
    });
    row.append(open, reuse); list.append(row);
  }
  list.scrollIntoView({ block: "start" });
}

async function initialize() {
  bindScene(); stage("home"); lockFields();
  capabilities = await api("/api/capabilities");
  config = importResourceHardware(stored("astermind-config", null) || newWebConfiguration(capabilities));
  renderForms(); changed();
  const current = new URLSearchParams(location.hash.slice(1)).get("job") || stored("astermind-active-job", null);
  if (current && /^AM-\d{8}-[0-9a-f]{32}$/.test(current)) { jobId = current; if (location.hash) watch(current, true); }
}

document.querySelectorAll("[data-stage-target]").forEach(node => node.addEventListener("click", () => stage(node.dataset.stageTarget)));
document.querySelectorAll("[data-action=home]").forEach(node => node.addEventListener("click", () => { window.history.replaceState(null,"",location.pathname + location.search); stage("home"); }));
document.querySelectorAll("[data-action=start-config]").forEach(node => { node.onclick = () => stage("formation"); });
$("#show-history").onclick = historyView;
$("#reset-view").onclick = () => scene?.resetConfigView();
$("#open-job").onclick = () => { if (jobId) stage("analysis"); };
$("#advanced-config").oninput = () => { dirty = true; ++previewVersion; previewKey = ""; clearTimeout(previewTimer); previewAbort?.abort(); lockFields(); };
$("#apply-json").onclick = () => { try { const next = importResourceHardware(parseConfiguration($("#advanced-config").value)); for (const key of ["hardware", "deployment", "scheduler", "requests", "topology", "runtime_inputs", "runtime"]) if (!next[key]) throw new Error(`缺少配置段：${key}`); const error=validationError(configurationErrors(next,capabilities.schema)); if(error) throw error; config = next; drafts.clear(); dirty = false; renderForms(); changed(); message("JSON 已应用"); } catch (error) { message(`JSON 未应用：${error.message}`, true); } };
$("#discard-json").onclick = () => { dirty = false; renderForms(); changed(); message(''); };
$("#export-config").onclick = () => { try { validateFormInputs(); const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([JSON.stringify(config, null, 2)], {type:"application/json"})); link.download = "astermind-config.json"; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000); } catch(error) { showError(error); } };
$("#bind-request-identity").onclick = () => guarded(async () => {
  const identity = await post("/api/requests/identity", config.requests);
  const count = bindRequestIdentity(config, identity);
  renderForms(); changed(); message(count ? `已绑定 ${count} 个 MTP 假设；后续修改请求流后需要重新绑定。` : "请先在调度器中设置 MTP 接受假设。");
}, true);
$("#run-simulation").onclick = () => guarded(async () => {
  message("正在校验并提交…"); await post("/api/config/validate", config);
  const job = await post("/api/simulations", config); $("#history-list").replaceChildren(); watch(job.id, true);
}, true);
bindUpload("#tle-upload", "tle", resource => {
  config.constellation ||= { timestamp: new Date().toISOString(), earth_orientation_mode: "stk11_compat" };
  config.constellation.tle_file = resource.path; config.constellation.formation = null; config.topology.source = "orbit"; config.topology.file = null;
  message(`已导入 TLE：${resource.satellite_ids.join(", ")}`);
});
bindUpload("#request-upload", "requests", resource => {
  config.requests.file = resource.path; config.requests.synthetic = null;
  if (config.requests.source === "synthetic") { config.requests.source = "canonical"; config.requests.production = null; }
  $("#request-upload-status").textContent = resource.path.split("/").at(-1);
});
bindUpload("#kv-replay-upload", "kv_replay", resource => { const pool = $("#kv-replay-pool").value; if (!pool) throw new Error("请先配置 Scheduler pool"); config.kv_offload.enabled = true; config.kv_offload.replay_files[pool] = resource.path; });
window.addEventListener("beforeunload", event => { if (dirty || drafts.size) { event.preventDefault(); event.returnValue = ""; } });
initialize().catch(error => message(`初始化失败，请刷新重试：${error.message}`, true));
