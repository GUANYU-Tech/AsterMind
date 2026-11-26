import { AsterMindScene, sceneState, syncSceneConfiguration } from "./scene.js?v=profile-1";
import { parseConfiguration } from "./config-form.mjs?v=validation-1";
import { buildWorkspace, renderWorkspaceForms } from "./workspace-ui.mjs?v=profile-1";
import { stages, stageNames, importResourceHardware, validationTarget } from "./config-layout.mjs?v=compact-2";
import { editDependencies, newWebConfiguration, networkInputs } from "./config-edit.mjs?v=validation-1";
import {mappingDraft} from './deployment-mapping.mjs';
import {normalizeComputeSettings} from './compute-config.mjs';
import {renderSchedulingReadout} from './parameter-panel.mjs';
import {linkMode, setLinkMode, rememberEmptyLinkMode, newInterSatelliteLink, applyOpticalDefaults} from './link-mode.mjs';
import {synchronizeManualResources as synchronizeResources,clearDependentDrafts} from './resource-sync.mjs';
import { renderResults, renderArtifacts, renderNetworkPreview, element } from "./results.mjs?v=compact-2";
import { appUrl, appStorageKey } from "./app-path.mjs?v=compact-2";
import { sceneData } from "./scene-data.mjs?v=compact-2";
import { configurationErrors, validationError } from "./config-validation.mjs?v=validation-1";

import {renderCellConfigProfile} from "../assets/config-profile.js";
import {formatTaskLog,taskPhase,taskFailure} from "../assets/progress_log.js";
import {labels,formationNames} from "./config-labels.mjs?v=compact-2";
const $ = selector => document.querySelector(selector);
buildWorkspace();
const statuses = { queued: "排队", running: "运行中", success: "成功", failed: "失败", interrupted: "已中断", cancelled: "已取消", rejected: "校验失败" };
let config, capabilities, scene, dirty = false, busy = false, jobId = null, pollVersion = 0, previewVersion = 0, refreshForms = false;
let previewTimer, previewAbort, previewKey = "", networkPreview = null;
let preflightTimer, preflightVersion = 0, preflightKey = '', preflightStatus = 'pending', effectiveCell;
const drafts = new Map();

function draftChanged() {
  if(drafts.size) {
    ++previewVersion; previewKey=''; clearTimeout(previewTimer); previewAbort?.abort();
    $('#scene-links').textContent='请修正输入 预览未更新';
  } else if(config) { message(''); scheduleNetworkPreview(); }
  if(config) scheduleConfigurationPreflight();
}

function scheduleConfigurationPreflight() {
  clearTimeout(preflightTimer); const version = ++preflightVersion;
  if (!config || !capabilities) return;
  const output = $('#configuration-check');
  const issues = configurationErrors(config,capabilities.schema);
  if (dirty || drafts.size || issues.length) {
    preflightStatus='invalid'; preflightKey='';
    renderSchedulingReadout(document,null);
    output.textContent=dirty ? '请先应用或放弃 JSON 编辑。' : drafts.size ? '请修正输入框下方的提示。' : issues[0].msg;
    lockFields(); return;
  }
  const key=JSON.stringify(config);
  if (['valid','blocked'].includes(preflightStatus) && preflightKey===key) {
    renderSchedulingReadout(document,effectiveCell);lockFields();return;
  }
  renderSchedulingReadout(document,null);
  preflightStatus='pending'; output.textContent='进入调度运行后自动检查配置。'; lockFields();
  if (document.body.dataset.stage!=='runtime') return;
  output.textContent='正在进行后端预检…';
  preflightTimer=setTimeout(async()=>{
    try {
      const result=await post('/api/config/validate',config);
      if (version!==preflightVersion) return;
      effectiveCell=result.effective_configuration.cells[0];
      renderSchedulingReadout(document,effectiveCell);
      preflightStatus=result.launch_blocker?'blocked':'valid'; preflightKey=key;
      output.textContent=result.launch_blocker||'配置预检通过，可以发布任务。';
    } catch(error) {
      if (version!==preflightVersion) return;
      preflightStatus='invalid'; preflightKey=''; output.textContent=`配置需调整：${error.message}`;
    }
    lockFields();
  },500);
}

function scheduleNetworkPreview() {
  if(drafts.size || ['formation','network'].some(name=>configurationErrors(config,capabilities.schema,name).length)) {
    ++previewVersion; previewKey=''; clearTimeout(previewTimer); previewAbort?.abort();
    $('#scene-links').textContent='请修正输入 预览未更新'; return;
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
      $("#scene-links").textContent = `${edges.length} 条链路 ${reached.size === config.hardware.satellite_ids.length ? "全部接入" : "存在未接入卫星"}`;
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
function prepareOpticalSettings(value) {
  applyOpticalDefaults(value, capabilities.default_configuration.fso);
  rememberEmptyLinkMode(value, stored('astermind-link-mode', null));
  const mode = linkMode(value); if (mode) setLinkMode(value, mode);
  synchronizeResources(value,capabilities);
  return value;
}

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
    if (!input.checkValidity()) { const error = new Error(`${input.labels?.[0]?.textContent || input.id}：${input.validationMessage || '请检查输入格式或范围'}`); error.detail = [{loc:JSON.parse(input.dataset.configPath || "[]"),msg:error.message}]; throw error; }
  }
  const error=validationError(configurationErrors(config,capabilities.schema,name));
  if(error) throw error;
}

function lockFields() {
  document.querySelectorAll('[data-action=start-config]').forEach(control => { control.disabled = !config; });
  document.querySelectorAll('.chapter-rail button, .cell-overview, #show-history, .orbit-node-action').forEach(control => { control.disabled = !config; });
  document.querySelectorAll(".basic-fields").forEach(field => { field.disabled = dirty || busy; });
  document.querySelectorAll("input[type=file], #run-simulation, #export-config").forEach(control => { control.disabled = dirty || busy || !config; });
  $('#run-simulation').disabled = dirty || busy || !config || drafts.size>0 || preflightStatus!=='valid';
  $("#advanced-config").disabled = busy;
  $("#apply-json").disabled = $("#discard-json").disabled = busy;
}

function changed(path) {
  scheduleNetworkPreview();
  const layout = config.topology.source === "orbit" ? "轨道快照" : formationNames[config.constellation?.formation?.layout] || "环形编队";
  $("#scene-summary").textContent = `${layout} ${config.hardware.satellite_ids.length} 颗卫星`;
  $("#advanced-config").value = JSON.stringify(config, null, 2);
  renderCellConfigProfile($("#config-profile"), config, labels);
  save("astermind-config", config);
  if (linkMode(config)) save('astermind-link-mode', linkMode(config));
  if (scene) syncSceneConfiguration(scene, config, networkPreview);
  if (refreshForms) { refreshForms = false; renderForms(); }
  scheduleConfigurationPreflight();
}

function renderForms() {
  renderWorkspaceForms(config, capabilities, changed, (path, value, previous) => {
    if (['hardware','model','compute_mode'].includes(path[0]) || path[0]==='deployment' && ['worker_pools','mode'].includes(path[1])) clearDependentDrafts(drafts);
    refreshForms = editDependencies(config, path, value, capabilities, previous);
    if (path[0]==='deployment' && path[1]==='placements') refreshForms=true;
  }, text => message(text, true), {drafts,onDraftChange:draftChanged});
  $("#advanced-config").value = JSON.stringify(config, null, 2); lockFields();
  scheduleNetworkPreview();
  scheduleConfigurationPreflight();
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
  $('.studio-logo').href = name === 'home' ? '/platform' : appUrl('');
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
  if (!linkMode(config)) { message('请先选择统一的星间链路模式', true); return; }
  if (sceneState.linkBuildStart === null) {
    sceneState.linkBuildStart = index;
    $("#link-selection").textContent = `已选 ${config.hardware.satellite_ids[index]}，请选择另一颗卫星`;
    scene?.setHardwareSatelliteSelection(index); return;
  }
  const source = config.hardware.satellite_ids[sceneState.linkBuildStart], target = config.hardware.satellite_ids[index];
  const links = config.topology.inter_satellite.links;
  if (source !== target && !links.some(link => [link.source, link.target].includes(source) && [link.source, link.target].includes(target))) links.push(newInterSatelliteLink(config, source, target));
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
  $('#artifact-shortcut').hidden=true;
  $('#cell-task-log').hidden=true;
  const version = ++pollVersion; jobId = id; remember(id);
  if (open) stage("analysis");
  $("#job-status").hidden = false;
  while (version === pollVersion) {
    try {
      const status = await api(`/api/simulations/${id}`);
      if (version !== pollVersion) return;
      $("#job-progress").textContent = [statuses[status.status] || status.status,taskPhase(status),
        Number.isFinite(status.progress) ? `${status.progress}%` : ''].filter(Boolean).join(' · ');
      $('#cell-log-lines').textContent=formatTaskLog(status);
      $('#cell-task-log').hidden=false;
      if (status.status === "success") {
        const result = await api(`/api/simulations/${id}/result`);
        if (version !== pollVersion) return;
        await renderResults($("#simulation-results"), id, result, api); message(""); return;
      }
      if (!["queued", "running", "received"].includes(status.status)) {
        $("#simulation-results").replaceChildren(element("h2", statuses[status.status] || status.status), element("pre", taskFailure(status) || "任务未完成，请检查配置后重新提交。"));
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
      config = prepareOpticalSettings(importResourceHardware(await api(`/api/simulations/${id}/configuration`)));
      normalizeComputeSettings(config, capabilities);
      config.deployment.placements = mappingDraft(config); renderForms(); changed(); stage("formation");
    });
    row.append(open, reuse); list.append(row);
  }
  list.scrollIntoView({ block: "start" });
}

async function initialize() {
  bindScene(); stage("home"); lockFields();
  capabilities = await api("/api/capabilities");
  config = prepareOpticalSettings(importResourceHardware(stored("astermind-config", null) || newWebConfiguration(capabilities)));
  normalizeComputeSettings(config, capabilities);
  config.deployment.placements = mappingDraft(config);
  renderForms(); changed();
  const current = new URLSearchParams(location.hash.slice(1)).get("job") || stored("astermind-active-job", null);
  if (current && /^AM-\d{8}-[0-9a-f]{32}$/.test(current)) { jobId = current; if (location.hash) watch(current, true); }
}

document.querySelectorAll("[data-stage-target]").forEach(node => node.addEventListener("click", () => stage(node.dataset.stageTarget)));
function returnToCellOverview() {
  history.replaceState(null, "", location.pathname + location.search);
  stage("home");
}
document.querySelectorAll("[data-action=home]").forEach(node => node.addEventListener("click", returnToCellOverview));
$('.studio-logo').addEventListener('click', event => {
  if (document.body.dataset.stage === 'home' || event.button !== 0 ||
      event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
  event.preventDefault();
  returnToCellOverview();
});
document.querySelectorAll("[data-action=start-config]").forEach(node => { node.onclick = () => stage("formation"); });
$("#show-history").onclick = historyView;
$("#reset-view").onclick = () => scene?.resetConfigView();
$("#open-job").onclick = () => { if (jobId) stage("analysis"); };
$("#advanced-config").oninput = () => { dirty = true; ++previewVersion; previewKey = ""; clearTimeout(previewTimer); previewAbort?.abort(); lockFields(); };
$("#apply-json").onclick = () => { try { const next = importResourceHardware(parseConfiguration($("#advanced-config").value)); for (const key of ["hardware", "deployment", "scheduler", "requests", "topology", "runtime_inputs", "runtime"]) if (!next[key]) throw new Error(`缺少配置段：${key}`); prepareOpticalSettings(next); normalizeComputeSettings(next, capabilities); const error=validationError(configurationErrors(next,capabilities.schema)); if(error) throw error; config = next; drafts.clear(); dirty = false; renderForms(); changed(); message("JSON 已应用"); } catch (error) { message(`JSON 未应用：${error.message}`, true); } };
$("#discard-json").onclick = () => { dirty = false; renderForms(); changed(); message(''); };
$("#export-config").onclick = () => { try { validateFormInputs(); const link = document.createElement("a"); link.href = URL.createObjectURL(new Blob([JSON.stringify(config, null, 2)], {type:"application/json"})); link.download = "astermind-config.json"; link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 1000); } catch(error) { showError(error); } };
$('#restore-defaults').onclick = () => {
  if(busy) return;
  save('astermind-before-defaults',config);
  config=newWebConfiguration(capabilities); synchronizeResources(config,capabilities);
  dirty=false; drafts.clear(); renderForms(); changed(); stage('formation'); message('已恢复可运行的默认配置');
};
$('#generate-mapping').onclick = () => {
  try {
    if (dirty || busy || drafts.size) throw Error('请先应用或放弃 JSON 编辑，并修正未完成的输入');
    const rows = mappingDraft(config, {reset:true});
    if (rows === config.deployment.placements) throw Error('当前资源或并行配置无效，请先调整后再生成卡映射');
    config.deployment.placements = rows;
    renderForms(); changed(); message('已按当前 TP / PP、卫星顺序和星内卡号重新生成映射');
  } catch (error) { showError(error); }
};
$("#run-simulation").onclick = () => guarded(async () => {
  message("正在校验并提交…"); await post("/api/config/validate", config);
  const job = await post("/api/simulations", config); $("#history-list").replaceChildren(); watch(job.id, true);
}, true);
bindUpload("#tle-upload", "tle", resource => {
  config.constellation ||= { timestamp: new Date().toISOString(), earth_orientation_mode: "stk11_compat" };
  config.constellation.tle_file = resource.path; config.constellation.formation = null; config.topology.source = "orbit"; config.topology.file = null;
  config.hardware.satellite_ids=resource.satellite_ids; config.hardware.satellite_count=resource.satellite_ids.length;
  config.topology.inter_satellite.links=resource.satellite_ids.slice(1).map((id,index)=>newInterSatelliteLink(config,resource.satellite_ids[index],id));
  clearDependentDrafts(drafts); synchronizeResources(config,capabilities);
  message(`已导入 TLE：${resource.satellite_ids.join(", ")}`);
});
bindUpload("#request-upload", "requests", resource => {
  config.requests.file = resource.path; config.requests.synthetic = null;
  if (config.requests.source === "synthetic") { config.requests.source = "canonical"; config.requests.production = null; }
  $("#request-upload-status").textContent = resource.path.split("/").at(-1);
});
window.addEventListener("beforeunload", event => { if (dirty || drafts.size) { event.preventDefault(); event.returnValue = ""; } });
initialize().catch(error => message(`初始化失败，请刷新重试：${error.message}`, true));
