import { stages, stageNames, groups } from "./config-layout.mjs?v=compact-2";
import { renderConfigForm } from "./config-form.mjs?v=validation-1";
import {optionLabels} from "./config-labels.mjs?v=compact-2";
import { editorChoices } from "./config-edit.mjs?v=validation-1";
import { element } from "./results.mjs?v=compact-2";
import { renderHardwareProfiles } from "./hardware-profile.mjs?v=profile-1";
import {deploymentCapacity} from './deployment-mapping.mjs';
import {automaticDeployment, resourceSyncError} from './resource-sync.mjs';

function action(text, id, cls = "small-action") { const b = element("button", text, cls); b.type = "button"; if (id) b.id = id; return b; }
function upload(title, id, accept) {
  const label = element("label", undefined, "import-row"); const caption = element("span");
  caption.append(element("b", title));
  const input = element("input"); input.type = "file"; input.accept = accept; input.id = id;
  label.append(caption, element("strong", "导入文件"), input); return label;
}
export function buildWorkspace() {
  const rail = document.querySelector("#chapter-rail"), shell = document.querySelector("#config-shell");
  stages.forEach((name, i) => {
    const button = action(""); button.className = ""; button.dataset.stageTarget = name;
    button.append(element("b", String(i + 1).padStart(2, "0")), element("span", stageNames[i])); rail.append(button);
    if (name === "analysis") return;
    const panel = element("section", undefined, "config-panel"); panel.dataset.panel = name; panel.hidden = true;
    const header = element("header", undefined, "panel-header"), heading = element("div");
    heading.append(element("span", `${["RESOURCES","NETWORK","WORKLOAD","DEPLOYMENT","RUNTIME"][i]} / 0${i + 1}`, "eyebrow"), element("h1", stageNames[i]));
    const close = action("×", null, "panel-close icon-button"); close.dataset.action = "home"; close.setAttribute("aria-label", "关闭配置"); header.append(heading, close);
    const scroll = element("div", undefined, "panel-scroll");
    if (name === 'deployment') {
      const summary = element('section', undefined, 'deployment-assistant');
      const capacity = element('p'); capacity.id = 'deployment-capacity'; capacity.setAttribute('role', 'status');
      const note = element('p', '默认自动适配资源与模型，生成 TP / PP 和卡映射。不能参与当前并行方案的卡保持空闲；可切换手动配置。', 'muted');
      const modes = element('div', undefined, 'boolean-modes'); modes.setAttribute('role','group'); modes.setAttribute('aria-label','部署配置方式');
      modes.append(action('自动适配资源','automatic-deployment','boolean-mode'),action('手动配置','manual-deployment','boolean-mode'));
      const actions = element('div', undefined, 'deployment-tools');
      actions.append(action('按资源匹配 TP / PP', 'match-parallelism'), action('重新生成卡映射', 'generate-mapping'));
      summary.append(capacity, note, modes, actions); scroll.append(summary);
    }
    for (const [index, [title]] of groups[name].entries()) {
      const section = element("details", undefined, "editor-section"); section.dataset.group = `${name}-${index}`; section.open = index < 2;
      section.append(element("summary", title)); const fields = element("fieldset", undefined, "basic-fields"); fields.id = `form-${name}-${index}`; section.append(fields); scroll.append(section);
      if (name === "formation" && index === 1) {
        const tle = element("div", undefined, "tle-input"); tle.id = "tle-input";
        const status = element("p", "", "upload-status"); status.id = "tle-upload-status";
        tle.append(upload("TLE 文件", "tle-upload", ".tle,.txt"), status); section.append(tle);
      }
    }
    if (name === "formation") {
      scroll.append(action('恢复默认配置','restore-defaults'));
      const details = element("details", undefined, "editor-section"); details.open = true; details.append(element("summary", "硬件档案")); const profile = element("div", undefined, "hardware-profile"); profile.id = "hardware-profile"; details.append(profile); scroll.append(details);
    }
    if (name === "network") {
      const links = element("p", "", "upload-status"); links.id = "link-selection";
      const note = element("p", "", "upload-status"); note.id = "preview-note";
      const preview = element("div", undefined, "table-scroll"); preview.id = "topology-preview";
      note.setAttribute("role", "status");
      scroll.append(links, note, preview);
    }
    if (name === "workload") {
      const section = scroll.querySelector('[data-group="workload-1"]');
      const modelCard = element("p", "", "upload-status"); modelCard.id="supported-models";
      scroll.querySelector('[data-group="workload-0"] .basic-fields').before(modelCard);
      const box = element("div"); box.id = "canonical-import"; box.hidden = true;
      const status = element("p", "", "upload-status"); status.id = "request-upload-status";
      box.append(upload("标准请求文件（CSV）", "request-upload", ".csv"), status);
      section.append(box);
    }
    if (name === "runtime") {
      const check = element('p','','configuration-check'); check.id='configuration-check'; check.setAttribute('role','status'); scroll.prepend(check);
      const json = element("details", undefined, "editor-section"); json.open = false; json.append(element("summary", "完整配置"));
      const area = element("textarea"); area.id = "advanced-config"; area.className = "config-json"; area.spellcheck = false; area.setAttribute("aria-label", "完整 AsterMindConfig");
      const profile = element("div"); profile.id = "config-profile";
      const editor = element("details"); editor.append(element("summary", "编辑 JSON"));
      json.append(profile, editor, action("导出配置", "export-config"));
      editor.append(area,
        action("应用 JSON", "apply-json"), action("放弃 JSON 编辑", "discard-json")); scroll.append(json);
    }
    const footer = element("div", undefined, `panel-actions${i === 0 ? " panel-actions-single" : ""}`);
    if (i) { const back = action("← 上一步", null, "back-action"); back.dataset.stageTarget = stages[i - 1]; footer.append(back); }
    const next = action("", i === 4 ? "run-simulation" : null, "primary-action"); next.append(element("span", i === 4 ? "发布任务" : `进入${stageNames[i + 1]}`)); if (i !== 4) next.dataset.stageTarget = stages[i + 1]; footer.append(next);
    panel.append(header, scroll, footer); shell.append(panel);
  });
}

export function renderWorkspaceForms(config, capabilities, changed, onEdit, onError, validation = {}) {
  const choices = {...editorChoices(config, capabilities), onEdit, onError, ...validation};
  for (const [stage, sections] of Object.entries(groups)) sections.forEach(([, paths], i) => {
    const container = document.querySelector(`#form-${stage}-${i}`);
    const opened = new Map([...container.querySelectorAll("details[data-detail-path]")].map(d => [d.dataset.detailPath, d.open]));
    renderConfigForm(container, config, paths, capabilities.schema, changed, {...choices, stage});
    if (paths.includes('fso')) container.parentElement.hidden = !container.childElementCount;
    for (const d of container.querySelectorAll("details[data-detail-path]")) if (opened.has(d.dataset.detailPath)) d.open = opened.get(d.dataset.detailPath);
  });
  if(validation.drafts) {
    const present=new Set([...document.querySelectorAll('.basic-fields [data-config-path]')].map(n=>n.dataset.configPath));
    for(const path of validation.drafts.keys()) if(!present.has(path)) validation.drafts.delete(path);
  }
  const cardId=config.hardware.default_hardware_id;
  document.querySelector('#supported-models').textContent=(optionLabels[cardId]||cardId)+' 可用模型 '+document.querySelectorAll('#field-model option:not(:disabled)').length+' 个';
  const capacity = deploymentCapacity(config);
  const idle = Math.max(0,capacity.physical-capacity.required);
  const status = resourceSyncError(config) || (capacity.required > capacity.physical ? '资源不足，请启用自动适配资源。' : `已分配 ${capacity.required} 张卡，空闲 ${idle} 张。`);
  document.querySelector('#deployment-capacity').textContent = `${config.hardware.satellite_ids.length} 颗卫星 × 每星 ${config.hardware.accelerators_per_satellite} 张卡 = ${capacity.physical} 张物理卡；${status}`;
  document.querySelector('#automatic-deployment').setAttribute('aria-pressed',String(automaticDeployment(config)));
  document.querySelector('#manual-deployment').setAttribute('aria-pressed',String(!automaticDeployment(config)));
  const profile = document.querySelector("#hardware-profile");
  const models = new Set([config.hardware.default_hardware_id, ...(config.hardware.accelerator_overrides || []).map(x => x.hardware_id)]);
  renderHardwareProfiles(profile, models, capabilities.hardware_details);
  document.querySelector("#canonical-import").hidden = config.requests.source !== "canonical";
  document.querySelector("#request-upload-status").textContent = config.requests.file?.split("/").at(-1) || "";
  document.querySelector("#tle-input").hidden = config.topology.source !== "orbit";
  document.querySelector("#tle-upload-status").textContent = config.constellation?.tle_file ? config.constellation.tle_file.split("/").at(-1) : "";
}
