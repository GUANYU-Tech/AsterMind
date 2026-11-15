import { stages, stageNames, groups } from "./config-layout.mjs?v=compact-2";
import { renderConfigForm } from "./config-form.mjs?v=validation-1";
import { editorChoices } from "./config-edit.mjs?v=validation-1";
import { element } from "./results.mjs?v=compact-2";
import { renderHardwareProfiles } from "./hardware-profile.mjs?v=profile-1";

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
    for (const [index, [title]] of groups[name].entries()) {
      const section = element("details", undefined, "editor-section"); section.dataset.group = `${name}-${index}`; section.open = index < 2 || title === "FlowSim";
      section.append(element("summary", title)); const fields = element("fieldset", undefined, "basic-fields"); fields.id = `form-${name}-${index}`; section.append(fields); scroll.append(section);
      if (name === "network" && title === "FlowSim") {
        const note = element("p", "当前 V1 使用 Runtime Comm。旧 FlowSim ACK 与 ECMP 固定种子不参与仿真；启用 ACK 或设置非零种子会被运行校验拒绝。", "unsupported-note");
        section.insertBefore(note, fields);
      }
      if (name === "formation" && index === 1) {
        const tle = element("div", undefined, "tle-input"); tle.id = "tle-input";
        const status = element("p", "", "upload-status"); status.id = "tle-upload-status";
        tle.append(upload("TLE 文件", "tle-upload", ".tle,.txt"), status); section.append(tle);
      }
    }
    if (name === "formation") {
      const details = element("details", undefined, "editor-section"); details.open = true; details.append(element("summary", "硬件档案")); const profile = element("div", undefined, "hardware-profile"); profile.id = "hardware-profile"; details.append(profile); scroll.append(details);
    }
    if (name === "network") {
      const links = element("p", "", "upload-status"); links.id = "link-selection";
      const note = element("p", "", "upload-status"); note.id = "preview-note";
      const preview = element("div", undefined, "table-scroll"); preview.id = "topology-preview";
      note.setAttribute("role", "status");
      scroll.append(links, note, preview);
    }
    if (name === "workload") { scroll.append(upload("请求流 · CSV / JSONL / Parquet", "request-upload", ".csv,.jsonl,.parquet")); const status = element("p", "", "muted"); status.id = "request-upload-status"; scroll.append(status); }
    if (name === "runtime") {
      scroll.append(action("为 MTP 假设绑定当前请求流", "bind-request-identity"));
      const pool = element("select"); pool.id = "kv-replay-pool"; pool.className = "result-select"; pool.setAttribute("aria-label", "KV 回放目标 pool");
      scroll.append(element("p", "KV 回放目标 pool", "fso-help"), pool, upload("KV 生命周期回放", "kv-replay-upload", ".json"));
      const json = element("details", undefined, "editor-section"); json.open = false; json.append(element("summary", "完整配置 · 导入与导出"));
      const area = element("textarea"); area.id = "advanced-config"; area.className = "config-json"; area.spellcheck = false; area.setAttribute("aria-label", "完整 AsterMindConfig");
      json.append(area,
        action("应用 JSON", "apply-json"), action("放弃 JSON 编辑", "discard-json"), action("导出配置", "export-config")); scroll.append(json);
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
    for (const d of container.querySelectorAll("details[data-detail-path]")) if (opened.has(d.dataset.detailPath)) d.open = opened.get(d.dataset.detailPath);
  });
  if(validation.drafts) {
    const present=new Set([...document.querySelectorAll('.basic-fields [data-config-path]')].map(n=>n.dataset.configPath));
    for(const path of validation.drafts.keys()) if(!present.has(path)) validation.drafts.delete(path);
  }
  const optical = config.topology.inter_satellite.links.some(link => (link.mode || "islsim") === "islsim") || !config.topology.inter_satellite.links.length;
  for (const i of [2, 3, 4]) document.querySelector(`[data-group="network-${i}"]`).hidden = !optical;
  const profile = document.querySelector("#hardware-profile");
  const models = new Set([config.hardware.default_hardware_id, ...(config.hardware.accelerator_overrides || []).map(x => x.hardware_id)]);
  renderHardwareProfiles(profile, models, capabilities.hardware_details);
  const poolSelect = document.querySelector("#kv-replay-pool"), selected = poolSelect.value;
  poolSelect.replaceChildren(...Object.keys(config.scheduler).map(name => { const option = element("option", name); option.value = name; return option; }));
  if (Object.hasOwn(config.scheduler, selected)) poolSelect.value = selected;
  document.querySelector("#tle-input").hidden = config.topology.source !== "orbit";
  document.querySelector("#tle-upload-status").textContent = config.constellation?.tle_file ? config.constellation.tle_file.split("/").at(-1) : "";
}
