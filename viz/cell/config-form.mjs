import { fieldStage, hardwareForPool, visibleField } from "./config-layout.mjs?v=compact-2";
import { labels, modelDisplayName, optionLabels } from "./config-labels.mjs?v=compact-2";
import { fieldError } from "./config-validation.mjs?v=validation-1";
import { enhanceSelects } from "../assets/design-controls.js";
import {linkMode, setLinkMode} from './link-mode.mjs';

export function setAt(root, path, value) {
  if (path.some(key => ["__proto__", "prototype", "constructor"].includes(String(key)))) throw new Error("非法字段名");
  let owner = root;
  for (const key of path.slice(0, -1)) owner = owner[key];
  owner[path.at(-1)] = value;
}
export function resolveSchema(schema, definitions) {
  return schema?.$ref ? definitions[schema.$ref.split("/").at(-1)] : schema || {};
}
export function defaultValue(raw, definitions) {
  const spec = resolveSchema(raw, definitions);
  if (spec.default !== undefined) return structuredClone(spec.default);
  if (spec.anyOf) return defaultValue(spec.anyOf.find(s => s.type !== "null"), definitions);
  if (spec.enum) return spec.enum[0];
  if (spec.type === "object") return Object.fromEntries(Object.entries(spec.properties || {}).filter(([key, s]) => spec.required?.includes(key) || s.default !== undefined).map(([key, s]) => [key, defaultValue(s, definitions)]));
  if (spec.type === "array") return (spec.prefixItems || []).map(item => defaultValue(item, definitions));
  if (spec.type === "boolean") return false;
  if (["integer", "number"].includes(spec.type)) return spec.minimum ?? ((spec.exclusiveMinimum ?? -1) + 1);
  return "";
}
const fixed = new Set(["data_parallel_size", "sequence_parallel_size", "context_parallel_size", "expert_communication_backend", "enable_eplb", "backend", "framework", "trust_remote_code"]);
const blobs = new Set(["profile", "replay", "mtp_acceptance_replay"]);
const managed = new Set(["output_directory", "host_evidence_directory"]);

export function renderConfigForm(container, root, keys, schema, onChange, choices = {}) {
  const defs = schema.$defs || {};
  const el = (tag, text, cls) => { const n = document.createElement(tag); if (text !== undefined) n.textContent = text; if (cls) n.className = cls; return n; };
  const button = (text, action, cls = "small-action") => { const n = el("button", text, cls); n.type = "button"; n.addEventListener("click", action); return n; };
  function validation(input, wrap, path, spec, read) {
    const id=JSON.stringify(path), saved=choices.drafts?.get(id);
    const originalPlaceholder = input.getAttribute('placeholder') || '';
    if(saved) input.value=saved.raw;
    const note=el('small', '', 'field-validation');
    note.id=`error-${path.join('-')}`; note.setAttribute('aria-live','polite');
    input.setAttribute('aria-describedby',note.id);
    const check=(remember=true)=>{
      const message=(input.type==='number' && input.value==='') ? '请输入数值' : fieldError(read(),path,root,spec,defs);
      input.setCustomValidity(message); input.setAttribute('aria-invalid',String(!!message));
      note.textContent=message; note.hidden=!message;
      const emptyNumber = input.type === 'number' && input.value === '';
      note.classList.toggle('input-placeholder-error', emptyNumber);
      if (input.type === 'number') input.placeholder = emptyNumber ? message : originalPlaceholder;
      if(!message) wrap.classList?.remove('field-error');
      if(remember) {
        if(message) choices.drafts?.set(id,{raw:input.value,error:message});
        else choices.drafts?.delete(id);
        choices.onDraftChange?.();
      }
      return !message;
    };
    check(false);
    wrap.append(note);
    return {check,note};
  }
  const rerender = (path = []) => {
    choices.onEdit?.(path, path.reduce((v, k) => v?.[k], root));
    renderConfigForm(container, root, keys, schema, onChange, choices); onChange(path);
  };
  function schemaAt(path) {
    let spec = schema;
    for (const key of path) {
      spec = resolveSchema(spec, defs);
      if (spec.anyOf) spec = resolveSchema(spec.anyOf.find(s => s.type !== "null"), defs);
      spec = typeof key === "number" ? (spec.prefixItems?.[key] || spec.items) : (spec.properties?.[key] || spec.additionalProperties);
    }
    return spec || {};
  }
  function expand(parts, path = []) {
    if (!parts.length) return [path];
    const [key, ...rest] = parts;
    if (key !== "*") return expand(rest, [...path, key]);
    const value = path.reduce((v, k) => v?.[k], root);
    if (Array.isArray(value)) return value.flatMap((_, i) => expand(rest, [...path, i]));
    return value && typeof value === 'object' ? Object.keys(value).flatMap(name => expand(rest, [...path, name])) : [];
  }
  function field(raw, path, depth = 0) {
    const key = path.at(-1), full = path.join(".");
    const linkField = path[0] === "topology" && path.includes("links");
    if (linkField && key === 'mode') return null;
    if (linkField && key === "bandwidth_gbps" && root.topology.inter_satellite.links[path[3]].mode !== "fixed") return null;
    if (!visibleField(path, root) || choices.visible?.(path) === false) return null;
    if (path[0] === 'fso' && linkMode(root) === 'fixed') return null;
    if (full === 'fso.tx.baud_rate_options_ghz' && root.fso.tx?.baud_rate_ghz != null) return null;
    if (choices.stage && fieldStage(path) !== choices.stage) return null;
    if (key === "hardware_id" && path.includes("worker_pools")) return null;
    if (path[0] === "constellation" && root.topology.source === "file") return null;
    if (key === "formation" && root.topology.source !== "formation") return null;
    if (key === "tle_file" && root.topology.source !== "orbit") return null;
    if (full === "topology.file" && root.topology.source !== "file") return null;
    if (full === "requests.synthetic" && root.requests.source !== "synthetic") return null;
    if (full === "requests.production" && root.requests.source !== "production") return null;
    if (full === "requests.file" && root.requests.source === "synthetic") return null;
    if (["vllm", "sglang"].includes(key) && path[0] === "scheduler" && root.scheduler[path[1]].framework !== key) return null;
    let spec = resolveSchema(raw, defs);
    if (key === 'max_model_len' && choices.modelContext) {
      spec = {...spec, anyOf:spec.anyOf?.map(option=>option.type==='integer'?{...option, maximum:choices.modelContext}:option)};
    }
    let value = path.reduce((v, k) => v?.[k], root);
    if (linkField && key === "mode" && value == null) value = "islsim";
    let title = choices.labels?.[key] || labels[key] || String(key);
    if (key === "source" && path.includes("links")) title = "源卫星";
    if (linkField && key === "mode") title = "链路模式";
    if (linkField && key === "bandwidth_gbps") title = "链路带宽 / Gbps";
    if (key === "kv_cache") title = `${root.deployment.worker_pools[path[2]]?.name || "pool"} KV 分层`;
    const nullable = spec.anyOf?.some(s => s.type === "null");
    const scalarTypes = spec.anyOf?.map(s => s.type).filter(t => ["string", "integer", "number", "boolean"].includes(t));
    if (spec.anyOf) spec = resolveSchema(spec.anyOf.find(s => s.type === (typeof value === "number" ? "integer" : typeof value)) || spec.anyOf.find(s => s.type !== "null"), defs);
    const wrap = el("div", undefined, "config-field");
    wrap.dataset.fieldPath = JSON.stringify(path);
    if (path[0]==='scheduler' && root.compute_mode==='fast_mode' && ['num_kv_blocks','kv_block_size_tokens'].includes(key)) {
      const output=el('output','自动 · 等待后端预检');
      output.dataset.schedulerValue=key;
      wrap.className+=' read-only-field';
      wrap.append(el('span',title,'field-title'),output);
      return wrap;
    }
    if (scalarTypes?.length > 1) {
      const typeSelect = el("select"); typeSelect.setAttribute("aria-label", `${title} 值类型`);
      for (const type of scalarTypes) { const option = el("option", {string:"文本",integer:"整数",number:"数值",boolean:"布尔值"}[type]); option.value = type; typeSelect.append(option); }
      typeSelect.value = typeof value === "number" ? "integer" : typeof value === "boolean" ? "boolean" : "string";
      typeSelect.addEventListener("change", () => { setAt(root, path, {string:"",integer:0,number:0,boolean:false}[typeSelect.value]); rerender(); });
      wrap.append(typeSelect);
    }
    const readonly = fixed.has(key) && !(key === "data_parallel_size" && ['fast_mode','detail_mode'].includes(root.compute_mode))
      || managed.has(key) || String(key).endsWith("sha256") || (["model","dtype"].includes(key) && path[0] === "scheduler")
      || (key === "topology" && path[0] === "deployment") || spec.readOnly;
    if (readonly) {
      wrap.className += " read-only-field";
      const display = key === "topology" ? optionLabels[value] ?? value : value;
      wrap.append(el("span", title, "field-title"), el("output", display == null ? "自动 / 未启用" : String(display)));
      return wrap;
    }
    if (blobs.has(key)) {
      wrap.className += " field-wide";
      const label = el("label", undefined, "import-row");
      const caption = el("span"); caption.append(el("b", title));
      const input = el("input"); input.type = "file"; input.accept = ".json";
      input.addEventListener("change", async () => {
        try { const file = input.files[0]; if (!file) return; if (file.size > 16 * 1024 * 1024) throw new Error("文件超过 16 MiB"); const data = parseConfiguration(await file.text()); choices.onObservation?.(path, data); setAt(root, path, data); rerender(); }
        catch (error) { choices.onError?.(error.message); }
      });
      label.append(caption, el("strong", "导入"), input); wrap.append(label);
      if (value) wrap.append(button("清除观测数据", () => { setAt(root, path, null); rerender(); }));
      return wrap;
    }
    const unsetCard = key === "local_index" && path.includes("placements") && value == null;
    const fixedBaudRate = full === 'fso.tx.baud_rate_ghz';
    if ((value === null || value === undefined) && !unsetCard) {
      const automatic = ['max_model_len','max_batched_tokens','num_kv_blocks','kv_block_size_tokens'].includes(key);
      const fixedKvPool = key === "fast_kv_pool_blocks";
      const note = fixedBaudRate ? "根据候选波特率自动选档"
        : fixedKvPool ? "自动按可用显存分配"
        : automatic ? "自动 · 等待后端预检" : "未设置 / 未启用";
      const status = el("small", note);
      if (automatic) status.dataset.schedulerValue = key;
      wrap.append(el("span", title, "field-title"), status, button(fixedBaudRate ? "设为固定波特率" : fixedKvPool ? "指定块数" : "设置", () => {
        setAt(root, path, choices.seed?.(path, spec) ?? defaultValue(spec, defs)); rerender(path);
      })); return wrap;
    }
    if (spec.type === "object" || (value !== null && typeof value === "object" && !Array.isArray(value))) {
      wrap.className += " field-container";
      if (full === "flowsim.acknowledgement") wrap.classList.add("ack-settings");
      const group = el("fieldset", undefined, "field-group");
      if (full === 'topology.inter_satellite') {
        const modeField = el('div', undefined, 'config-field field-wide global-link-mode');
        modeField.append(el('span', '全部星间链路模式', 'field-title'));
        const modes = el('div', undefined, 'link-modes'); modes.setAttribute('role', 'group'); modes.setAttribute('aria-label', '全部星间链路模式');
        for (const [mode, text] of [['fixed','默认模式'], ['islsim','精细模式']]) {
          const choice = button(text, () => { setLinkMode(root, mode); rerender(['topology','inter_satellite','links']); }, `link-mode${linkMode(root) === mode ? ' is-selected' : ''}`);
          choice.setAttribute('aria-pressed', String(linkMode(root) === mode)); modes.append(choice);
        }
        modeField.append(modes); group.append(modeField);
        const note = el('p', linkMode(root) === null ? '已有链路使用不同模式，请选择一个模式统一应用。' : '对现有及新增链路统一生效。默认模式使用配置带宽；精细模式使用下方统一光学参数计算。', 'link-mode-note');
        note.setAttribute('role', 'status'); group.append(note);
      }
      if (depth > 0 && typeof key !== "number" && key !== "formation" && !(path[0] === "scheduler" && path.length === 2)) group.append(el("legend", title));
      if (path[1] === "worker_pools" && path.length === 3 && typeof key === "number") group.append(el("div", `卡型号　${optionLabels[hardwareForPool(root, value.name)] || hardwareForPool(root, value.name)}`, "pool-hardware field-wide"));
      for (const [child, sub] of Object.entries(spec.properties || {})) { const rendered = field(sub, [...path, child], depth + 1); if (rendered) group.append(rendered); }
      if (spec.additionalProperties && typeof spec.additionalProperties === "object") {
        for (const child of Object.keys(value)) {
          if (full === "scheduler" && Object.keys(value).length === 1) {
            const rendered = field(spec.additionalProperties, [...path, child], depth + 1); if (rendered) group.append(rendered); continue;
          }
          const row = el("details", undefined, "repeat-item"); row.open = true; row.dataset.detailPath = JSON.stringify([...path, child]);
          row.append(el("summary", child)); const rendered = field(spec.additionalProperties, [...path, child], depth + 1); if (rendered) row.append(rendered);
          if (full !== "scheduler") row.append(button(`删除 ${child}`, () => { delete value[child]; rerender(path); })); group.append(row);
        }
        if (full !== "scheduler") {
        const name = el("input"); name.placeholder = key === "pcie_latency" ? "硬件 ID" : "pool 名称"; name.setAttribute("aria-label", `${title} 新增名称`);
        group.append(name, button("添加", () => {
          const id = name.value.trim();
          if (!id || Object.hasOwn(value, id) || ["__proto__", "constructor", "prototype"].includes(id)) { name.setCustomValidity("名称为空、重复或非法"); name.reportValidity(); return; }
          setAt(root, [...path, id], choices.seed?.([...path, id], spec.additionalProperties) ?? defaultValue(spec.additionalProperties, defs)); rerender();
        }));
        }
      }
      if (nullable && !["requests", "scheduler"].includes(path[0]) && !["constellation", "constellation.formation"].includes(full)) group.append(button("设为未启用", () => { setAt(root, path, null); rerender(); }));
      wrap.append(group); return wrap;
    }
    if (Array.isArray(value)) {
      const itemSpec = resolveSchema(spec.items, defs);
      if (!spec.prefixItems && ["string", "integer", "number"].includes(itemSpec.type)) {
        if (key === "satellite_ids") wrap.className += " field-wide";
        const isSu = full === 'hardware.su_group_sizes';
        const input = el("input"); input.type = "text"; input.value = value.join(isSu ? " + " : ", "); input.setAttribute("aria-label", title);
        input.dataset.configPath = JSON.stringify(path);
        input.placeholder = isSu ? choices.autoSuGroups ?? '' : "逗号分隔";
        input.id = `field-${path.join('-')}`;
        const label = el("label", title); label.htmlFor = input.id; wrap.append(label, input);
        const read=()=>input.value.trim()==='' ? [] : input.value.split(isSu ? /[+＋,，\n]/ : /[,，\n]/).map(x=>x.trim()).map(x=>itemSpec.type==='string' ? x : x==='' ? NaN : Number(x));
        if (isSu && choices.suLimit) spec = {...spec, items:{...itemSpec, maximum:choices.suLimit}};
        const state=validation(input,wrap,path,spec,read);
        input.addEventListener('input',()=>state.check());
        input.addEventListener("change", () => {
          if(!state.check()) return;
          const values=read(); setAt(root, path, values); choices.onEdit?.(path, values, value); onChange(path);
        }); return wrap;
      }
      wrap.className += " field-container";
      if (full === "topology.inter_satellite.links") wrap.classList.add("link-collection");
      wrap.append(el("div", `${title} ${value.length}`, "field-title"));
      value.forEach((item, index) => {
        const row = el("details", undefined, "repeat-item"); row.open = true; row.dataset.detailPath = JSON.stringify([...path, index]);
        const desc = item?.name || item?.operation || (item?.worker_pool ? `${item.worker_pool} DP${item.parallel_coordinate.dp} / PP${item.parallel_coordinate.pp} / TP${item.parallel_coordinate.tp}` : item?.satellite_id ? `${item.satellite_id} / 卡 ${item.local_index}` : item?.source ? `${item.source} → ${item.target}` : `第 ${index + 1} 项`);
        const poolRole = full === 'deployment.worker_pools' && root.deployment.mode === 'pd_disagg'
          ? {prefill:'Prefill', decode:'Decode'}[item.role] : null;
        const heading = poolRole && item.name !== item.role ? `${poolRole} · ${desc}` : poolRole || desc;
        row.append(el("summary", heading));
        const rendered = field(spec.prefixItems?.[index] || spec.items || {}, [...path, index], depth + 1); if (rendered) row.append(rendered);
        if (!spec.prefixItems && full !== "deployment.placements") {
          const remove = button("删除此项", () => {
          for(const [id,draft] of [...(choices.drafts || [])]) {
            const parts=JSON.parse(id);
            if(path.every((k,i)=>parts[i]===k) && typeof parts[path.length]==='number' && parts[path.length]>=index) {
              choices.drafts.delete(id);
              if(parts[path.length]>index) { parts[path.length]--; choices.drafts.set(JSON.stringify(parts),draft); }
            }
          }
          value.splice(index, 1); rerender(path);
          });
          if (full === "topology.inter_satellite.links") { remove.textContent = '−'; remove.classList.add('link-remove'); remove.setAttribute('aria-label', `删除链路 ${desc}`); remove.title = `删除链路 ${desc}`; }
          row.append(remove);
        }
        wrap.append(row);
      });
      if (!spec.prefixItems && full !== "deployment.placements") {
        const add = button(`添加${title}`, () => { value.push(choices.seed?.([...path, value.length], spec.items) ?? defaultValue(spec.items, defs)); rerender(path); });
        if (full === "topology.inter_satellite.links") { add.textContent = '+'; add.classList.add('link-add'); add.setAttribute('aria-label', `添加${title}`); add.title = `添加${title}`; add.disabled = !linkMode(root); }
        wrap.append(add);
      }
      return wrap;
    }
    // Choice callbacks share this object with field options (e.g. `seed`).
    // Only arrays are dropdown data; a same-named callback is not iterable.
    const candidate = full === "topology.source" ? ["formation", "orbit"] : choices.options?.(path) || spec.enum || choices[full] || choices[key];
    const options = Array.isArray(candidate) ? candidate : undefined;
    const input = el(options ? "select" : "input");
    input.id = `field-${path.map(x => encodeURIComponent(x)).join("-")}`; input.dataset.configPath = JSON.stringify(path);
    const label = el("label", title); label.htmlFor = input.id; wrap.append(label);
    if (options) {
      const strict = ["topology.source", "model", "requests.source", "requests.synthetic.arrival_distribution"].includes(full);
      const values = strict ? options : [...new Set([...options, value])];
      if (strict && !values.includes(value)) {
        const label = full === "model" ? `${modelDisplayName(choices.optionLabels?.[value] || value)}（当前模式或卡型不支持）` : "已载入的外部配置";
        const old = el("option", label); old.value = value; old.disabled = true; input.append(old);
      }
      for (const option of values) {
        if (option === null) continue;
        const label = choices.optionLabels?.[option] || optionLabels[option] || (option === "" ? "默认" : String(option));
        const n = el("option", key === "model" ? modelDisplayName(label) : label);
        n.value = option;
        input.append(n);
      }
      input.value = value;
    } else if (typeof value === "boolean") { input.type = "checkbox"; input.checked = value; }
    else { input.type = ["integer","number"].includes(spec.type) || typeof value === "number" || unsetCard ? "number" : "text"; input.value = unsetCard ? "" : value;
      if (unsetCard) input.placeholder = "待指定";
      if (input.type === "number") { input.required = true; input.step = spec.type === "integer" ? "1" : "any"; if (spec.minimum !== undefined) input.min = spec.minimum; if (spec.maximum !== undefined) input.max = spec.maximum; }
    }
    const read=()=>typeof value==='boolean' ? input.checked : input.type==='number' || ['integer','number'].includes(spec.type) ? Number(input.value) : input.value;
    const state=validation(input,wrap,path,spec,read);
    const commit = () => {
      if (!state.check() || !input.checkValidity()) return;
      const next = read();
      const previous = path.reduce((v, k) => v?.[k], root);
      setAt(root, path, next); choices.onEdit?.(path, next, previous); onChange(path);
    };
    const commitEvent=options || ["satellite_count", "accelerators_per_satellite"].includes(key) || (path[1] === "worker_pools" && ((path.length === 4 && key === "name") || path[3] === "parallelism")) ? "change" : "input";
    input.addEventListener(commitEvent,commit);
    if(commitEvent==='change') input.addEventListener('input',()=>state.check());
    if (input.type === "checkbox") wrap.className += " toggle-field";
    if (["satellite_count", "accelerators_per_satellite"].includes(key)) {
      wrap.className += " node-count-block";
      const stepper = el("div", undefined, "stepper");
      const formationCount=key==='satellite_count' && root.topology?.source==='formation';
      const min=formationCount ? 2 : spec.minimum ?? 1, max=formationCount ? 35 : spec.maximum ?? Infinity;
      if(formationCount) { input.min=min; input.max=max; }
      const slider = el("input"); slider.type = "range"; slider.min = min;
      // The slider is a convenience range; the numeric input retains the schema's full bounds.
      slider.max = Number.isFinite(max) ? max : Math.max(64, Number(value));
      slider.step = "1"; slider.value = value; slider.setAttribute("aria-label", `${title}滑块`);
      const paint = () => slider.style.setProperty("--range-progress", `${(Number(slider.value)-Number(slider.min))/(Number(slider.max)-Number(slider.min))*100}%`);
      paint();
      slider.addEventListener("input", () => { input.value = slider.value; paint(); });
      slider.addEventListener("change", commit);
      input.addEventListener("input", () => { if (input.checkValidity()) { slider.max = Math.max(Number(slider.max),Number(input.value)); slider.value = input.value; paint(); } });
      stepper.append(input, slider); wrap.append(stepper);
    } else if (["spacing_km", "altitude_km"].includes(key)) {
      wrap.className += " field-wide range-field";
      label.textContent = key === "spacing_km" ? "编队距离" : "轨道高度";
      const line = el("div", undefined, "field-line"), amount = el("div", undefined, "range-value");
      amount.append(input, el("span", "km")); line.append(label, amount);
      const slider = el("input"); slider.type = "range"; slider.className = "range"; slider.min = spec.minimum ?? 0.001; slider.max = spec.maximum ?? Math.max(value, key === "spacing_km" ? 10 : 1200); slider.step = key === "spacing_km" ? ".1" : "1"; slider.value = value; slider.setAttribute("aria-label", title);
      const updateRange = () => slider.style.setProperty("--range-progress", `${Math.max(0, Math.min(100, (Number(slider.value) - Number(slider.min)) / (Number(slider.max) - Number(slider.min)) * 100))}%`);
      updateRange();
      const endpoints = el("div", undefined, "range-labels"); endpoints.append(el("span", Number(slider.min) < 1 ? Number(slider.min)*1000 + " m" : slider.min + " km"), el("span", slider.max + " km"));
      slider.addEventListener("input", () => { updateRange(); input.value = slider.value; commit(); });
      input.addEventListener("input", () => { if(input.checkValidity()) { slider.value = input.value; updateRange(); } });
      wrap.replaceChildren(line, slider, endpoints, state.note);
    } else if (key === "layout") {
      wrap.className += " field-wide formation-field"; label.textContent = "编队构型";
      wrap.append(input);
    } else if (input.type === "checkbox") {
      const modes = el("div", undefined, "boolean-modes");
      modes.setAttribute("role", "group"); modes.setAttribute("aria-label", title);
      input.classList.add("segmented-source"); input.tabIndex = -1; input.setAttribute("aria-hidden", "true");
      for (const [checked, text] of [[true,"开启"],[false,"关闭"]]) {
        const choice = button(text, () => {
          input.checked = checked; commit();
          for (const item of modes.querySelectorAll("button")) item.setAttribute("aria-pressed", String(item === choice));
        }, "boolean-mode");
        choice.setAttribute("aria-pressed", String(input.checked === checked)); modes.append(choice);
      }
      wrap.append(input, modes);
    } else {
      const unit = title.match(/ \/ (?!含时区)(.+)$/)?.[1];
      if (unit && input.type === "number") { label.textContent = title.split(" / ")[0]; const row = el("div", undefined, "input-value"); row.append(input, el("span", unit, "field-unit")); wrap.append(row); }
      else wrap.append(input);
      if (["timestamp", "default_hardware_id", "resource_id", "source_id", "provenance", "max_total_num_tokens_source"].includes(key)) wrap.className += " field-wide";
    }
    if (nullable && !['detail_prediction_method','memory_policy_file'].includes(key)) {
      const caption = fixedBaudRate ? "改为自动选档" : ["max_batched_tokens","fast_kv_pool_blocks"].includes(key) ? "改为自动" : "清空";
      wrap.append(button(caption, () => { setAt(root, path, null); rerender(); }));
    }
    return wrap;
  }
  container.replaceChildren(...keys.flatMap(key => expand(String(key).split("."))).map(path => field(schemaAt(path), path)).filter(Boolean));
  enhanceSelects(container);
}

export function parseConfiguration(text) {
  const result = JSON.parse(text);
  if (!result || Array.isArray(result) || typeof result !== "object") throw new Error("配置必须是 JSON 对象");
  return result;
}
