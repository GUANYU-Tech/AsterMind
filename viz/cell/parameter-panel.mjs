import {renderConfigForm} from './config-form.mjs';
import {optionLabels} from './config-labels.mjs';
import {enhanceSelects} from '../assets/design-controls.js';

function element(tag, text) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  return node;
}

export function renderParameterGroup(container, config, paths, capabilities, changed, choices) {
  if (paths.includes('deployment.routes')) {
    renderPdEndpoints(container, config, changed, choices);
  } else if (paths.includes('hardware.su_group_sizes')) {
    renderIntraSatelliteNetwork(container, config, paths, capabilities, changed, choices);
  } else if (paths.length === 1 && paths[0] === 'deployment.placements') {
    renderRankMapping(container, config, changed, choices);
  } else {
    renderConfigForm(container, config, paths, capabilities.schema, changed, choices);
  }
}

function renderPdEndpoints(container, config, changed, choices) {
  const section = container.closest('details');
  const pd = config.compute_mode === 'detail_mode' && config.deployment.mode === 'pd_disagg';
  if (section.hidden && pd) section.open = true;
  section.hidden = !pd;
  container.replaceChildren();
  if (!pd) return;

  const pools = config.deployment.worker_pools;
  const roles = ['prefill', 'decode'];
  const selected = roles.map(role => pools.find(pool => pool.role === role));
  const validPair = pools.length === 2 && selected.every(Boolean) && pools[0].name !== pools[1].name;
  for (const [index, caption] of ['源 pool（Prefill）', '目标 pool（Decode）'].entries()) {
    const field = element('div'), label = element('label', caption), select = element('select');
    field.className = 'config-field';
    select.id = 'pd-' + roles[index] + '-pool';
    label.htmlFor = select.id;
    select.append(...pools.map(pool => new Option(pool.name, pool.name)));
    select.value = selected[index]?.name ?? '';
    select.disabled = !validPair;
    select.onchange = () => {
      if (select.value === selected[index].name) return;
      // The native KV route is always Prefill → Decode. Bind the chosen physical
      // pool to that role; placements and scheduler settings stay keyed by pool name.
      const [source, target] = [selected[1], selected[0]];
      source.role = 'prefill';
      target.role = 'decode';
      config.deployment.worker_pools = [source, target];
      const path = ['deployment', 'worker_pools'];
      choices.onEdit?.(path, config.deployment.worker_pools);
      changed(path);
    };
    field.append(label, select);
    container.append(field);
  }
  enhanceSelects(container);
}

function renderIntraSatelliteNetwork(container, config, paths, capabilities, changed, choices) {
  const type = config.topology.intra_satellite.type;
  const direct = type === 'ring' || type === 'fully_connected';
  const fields = paths.filter(path => path !== 'topology.intra_satellite');
  if (direct) fields.unshift('topology.intra_satellite.gateway_local_index');
  const description = element('div');
  description.className = 'topology-description';
  description.id = container.id + '-description';
  const profile = capabilities.su_profiles?.[config.hardware.default_hardware_id];
  const cards = config.hardware.accelerators_per_satellite;
  const formChoices = {...choices, labels:{...choices.labels}};
  let disableUplinks = false;
  if (type === 'profile_su' && profile) {
    const {domains, autoHint} = describeSuNetwork(description, config.hardware, profile);
    disableUplinks = domains === 1;
    formChoices.suLimit = profile.fast_gpu_limit_per_satellite;
    formChoices.autoSuGroups = autoHint;
    formChoices.labels.intra_satellite_bandwidth_gbps = 'SU 间上联带宽 / Gbps';
    formChoices.labels.intra_satellite_latency_ns = 'SU 间单跳时延 / ns';
  } else if (type === 'switch') {
    description.append(
      element('p', `每星 ${cards} 张卡各自连接同一个交换节点。卡间流量由该节点转发，星间流量也从这里接入卫星出口。`),
      element('p', '带宽设置用于每条链路及交换节点的共享容量，时延按单跳计算。'));
  } else if (direct) {
    const links = type === 'ring' ? (cards < 3 ? cards - 1 : cards) : cards * (cards - 1) / 2;
    description.append(
      element('p', type === 'ring' ? '各卡按编号连接相邻卡，三张及以上时首尾相连成环。' : '全连接为每一对卡建立一条直连链路。'),
      element('p', `当前每星 ${cards} 张卡，共 ${links} 条卡间直连链路。网关卡连接卫星出口，带宽和单跳时延按链路设置。`));
  } else {
    description.append(element('p', type === 'profile_su'
      ? '当前卡型缺少 SU 互联参数，请在星座资源中选择支持的卡型。'
      : '无法识别配置中的星内互联方式。'));
  }
  if (type !== 'profile_su') {
    // Keep the graph in a saved experiment until the user converts its working copy.
    description.prepend(element('strong', optionLabels[type] || type));
    const convert = element('button', '改用 SU 分域互联');
    convert.type = 'button';
    convert.className = 'small-action';
    convert.onclick = () => {
      const path = ['topology', 'intra_satellite', 'type'];
      config.topology.intra_satellite.type = 'profile_su';
      choices.onEdit?.(path, 'profile_su', type);
      changed(path);
      container.querySelector('#field-hardware-su_group_sizes')?.focus();
    };
    description.append(convert);
  }
  renderConfigForm(container, config, fields, capabilities.schema, changed, formChoices);
  container.prepend(description);
  container.setAttribute('aria-describedby', description.id);
  if (disableUplinks) {
    for (const key of ['intra_satellite_bandwidth_gbps', 'intra_satellite_latency_ns']) {
      container.querySelector(`#field-hardware-${key}`).disabled = true;
    }
  }
}

function describeSuNetwork(description, hw, profile) {
  const cards = hw.accelerators_per_satellite;
  const limit = profile.fast_gpu_limit_per_satellite;
  const groups = hw.su_group_sizes || [];
  const autoCount = Math.ceil(cards / limit);
  const domains = groups.length || autoCount;
  const autoSize = i => Math.min(limit, cards - i * limit);
  const autoHint = autoCount <= 8 ? Array.from({length:autoCount}, (_, i) => autoSize(i)).join(' + ') : '自动分组';
  const overview = element('p', '每星 ');
  for (let i = 0; i < Math.min(domains, 8); i++) {
    overview.append(element('span', `${i ? '、' : ''}SU ${i}（${groups[i] ?? autoSize(i)} 卡）`));
  }
  if (domains > 8) overview.append(element('span', `……共 ${domains} 个 SU 域`));
  const technology = {nvlink:'NVLink', hccs:'HCCS', metaxlink:'MetaXLink'}[profile.su_technology] || profile.su_technology;
  const bandwidth = profile.gpu_link_bandwidth_gbps || hw.intra_satellite_bandwidth_gbps;
  overview.append(element('span', `。域内使用 ${technology}，单向 ${bandwidth} Gbps。`));
  description.append(overview,
    element('p', `每个 SU 最多 ${limit} 张卡，总和须为每星 ${cards} 张卡。留空按每 ${limit} 张卡自动分组。`),
    element('p', domains > 1
      ? '每个 SU 通过一条上联接入同一个卫星出口。跨域通信经过源 SU 和目的 SU 的两条上联。下方可设置每条上联的单向带宽和单跳时延。'
      : '当前每星只有一个 SU 域，无需跨域上联。'));
  return {domains, autoHint};
}

function renderRankMapping(container, config, changed, choices) {
  const {satellite_ids: satellites, accelerators_per_satellite: cards} = config.hardware;
  const table = element('table');
  const wrap = element('div');
  wrap.className = 'cell-rank-table';
  wrap.dataset.fieldPath = JSON.stringify(['deployment','placements']);
  const header = table.createTHead().insertRow();
  for (const label of ['Rank','计算池 / 并行坐标','GPU','卫星']) header.append(element('th', label));
  const body = table.createTBody();
  const pools = config.deployment.worker_pools;
  const used = new Map();
  for (const [index, placement] of config.deployment.placements.entries()) {
    const {dp,pp,tp} = placement.parallel_coordinate;
    const pool = pools.find(p=>p.name===placement.worker_pool);
    const parallel = pool?.parallelism;
    const rank = parallel ? pools.slice(0,pools.indexOf(pool)).reduce((sum,p)=>sum+
      p.parallelism.tensor_parallel_size*p.parallelism.pipeline_parallel_size*(p.parallelism.data_parallel_size??1),0)
      +(dp*parallel.pipeline_parallel_size+pp)*parallel.tensor_parallel_size+tp : index;
    const row = body.insertRow();
    row.append(element('th', rank), element('td', `${placement.worker_pool} · DP${dp} PP${pp} TP${tp}`));
    const key = JSON.stringify([placement.satellite_id,placement.local_index]);
    const previous = used.get(key);
    if (previous) {
      previous.setAttribute('aria-invalid','true');
      row.setAttribute('aria-invalid','true');
    }
    used.set(key,row);
    for (const [key, label, options] of [
      ['local_index','GPU',Array.from({length:cards},(_,i)=>i)],
      ['satellite_id','卫星',satellites],
    ]) {
      const field = ['deployment','placements',index,key];
      const select = element('select');
      select.dataset.configPath = JSON.stringify(field);
      select.setAttribute('aria-label', `Rank ${rank} ${label}`);
      const empty = new Option('待分配', '');
      empty.disabled = true;
      select.append(empty, ...options.map(value=>new Option(key==='local_index'?`GPU ${value}`:value,value)));
      select.value = options.includes(placement[key]) ? placement[key] : '';
      select.onchange = () => {
        const previous = placement[key];
        const value = key==='local_index'?Number(select.value):select.value;
        placement[key]=value;
        choices.onEdit?.(field,value,previous);
        changed(field);
      };
      row.insertCell().append(select);
    }
  }
  wrap.append(table);
  const previous = container.querySelector('.cell-rank-table');
  container.replaceChildren(wrap);
  if (previous) wrap.scrollTo(previous.scrollLeft,previous.scrollTop);
}

export function renderSchedulingReadout(container, cell) {
  for (const field of container.querySelectorAll('[data-scheduler-value]')) {
    const key = field.dataset.schedulerValue;
    const value = cell?.[key];
    const runtimeKv = key==='num_kv_blocks' && cell?.scheduler_adapter==='vllm_0_30';
    field.textContent = runtimeKv ? '由 vLLM 0.30 在运行时分配'
      : value==null ? '自动 · 等待后端预检'
      : `自动 · ${Number(value).toLocaleString()} ${key==='num_kv_blocks'?'块':'Token'}`;
  }
}
