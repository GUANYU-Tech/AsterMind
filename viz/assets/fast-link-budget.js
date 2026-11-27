const linkTypes = [
  ['cross_layer', '计算星—通信星'],
  ['communication_isl', '通信星—通信星'],
  ['cell_isl', 'Cell 内部星间'],
  ['inter_cell', 'Cell 间直连'],
  ['sgl', '星地'],
];

const format = new Intl.NumberFormat(undefined, {maximumFractionDigits: 5});
const ratioLabel = '有效可用带宽比例';

function groundBudgets(config, profile) {
  if(config.network.optical_link_model){
    const profiles=config.network.link_profiles;
    return [['上行',[profiles.user_uplink.bandwidth_bps,profiles.feeder_uplink.bandwidth_bps]],
            ['下行',[profiles.user_downlink.bandwidth_bps,profiles.feeder_downlink.bandwidth_bps]]];
  }
  const shells = config.constellations.communication_shells;
  const kinds = new Set(config.ground_ingresses.map(ground => ground.ground_kind === 'user' ? 'user' : 'feeder'));
  const roles = config.network.ground_roles ?? [];
  return [['uplink', '上行'], ['downlink', '下行']].map(([direction, label]) => {
    const fallback = config.network.link_profiles[profile?.[`${direction}_profile`]]?.bandwidth_bps ?? profile?.bandwidth_bps;
    const caps = roles.includes('communication') ? shells.flatMap(shell =>
      Array.from(kinds, kind => shell.hardware_link_profiles?.[`${kind}_${direction}`]?.bandwidth_bps ?? fallback)) : [];
    if (!caps.length || roles.includes('compute')) caps.push(fallback);
    return [label, caps];
  });
}

function gbpsRange(values, ratio) {
  const low = format.format(Math.min(...values) * ratio / 1e9);
  const high = format.format(Math.max(...values) * ratio / 1e9);
  return low === high ? low : `${low}–${high}`;
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

export function createFastLinkBudget(host) {
  const fields = element('div', null, 'fields');
  host.append(element('h2', '链路带宽竞争（Fast Mode）'), fields);
  let config;
  const controls = new Map(linkTypes.map(([kind, title]) => {
    const card = element('div', null, 'fast-link-entry');
    const input = element('input');
    input.type = 'number';
    input.min = '0.01';
    input.max = '100';
    input.step = 'any';
    input.value = '100';
    input.required = true;
    input.dataset.designControl = 'number';
    input.setAttribute('aria-label', `${title} ${ratioLabel}`);
    const label = element('label', ratioLabel);
    const percent = element('span', null, 'fast-link-percent');
    percent.append(input, element('span', '%'));
    label.append(percent);
    const preview = element('output');
    card.append(element('h3', title), label, element('small', '带宽预算 → 有效带宽'), preview);
    fields.append(card);
    input.oninput = () => updateRow(kind);
    return [kind, {input, preview}];
  }));

  function updateRow(kind) {
    const {input, preview} = controls.get(kind);
    const enabled = kind !== 'inter_cell' || config.network.inter_cell_links;
    input.disabled = host.hidden || !enabled;
    preview.replaceChildren();
    if (!enabled) {
      preview.textContent = '当前未启用直连';
      return;
    }
    if (input.validity.badInput || input.value === '' || input.valueAsNumber < 0.01 || input.valueAsNumber > 100) return;
    const ratio = input.valueAsNumber / 100;
    const {link_profiles: profiles} = config.network;
    const profile = profiles[kind];
    let budgets = [['', [profile?.bandwidth_bps]]];
    if (kind === 'sgl') budgets = groundBudgets(config, profile);
    if (kind === 'communication_isl' && !config.network.optical_link_model && config.constellations.communication_shells.length) {
      budgets = [['', config.constellations.communication_shells.map(shell =>
        shell.hardware_link_profiles?.communication_isl?.bandwidth_bps
        ?? profiles[shell.isl_profile]?.bandwidth_bps ?? profile?.bandwidth_bps)]];
    }
    for (const [direction, caps] of budgets) {
      const values = caps.filter(value => Number.isFinite(value) && value > 0);
      if (!values.length) continue;
      const prefix = direction ? direction + ' ' : '';
      preview.append(element('span', `${prefix}${gbpsRange(values, 1)} → ${gbpsRange(values, ratio)} Gbps`));
    }
    if (!preview.childElementCount) preview.textContent = '—';
  }

  return {
    update(configuration, fast) {
      config = configuration;
      host.hidden = !fast;
      for (const kind of controls.keys()) updateRow(kind);
    },
    load(factors) {
      for (const [kind, {input}] of controls) {
        const factor = factors?.[kind] ?? 1;
        input.value = String(Number((factor * 100).toPrecision(15)));
      }
    },
    factors() {
      return Object.fromEntries(Array.from(controls, ([kind, {input}]) => [kind, input.valueAsNumber / 100])
        .filter(([, factor]) => factor > 0 && factor < 1));
    },
  };
}
