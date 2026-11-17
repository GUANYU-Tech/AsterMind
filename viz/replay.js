const $ = (id) => document.getElementById(id);
const esc = (value) =>
  String(value ?? "").replace(
    /[&<>"']/g,
    (ch) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[
        ch
      ])
  );
const ms = (ns) => `${(ns / 1e6).toFixed(ns < 1e6 ? 4 : 3)} ms`;
const duration = (ns) => (ns < 1e6 ? `${(ns / 1e3).toFixed(3)} μs` : ms(ns));
const exactTime = (ticks, ns) => {
  if (ticks == null) return ms(ns);
  const value=BigInt(ticks),part=(value%1000000000000000000n).toString().padStart(18,'0').replace(/0+$/,'');
  return `${value/1000000000000000000n}${part?'.'+part:''} s`;
};
const exactDuration = (row) => {
  if(row.start_ticks==null||row.end_ticks==null)return duration(row.end-row.start);
  const ns=Number(BigInt(row.end_ticks)-BigInt(row.start_ticks))/1e9;
  return ns<1000?`${Number(ns.toPrecision(9))} ns`:duration(ns);
};
const transportLabel = (data) => data.backend === "runtime_comm" ? "V1 Runtime Comm" : data.backend === "simgrid" ? "SimGrid" : data.backend === "ns3" ? "ns-3" : "FlowSim";
const communicationLabel = (data) => data.backend === 'runtime_comm' ? 'V1 Runtime Comm · 整条路径共享资源' : ({
  protobridge_cell: "Cell 内 ProtoBridge · Cell 外 FlowSim",
  protobridge_satellite: "卫星内 ProtoBridge · 星间 FlowSim",
}[data.provenance?.communication_mode] || (data.backend === "ns3" ? "ns-3 包级" : "全部通信使用 FlowSim")).replaceAll('FlowSim',data.backend==='simgrid'?'SimGrid':'FlowSim');
const bytes = (n) =>
  n >= 2 ** 20
    ? `${(n / 2 ** 20).toFixed(3)} MiB`
    : n >= 1024
    ? `${(n / 1024).toFixed(2)} KiB`
    : n > 0 && n < 0.001
    ? "< 0.001 B"
    : `${Number(n.toPrecision(6))} B`;
const rate = (n) =>
  n >= 1e9
    ? `${(n / 1e9).toFixed(2)} Gb/s`
    : n >= 1e6
    ? `${(n / 1e6).toFixed(2)} Mb/s`
    : `${(n / 1e3).toFixed(2)} kb/s`;
const empty = (text) => `<div class="empty">${esc(text)}</div>`;
const badge = (text, state = "") =>
  `<span class="badge ${state}">${esc(text)}</span>`;
const names = {
  VECTOR: "Vector",
  GEMM: "GEMM",
  FLASH_ATTN: "Attention",
  ATTENTION: "Attention",
  KV_CACHE: "KV cache",
  COMMUNICATION: "COMM",
};
const colors = {
  VECTOR: "#79a9d8",
  GEMM: "#79c2ff",
  FLASH_ATTN: "#b6a0ef",
  ATTENTION: "#b6a0ef",
  KV_CACHE: "#87cfcb",
  COMMUNICATION: "#f4bd70",
};
const state = {
  data: null,
  now: 0,
  selected: -1,
  flow: null,
  step: null,
  view: [0, 1],
  playing: false,
  hits: [],
  epoch: 0,
};
let catalog = [],
  transfers = new Map(),
  flowHops = new Map(),
  successors = [],
  resources = new Map(),
  pos = new Map(),
  lastFrame = 0,
  lastPaint = 0;
const kindPass = (c) =>
  !$("kindFilter").value ||
  ($("kindFilter").value === "comm" ? c.comm : !c.comm);
const filterPass = (c) =>
  (!$("cellFilter").value || c.cell === $("cellFilter").value) &&
  (!$("deviceFilter").value || c.device === $("deviceFilter").value) &&
  kindPass(c);
const active = (row, now = state.now) =>
  row.start != null && row.start <= now && now < row.end;
function fail(error) {
  $("error").hidden = false;
  $("error").textContent = `回放未能加载：${error.message || error}`;
  state.playing = false;
}
async function get(url) {
  const {loadReplayJson} = await import('/assets/replay_page_loader.js');
  return loadReplayJson(url);
}
function setTime(value) {
  if (!state.data) return;
  state.now = Math.max(0, Math.min(Number(value), state.limit));
  $("time").value = state.now;
  $("timeInput").value = (state.now / 1e6).toFixed(4);
  $("clock").textContent = ms(state.now);
  renderClock();
}
function setView(a, b) {
  const limit = state.limit;
  const width = Math.max(1000, Math.min(limit, b - a));
  a = Math.max(0, Math.min(a, limit - width));
  state.view = [a, a + width];
  drawTimeline();
}
function jump(time) {
  state.playing = false;
  $("play").textContent = "▶ 播放";
  setTime(time);
}
function fitCall(call) {
  const span = Math.max(call.end - call.start, 50000);
  setView(call.start - span * 0.15, call.end + span * 0.15);
}
function selectCall(index, { seek = true, fit = false } = {}) {
  if (!state.data.calls[index]) return;
  state.selected = index;
  const call = state.data.calls[index];
  state.step = state.data.steps.findIndex(
    (s) => s.cell === call.cell && s.id === call.batch
  );
  state.flow = call.phases.flat()[0] || null;
  updateCallOptions();
  $("callSelect").value = String(index);
  if (fit) fitCall(call);
  if (seek) jump(call.start + Math.min((call.end - call.start) / 2, 1000));
  renderCall();
  renderSteps();
  renderStep();
  drawTimeline();
  renderTopology();
}
function selectFlow(id) {
  const flow = transfers.get(id);
  if (!flow) return;
  state.flow = id;
  const callIndex = state.data.calls.findIndex((c) =>
    c.phases.some((p) => p.includes(id))
  );
  if (callIndex >= 0) {
    state.selected = callIndex;
    state.step = state.data.steps.findIndex(
      (s) =>
        s.cell === state.data.calls[callIndex].cell &&
        s.id === state.data.calls[callIndex].batch
    );
    renderCall();
  } else {
    state.selected = -1;
    state.step = -1;
    $("selectedType").textContent = "独立物理传输";
    $("callDetails").innerHTML = `<div class="call-name">${esc(
      flow.operation
    )} · ${esc(flow.id)}</div><div class="evidence">${esc(
      flow.evidence
    )} · ${esc(flow.source)} → ${esc(flow.target)} · ${bytes(
      flow.size_bytes
    )}</div>`;
    $("dependencies").innerHTML = empty(
      "此传输由请求或 KV 事务触发，不属于某个计算算子的依赖图。"
    );
    renderFlows();
  }
  updateCallOptions();
  renderSteps();
  renderStep();
  drawTimeline();
  renderTopology();
}
function selectStep(index) {
  const step = state.data.steps[index];
  state.step = index;
  $("cellFilter").value = step.cell;
  $("deviceFilter").value = "";
  updateCallOptions();
  const indexCall = state.data.calls.findIndex(
    (c) => c.cell === step.cell && c.batch === step.id
  );
  if (indexCall >= 0) selectCall(indexCall, { seek: false });
  setView(
    step.start - Math.max(1000, (step.end - step.start) * 0.05),
    step.end + (step.end - step.start) * 0.05
  );
  jump(step.start);
  renderStep();
  renderSteps();
}
function eventLabel(event) {
  const labels = {
    fault_injected: "链路故障注入",
    fault_restored: "故障修复已观测",
    capacity_observed: `服务容量观测 → ${rate(event.capacity || 0)}`,
    route_wait: "通信等待可用路由",
    route_recovery: "从缓存节点重试",
    transfer_cancelled: "当前跳中断并丢弃未交付数据",
    token_produced: `产出 token ${event.token_start ?? ""}`,
    link_state: `${
      event.state === "UP"
        ? "建链完成"
        : event.state === "DOWN"
        ? "链路断开"
        : "正在建链"
    } · ${event.link_id}`,
  };
  return labels[event.kind] || event.kind;
}
function renderMetrics() {
  const d = state.data;
  const completed = Object.values(d.requests).filter(
    (r) => r.status === "completed"
  ).length;
  const kernels = d.calls.filter((c) => !c.comm).length;
  const passed = d.audit.filter((c) => c.state === "pass").length;
  $("metrics").innerHTML = [
    [
      String(completed).padStart(2, "0"),
      "最终完成请求",
      `${Object.keys(d.requests).length} 请求 · ${Object.values(
        d.requests
      ).reduce((s, r) => s + r.received_tokens, 0)} tokens`,
    ],
    [
      kernels.toLocaleString(),
      "计算 / 存储算子调用",
      `${d.calls.length - kernels} 个通信操作 · ${d.page_scope ? '当前批次的 GPU 汇总' : '跨批次和 GPU 汇总'}`,
    ],
    [
      d.transfers.length.toLocaleString(),
      "端到端消息记录",
      `${d.transfers.filter(t => t.backend !== "protobridge").length} ${transportLabel(d)} / ${d.transfers.filter(t => t.backend === "protobridge").length} ProtoBridge · ${d.hops.length.toLocaleString()} 逐段服务记录`,
    ],
    [
      `${passed}/${d.audit.length}`,
      "记录约束核查",
      d.status === "completed" ? "执行已完成" : "执行含传输失败",
    ],
  ]
    .map(
      ([n, a, b]) =>
        `<div class="metric"><strong>${esc(n)}</strong><span>${esc(
          a
        )}<small>${esc(b)}</small></span></div>`
    )
    .join("");
}
function renderMoments() {
  const d = state.data;
  const items = [];
  for (const e of d.events) {
    if (
      [
        "fault_injected",
        "fault_restored",
        "route_wait",
        "capacity_observed",
      ].includes(e.kind)
    )
      items.push({
        time: e.time,
        label: eventLabel(e),
        bad: e.kind === "fault_injected",
      });
    if (
      e.kind === "link_state" &&
      e.state === "UP" &&
      e.time > 1e8 &&
      e.time < d.end
    )
      items.push({ time: e.time, label: "链路重新就绪" });
  }
  for (const request of Object.values(d.requests)) {
    if (request.first_token_received_ns != null)
      items.push({
        time: request.first_token_received_ns,
        label: "用户首 token",
      });
  }
  for (const m of Object.values(d.migrations)) {
    if (m.network_complete_time_s)
      items.push({
        time: m.network_complete_time_s * 1e9,
        label: "KV 全分片到齐",
      });
    items.push({
      time: m.completed_time_s * 1e9,
      label: m.state === "completed" ? "迁移提交" : "迁移回滚",
      bad: m.state !== "completed",
    });
  }
  const seen = new Set();
  $("moments").innerHTML = items
    .sort((a, b) => a.time - b.time)
    .filter((x) => {
      const k = x.label + x.time;
      if (seen.has(k)) return false;
      seen.add(k);
      return true;
    })
    .slice(0, 10)
    .map(
      (x) =>
        `<button type="button" data-time="${x.time}" class="${
          x.bad ? "danger" : ""
        }">${esc(x.label)} · ${ms(x.time)}</button>`
    )
    .join("");
}
function geometry() {
  const d = state.data;
  pos = new Map();
  const ground = new Set(d.ground);
  const owners = [...new Set(d.devices.map((n) => n.satellite))];
  const stars = owners.filter((o) => !ground.has(o));
  let boxes = "";
  const width =
    stars.length <= 2 ? 340 : Math.max(210, 760 / stars.length - 20);
  for (const [index, owner] of stars.entries()) {
    const left =
      stars.length <= 2 ? 190 + index * 420 : 190 + index * (width + 20);
    const mirror = index % 2 === 1;
    boxes += `<rect class="topology-owner" x="${left}" y="18" width="${width}" height="270" rx="10"/><text class="owner-label" x="${
      left + 15
    }" y="40">${esc(owner)}</text>`;
    const devices = d.devices.filter((n) => n.satellite === owner);
    const grouped = {
      gpu: devices.filter((n) => n.kind === "gpu"),
      switch: devices.filter((n) => n.kind === "switch"),
      terminal: devices.filter((n) => n.kind === "terminal"),
    };
    for (const [kind, list] of Object.entries(grouped))
      for (const [i, node] of list.entries()) {
        const x =
          kind === "gpu"
            ? mirror
              ? left + width - 50
              : left + 50
            : kind === "switch"
            ? left + width / 2
            : mirror
            ? left + 55
            : left + width - 55;
        const y =
          list.length === 1
            ? 153
            : 85 + i * (150 / Math.max(1, list.length - 1));
        pos.set(node.id, { x, y, node });
      }
  }
  const gnodes = d.devices.filter((n) => ground.has(n.satellite));
  if (gnodes.length) {
    boxes += `<rect class="topology-owner" x="15" y="70" width="130" height="170" rx="10"/><text class="owner-label" x="29" y="94">地面用户</text>`;
    gnodes.forEach((node, i) =>
      pos.set(node.id, {
        x: 80,
        y:
          gnodes.length === 1
            ? 155
            : 119 + (i * 85) / Math.max(1, gnodes.length - 1),
        node,
      })
    );
  }
  return boxes;
}
function latestLinkState(resource) {
  let status = "UP";
  for (const e of state.data.events) {
    if (e.time > state.now) break;
    if (e.kind === "link_state" && e.link_id === resource) status = e.state;
  }
  return status;
}
function renderTopology() {
  if (!state.data) return;
  const d = state.data;
  let svg = geometry();
  const selected = transfers.get(state.flow);
  const selectedPairs = new Set();
  if (selected)
    for (let i = 1; i < selected.path.length; i++)
      selectedPairs.add(
        [selected.path[i - 1], selected.path[i]].sort().join("|")
      );
  const currentHops = d.hops.filter((h) => active(h));
  const currentCalls = d.calls.filter((c) => active(c));
  for (const link of d.links) {
    const a = pos.get(link.source),
      b = pos.get(link.target);
    if (!a || !b) continue;
    const external = a.node.satellite !== b.node.satellite;
    const res = [...resources.values()].find(
      (r) =>
        [r.source, r.target].sort().join("|") ===
        [link.source, link.target].sort().join("|")
    );
    const linkState = res ? latestLinkState(res.id) : "UP";
    const serving = currentHops.filter(
      (h) =>
        [h.source, h.target].sort().join("|") ===
        [link.source, link.target].sort().join("|")
    );
    const pair = [link.source, link.target].sort().join("|");
    const ground =
      d.ground.includes(a.node.satellite) ||
      d.ground.includes(b.node.satellite);
    const path = ground
      ? `M ${a.x},${a.y} C ${a.x},325 ${b.x},325 ${b.x},${b.y}`
      : `M ${a.x},${a.y} L ${b.x},${b.y}`;
    const cls =
      linkState === "DOWN"
        ? "down"
        : serving.length
        ? "active"
        : selectedPairs.has(pair)
        ? "selected"
        : "";
    svg += `<path class="topology-link ${cls}" d="${path}"/><path class="link-hit" d="${path}" data-link="${esc(
      link.id
    )}" tabindex="0" role="button" aria-label="链路 ${esc(link.id)}"/>`;
    if (external) {
      const x = (a.x + b.x) / 2;
      const y = ground ? 333 : (a.y + b.y) / 2 - 12;
      const label =
        linkState === "DOWN"
          ? "DOWN · 故障"
          : linkState === "SETTING_UP"
          ? "建链中"
          : serving.length
          ? `${serving.length} 流 · 单流最高 ${rate(
              Math.max(...serving.map((h) => h.rate))
            )}`
          : "UP";
      svg += `<text class="link-label ${
        serving.length ? "live" : ""
      }" x="${x}" y="${y}" text-anchor="middle">${esc(label)}</text>`;
    }
  }
  for (const [id, p] of pos) {
    const computing = currentCalls.filter((c) => !c.comm && c.device === id);
    const waiting = d.transfers.some(
      (t) => active(t) && (t.source === id || t.target === id)
    );
    const entry = Object.entries(d.cells)
      .flatMap(([cell, v]) =>
        v.config.placements.map((pl) => ({ ...pl, cell }))
      )
      .find((pl) => pl.device === id);
    const role = entry
      ? `${entry.cell} / rank ${entry.rank}`
      : p.node.kind === "switch"
      ? "共享交换容量"
      : p.node.kind === "terminal"
      ? "物理终端"
      : "";
    const w = p.node.kind === "gpu" ? 91 : 80;
    svg += `<g class="device-node" data-device="${esc(
      id
    )}" tabindex="0" role="button" aria-label="${esc(
      id + " " + role
    )}"><rect class="device-box ${
      computing.length ? "active" : waiting ? "comm" : ""
    }" x="${p.x - w / 2}" y="${p.y - 22}" width="${w}" height="44" rx="${
      p.node.kind === "terminal" ? 18 : 6
    }"/><text class="device-label" x="${p.x}" y="${
      p.y - 2
    }" text-anchor="middle">${esc(id)}</text><text class="device-role" x="${
      p.x
    }" y="${p.y + 13}" text-anchor="middle">${esc(
      computing.length
        ? "计算中 " + computing[0].kind
        : waiting
        ? "通信进行中"
        : p.node.kind.toUpperCase()
    )}</text><text class="device-role" x="${p.x}" y="${
      p.y + 38
    }" text-anchor="middle">${esc(role)}</text></g>`;
  }
  $("topology").innerHTML = svg;
  $("networkBadge").textContent = `${communicationLabel(d)} · 同一时间轴`;
}
function renderNow() {
  const d = state.data;
  const calls = d.calls.filter((c) => active(c)),
    flows = d.transfers.filter((t) => active(t));
  const waits = d.events.filter(
    (e) =>
      e.kind === "route_wait" &&
      e.time <= state.now &&
      !d.events.some(
        (r) =>
          r.kind === "route_recovery" &&
          r.request_id === e.request_id &&
          r.time > e.time &&
          r.time <= state.now
      )
  );
  const faults = d.events.filter(
    (e) =>
      e.kind === "fault_injected" &&
      e.time <= state.now &&
      !d.events.some(
        (r) =>
          r.kind === "fault_restored" &&
          r.target_id === e.target_id &&
          r.time >= e.time &&
          r.time <= state.now
      )
  );
  $("nowBadge").className = `badge ${
    faults.length ? "fail" : flows.length ? "wait" : ""
  }`;
  $("nowBadge").textContent = faults.length
    ? "链路中断"
    : calls.length || flows.length
    ? "执行中"
    : "无活动操作";
  let text;
  if (waits.length)
    text = `${waits.length} 条传输在等待路由恢复。此时可执行的本地算子继续按依赖推进；PP 接收端尚不能释放后继依赖。`;
  else if (faults.length)
    text = `链路故障已观测；${flows.length} 条传输尚未终结。是否重试由该场景的实际事件决定。`;
  else if (calls.length)
    text = `${calls.filter((c) => !c.comm).length} 个计算算子执行中，${
      calls.filter((c) => c.comm).length
    } 个 COMM 等待物理阶段完成。当前有 ${flows.length} 条端到端传输。`;
  else if (flows.length)
    text = `${flows.length} 条物理传输进行中；算子按数据与依赖的就绪时刻启动。`;
  else
    text =
      state.now >= d.end
        ? "已到达该运行的最后一个业务事件。最终结果与完整核查可在下方查看。"
        : "当前没有执行中的算子或传输，等待请求到达或下一个调度事件。";
  $("nowSummary").textContent = text;
  $("requests").innerHTML = Object.entries(d.requests)
    .map(([id, r]) => {
      const input = d.request_inputs.find((x) => x.id === id);
      let stage =
        state.now < r.arrival_ns
          ? "尚未到达"
          : state.now < (r.cell_ready_ns ?? Infinity)
          ? "输入传输"
          : state.now < (r.compute_complete_ns ?? Infinity)
          ? "Cell 计算 / 通信"
          : state.now < (r.completed_ns ?? Infinity)
          ? "输出交付"
          : r.status === "completed"
          ? "已完成"
          : "失败";
      const received = d.transfers.filter(
        (t) =>
          t.operation === "request_output" &&
          t.correlation_id === id &&
          t.kind === "completed" &&
          t.end <= state.now
      ).length;
      const frac = Math.max(
        0,
        Math.min(
          1,
          (state.now - r.arrival_ns) /
            Math.max(1, (r.completed_ns ?? d.end) - r.arrival_ns)
        )
      );
      return `<div class="request-card"><div class="request-head"><strong>${esc(
        id
      )} <span class="muted">→ ${esc(r.cell)}</span></strong>${badge(
        stage,
        stage === "已完成" ? "pass" : ""
      )}</div><div class="request-track"><div class="request-progress" style="width:${
        frac * 100
      }%"></div></div><div class="request-times"><span>${
        input?.input_tokens ?? "?"
      } prompt · 已交付 ${received} 输出块 / 最终 ${
        r.received_tokens
      } tokens</span><span>TTFT ${
        r.first_token_received_ns != null
          ? duration(r.first_token_received_ns - r.arrival_ns)
          : "—"
      }（最终）</span></div></div>`;
    })
    .join("");
  const recent = d.events
    .filter(
      (e) => e.time <= state.now && !(e.kind === "link_state" && e.time === 0)
    )
    .slice(-5)
    .reverse();
  $("eventFeed").innerHTML =
    recent
      .map(
        (e) =>
          `<div class="event-row ${
            e.kind.includes("fault") || e.kind === "transfer_cancelled"
              ? "failure"
              : ""
          }"><time>${ms(e.time)}</time><span>${esc(eventLabel(e))}</span></div>`
      )
      .join("") || empty("尚无关键事件");
}
function drawTimeline() {
  if (!state.data) return;
  const canvas = $("timeline"),
    dpr = window.devicePixelRatio || 1;
  const d = state.data;
  const lanes = Object.entries(d.cells).flatMap(([cell, v]) =>
    v.config.placements
      .filter(
        (p) =>
          (!$("cellFilter").value || cell === $("cellFilter").value) &&
          (!$("deviceFilter").value || p.device === $("deviceFilter").value)
      )
      .map((p) => ({ ...p, cell }))
  );
  const width = canvas.parentElement.clientWidth,
    height = 52 + lanes.length * 53;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);
  const left = 124,
    right = 15,
    plot = width - left - right,
    [a, b] = state.view;
  const x = (t) => left + ((t - a) / (b - a)) * plot;
  ctx.font = "10px Consolas,monospace";
  ctx.textBaseline = "middle";
  const tickCount = Math.max(2, Math.min(6, Math.floor(plot / 140)));
  for (let k = 0; k <= tickCount; k++) {
    const time = a + ((b - a) * k) / tickCount,
      px = x(time);
    ctx.strokeStyle = "#24364b";
    ctx.beginPath();
    ctx.moveTo(px, 28);
    ctx.lineTo(px, height - 15);
    ctx.stroke();
    ctx.fillStyle = "#839cb4";
    ctx.textAlign = k === 0 ? "left" : k === tickCount ? "right" : "center";
    ctx.fillText(ms(time), px, 12);
  }
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, 27, plot, height - 42);
  ctx.clip();
  for (const event of d.events.filter((e) => e.kind === "fault_injected")) {
    const restored = d.events.find(
      (e) =>
        e.kind === "fault_restored" &&
        e.target_id === event.target_id &&
        e.time >= event.time
    );
    const start = Math.max(a, event.time),
      end = Math.min(b, restored?.time ?? d.horizon);
    if (end > start) {
      ctx.fillStyle = "#f17e8716";
      ctx.fillRect(x(start), 27, x(end) - x(start), height - 42);
    }
  }
  ctx.restore();
  state.hits = [];
  lanes.forEach((lane, index) => {
    const y = 36 + index * 53;
    ctx.textAlign = "left";
    ctx.fillStyle = "#b4cce2";
    ctx.fillText(lane.device, 0, y + 7);
    ctx.fillStyle = "#627e9a";
    ctx.fillText(`${lane.cell} · rank ${lane.rank}`, 0, y + 21);
    ctx.fillStyle = "#0b1522";
    ctx.fillRect(left, y, plot, 38);
    const rows = d.calls
      .map((c, i) => ({ c, i }))
      .filter(
        ({ c }) =>
          c.cell === lane.cell &&
          (c.device === lane.device ||
            (c.comm &&
              c.phases.flat().some((id) => {
                const f = transfers.get(id);
                return (
                  f && (f.source === lane.device || f.target === lane.device)
                );
              }))) &&
          kindPass(c) &&
          c.end >= a &&
          c.start <= b
      );
    for (const { c, i } of rows) {
      const px = Math.max(left, x(c.start)),
        end = Math.min(width - right, x(c.end)),
        barWidth = Math.max(0.8, end - px),
        py = y + (c.comm ? 23 : 3),
        barHeight = c.comm ? 11 : 17;
      ctx.fillStyle = colors[c.kind] || (c.comm ? "#f4bd70" : "#7cbde6");
      ctx.globalAlpha =
        state.selected === i
          ? 1
          : state.selected >= 0 && d.calls[state.selected].batch !== c.batch
          ? 0.55
          : 0.85;
      ctx.fillRect(px, py, barWidth, barHeight);
      ctx.globalAlpha = 1;
      if (state.selected === i) {
        ctx.strokeStyle = "#fff0c5";
        ctx.lineWidth = 1.7;
        ctx.strokeRect(px - 0.7, py - 0.7, barWidth + 1.4, barHeight + 1.4);
      }
      if (barWidth > 65) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(px + 3, py, barWidth - 6, barHeight);
        ctx.clip();
        ctx.fillStyle = "#0d1c29";
        ctx.font = "9px Consolas,monospace";
        ctx.fillText(c.comm ? "COMM" : c.kind, px + 4, py + barHeight / 2);
        ctx.restore();
        ctx.font = "10px Consolas,monospace";
      }
      state.hits.push({ x: px, y: py, w: barWidth, h: barHeight, index: i });
    }
    ctx.strokeStyle = "#26374b";
    ctx.beginPath();
    ctx.moveTo(0, y + 44);
    ctx.lineTo(width, y + 44);
    ctx.stroke();
  });
  if (state.now >= a && state.now <= b) {
    const px = x(state.now);
    ctx.strokeStyle = "#ebf7ff";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px, 24);
    ctx.lineTo(px, height - 13);
    ctx.stroke();
    ctx.fillStyle = "#ebf7ff";
    ctx.beginPath();
    ctx.moveTo(px - 4, 23);
    ctx.lineTo(px + 4, 23);
    ctx.lineTo(px, 28);
    ctx.fill();
  }
  $("windowInfo").textContent = `视窗 ${ms(a)} — ${ms(b)} · ${
    lanes.length
  } 个 GPU rank`;
}
function renderCall() {
  const c = state.data.calls[state.selected];
  if (!c) {
    $("callDetails").innerHTML = empty("选择一个原生算子。");
    return;
  }
  $("selectedType").textContent = c.comm
    ? (c.communication_backend === "protobridge" ? "ProtoBridge 通信屏障" : `${transportLabel(state.data)} 通信屏障`)
    : names[c.kind] || c.kind;
  const depEnd = Math.max(
    c.host_ready,
    ...c.deps.filter((i) => i >= 0).map((i) => state.data.calls[i].end + c.gap)
  );
  $("callDetails").innerHTML = `<div class="call-name">${esc(
    c.id
  )} <span class="muted">/ occurrence ${
    c.occurrence
  }</span></div><div class="facts">${[
    [
      c.comm ? "参与 GPU" : "映射",
      c.surrogate ? c.surrogate.devices.join(" ↔ ") : c.comm
        ? [
            ...new Set(
              c.phases.flat().flatMap((id) => {
                const f = transfers.get(id);
                return f ? [f.source, f.target] : [];
              })
            ),
          ].join(" ↔ ")
        : `${c.device} · ${c.owner} · r${c.rank}`,
    ],
    ["仿真记录耗时", exactDuration(c)],
    ["求解方法", c.comm ? (c.communication_backend === "protobridge" ? "ProtoBridge · 当前物理链路参数" : `${transportLabel(state.data)} · 逐流执行`) : "AsterMind GPU 计算模型"],
    ["原生参考", duration(c.native_ns) + (c.comm ? " · 未额外计费" : "")],
    ["开始 → 结束", `${exactTime(c.start_ticks,c.start)} → ${exactTime(c.end_ticks,c.end)}`],
    ["直接依赖", `${c.deps.length} 个 · launch gap ${duration(c.gap)}`],
    ["依赖就绪后的间隔", duration(Math.max(0, c.start - depEnd))],
  ]
    .map(
      ([a, b]) =>
        `<div class="fact"><span>${esc(a)}</span><strong>${esc(
          b
        )}</strong></div>`
    )
    .join("")}</div><div class="evidence">${esc(c.evidence)} · ${esc(
    c.cell + "/" + c.batch
  )}</div>`;
  const short = (call) =>
    call.id.replace(/^main:\d+:/, "").replace(/:tp-rank-/g, ":r");
  const depButton = (i) =>
    `<button class="dep" data-call="${i}" type="button">${esc(
      short(state.data.calls[i])
    )}<br><span class="muted">结束 ${ms(
      state.data.calls[i].end
    )}</span></button>`;
  $("dependencies").innerHTML = `<div class="dep-col">${
    c.deps
      .filter((i) => i >= 0)
      .slice(0, 6)
      .map(depButton)
      .join("") || "<small>输入 / 主机就绪</small>"
  }${
    c.deps.length > 6
      ? `<small>其余 ${c.deps.length - 6} 个前驱见源记录</small>`
      : ""
  }</div><div class="dep-col"><div class="dep current">${esc(
    short(c)
  )}<br>${duration(c.end - c.start)}</div></div><div class="dep-col">${
    (successors[state.selected] || []).slice(0, 5).map(depButton).join("") ||
    "<small>批次输出 / 无直接后继</small>"
  }</div>`;
  renderFlows();
}
function renderFlows() {
  const c = state.data.calls[state.selected];
  const ids = c?.comm ? c.phases.flat() : state.flow ? [state.flow] : [];
  let content = "";
  if (!ids.length)
    content = empty(
      c?.comm && c.communication_backend === "protobridge"
        ? "此通信使用 ProtoBridge，未生成逐流传输记录。"
        : c?.comm ? "该通信没有记录可展示的逐跳传输。" : "该算子为本地计算或存储操作；选择 COMM 条带可查看实际传输阶段。"
    );
  else {
    for (const [phaseIndex, phase] of (c?.comm
      ? c.phases
      : [[state.flow]]
    ).entries()) {
      content += `<p class="muted">阶段 ${phaseIndex + 1} · ${
        phase.length
      } 条真实传输</p>`;
      for (const id of phase) {
        const t = transfers.get(id);
        if (!t) continue;
        content += `<div class="flow-row ${
          state.flow === id ? "active" : ""
        }" role="button" tabindex="0" data-flow="${esc(
          id
        )}"><div><div class="flow-name">${esc(t.operation)} · ${esc(
          id
        )}</div><div class="flow-route">${esc(
          t.path.join(" → ")
        )}</div></div><div class="flow-bytes ${
          t.kind === "failed" ? "flow-failed" : ""
        }">${bytes(t.size_bytes)}<br><span class="muted">${
          t.start != null ? duration(t.end - t.start) : "未记录 ready"
        } · ${t.kind === "completed" ? "交付" : "失败"}</span></div></div>`;
      }
    }
  }
  const t = transfers.get(state.flow);
  if (t) {
    const hops = flowHops.get(t.id) || [];
    content += `<div class="evidence">${esc(t.evidence)} · ${
      t.start == null ? "就绪时刻未记录" : `就绪 ${exactTime(t.start_ticks,t.start)}`
    } → 通知 ${exactTime(t.end_ticks,t.end)}<br>物理终态 ${
      t.physical_end != null ? ms(t.physical_end) : "未记录"
    } · ${esc(t.reason || "依赖在实际交付后释放")}</div>`;
    content += `<div class="scroll-table"><table class="hop-table"><thead><tr><th>服务资源 / 方向</th><th>区间 / ms</th><th>实际服务 / 容量</th><th>逻辑字节</th></tr></thead><tbody>${hops
      .map(
        (h) =>
          `<tr><td>${esc(
            h.source === h.target
              ? `${h.source} 容量门`
              : `${h.source} → ${h.target}`
          )}</td><td title="${exactTime(h.start_ticks,h.start)} → ${exactTime(h.end_ticks,h.end)}">${(
            h.start / 1e6
          ).toFixed(4)} → ${(h.end / 1e6).toFixed(4)}<br>${exactDuration(h)}</td><td>${rate(
            h.rate
          )} / ${rate(h.capacity)}</td><td title="逻辑字节原值：${
            h.bytes
          } B">${esc(bytes(h.bytes))}</td></tr>`
      )
      .join("")}</tbody></table></div>`;
    if (!hops.length)
      content += empty(
        "此传输没有已记录的有效服务区间，可能仍在等待或在发送前失败。"
      );
  }
  $("flowDetails").innerHTML = content;
}
function renderSteps() {
  $("steps").innerHTML = state.data.steps
    .map(
      (s, i) =>
        `<button class="step-row ${state.step === i ? "selected" : ""} ${
          active(s) ? "live" : ""
        }" type="button" data-step="${i}"><div class="step-top"><span>${esc(
          s.cell
        )} / ${esc(s.id)}</span><span>${duration(
          s.end - s.start
        )}</span></div><div class="step-body">${s.requests
          .map(
            (r) =>
              `${esc(r.request_id)} · ${esc(r.phase)} · 调度 ${
                r.input_tokens
              } prefill / ${r.output_tokens} decode · 实际产出 ${
                s.generated[r.request_id] ?? 0
              } token`
          )
          .join("<br>")}<br>${ms(s.start)} → ${ms(s.end)}</div></button>`
    )
    .join("");
  $(
    "schedulerBadge"
  ).textContent = `vLLM 0.31 · ${state.data.steps.length} 批次`;
}
function renderStep() {
  const s = state.data.steps[state.step];
  if (!s) {
    $("stepDetail").innerHTML = empty("点击批次查看本批 token 和 KV 事件。");
    return;
  }
  $("stepDetail").innerHTML = `<strong>${esc(
    s.cell + " / " + s.id
  )}</strong>${s.requests
    .map(
      (r) =>
        `<p>${esc(r.request_id)} · ${esc(r.phase)}<br>已计算 ${
          r.computed_tokens
        } · 本批 prefill ${r.input_tokens} / decode ${
          r.output_tokens
        }<br>实际产出 ${
          s.generated[r.request_id] ?? 0
        } 个 token<br>Scheduler 前缀命中 ${
          r.scheduler_prefix_hit_tokens ?? 0
        } · 重计算 ${r.recomputed_tokens ?? 0}</p>`
    )
    .join("")}<div>${s.kv
    .map(
      (e) =>
        `<span class="kv-event">${esc(e.action)} · ${bytes(e.bytes)} · tokens ${
          e.token_start
        }–${e.token_end}</span>`
    )
    .join("")}</div><div class="evidence">${esc(s.evidence)}</div>`;
}
function renderMigrations() {
  const entries = Object.values(state.data.migrations);
  $("migrationBadge").textContent = entries.length
    ? `${entries.length} 个事务`
    : "该场景无迁移";
  $("migrations").innerHTML =
    entries
      .map((m) => {
        const complete = m.state === "completed",
          start = (m.start_time_s ?? m.requested_time_s) * 1e9,
          end = m.completed_time_s * 1e9,
          net =
            m.network_complete_time_s != null
              ? m.network_complete_time_s * 1e9
              : null;
        const begun = state.now >= start,
          finished = state.now >= end;
        const readReady = m.source_read_complete_ns;
        const stage = !begun
          ? -1
          : readReady != null && state.now < readReady
          ? 0
          : net != null && state.now >= net
          ? finished
            ? 3
            : 2
          : 1;
        const labels = [
          "源读取 / 准备",
          "逐 rank 发送",
          complete ? "全部到齐 / 目标写入" : "分片失败 / 屏障未满足",
          complete ? "提交 / 源释放" : "目标回滚 / 源缓存保留",
        ];
        const stageClass = (i) =>
          !begun
            ? ""
            : !complete && finished
            ? i === 0 || i === 3
              ? "done"
              : "failed"
            : i < stage
            ? "done"
            : i === stage
            ? stage === 3
              ? "done"
              : "live"
            : "";
        return `<div class="migration"><div class="migration-head"><strong>${esc(
          m.id
        )} · ${esc(m.source)} → ${esc(m.target)} · ${esc(
          m.mode || ""
        )} · ${bytes(m.bytes || 0)}</strong>${badge(
          !begun
            ? "尚未开始"
            : !finished
            ? "进行中"
            : complete
            ? "提交完成"
            : "已回滚",
          finished ? (complete ? "pass" : "fail") : begun ? "wait" : ""
        )}</div><div class="migration-stages">${labels
          .map(
            (label, i) =>
              `<div class="migration-stage ${stageClass(i)}">${i + 1}. ${esc(
                label
              )}<time>${
                i === 0
                  ? `${ms(start)} → ${
                      readReady != null ? ms(readReady) : "未记录读完时刻"
                    }`
                  : i === 2
                  ? net != null
                    ? ms(net)
                    : "未发生写入 / 提交"
                  : i === 3
                  ? ms(end)
                  : "每分片实际事件"
              }</time></div>`
          )
          .join("")}</div>${(m.partitions || [])
          .map(
            (p) =>
              `<div class="flow-row" role="button" tabindex="0" data-flow="${esc(
                p.transfer_id
              )}"><span class="flow-name">rank ${p.rank} · ${esc(
                p.source
              )} → ${esc(p.target)}</span><span class="flow-bytes">${bytes(
                p.bytes
              )} · ${
                state.now < (p.terminal_time_s ?? Infinity) * 1e9
                  ? "尚未终结"
                  : p.state === "completed"
                  ? "已到齐"
                  : "失败"
              }</span></div>`
          )
          .join("")}<p class="footnote">${
          complete
            ? `全部分片屏障 ${ms(net)}；写入后提交 ${ms(end)}。${
                m.source_release
                  ? `最终源删除 ${m.source_release.removed_blocks} 个缓存块。`
                  : ""
              }`
            : `${esc(m.reason || m.state)}；目标保留与源缓存见最终原生快照。`
        }</p></div>`;
      })
      .join("") || empty("该运行只包含原生计算和通信，没有 KV 迁移事务。");
}
function renderMemory() {
  $("memory").innerHTML = Object.entries(state.data.cells)
    .map(([id, cell]) => {
      const storage = cell.storage,
        c = storage.capacity;
      const analytical = c.source_mode !== "debug_explicit_block_count";
      const synthetic = JSON.stringify(c.policy || {}).includes("Synthetic");
      return `<div class="memory-card"><h3>${esc(id)} · ${esc(
        cell.config.hardware_id
      )} ${badge(
        analytical
          ? synthetic
            ? "合成策略账目验证"
            : "绑定 HBM 策略"
          : "显式测试 KV 池",
        analytical ? "" : "wait"
      )}</h3><div class="memory-line"><span>模型 / 部署</span><strong>${esc(
        cell.config.model.split("/").pop()
      )} · TP${cell.config.parallel.tp} / PP${
        cell.config.parallel.pp
      }</strong></div><div class="memory-line"><span>每 rank KV block / 逻辑池块数</span><strong>${bytes(
        c.kv_block_bytes
      )} / ${c.num_kv_blocks}</strong></div>${
        analytical
          ? `<div class="memory-line"><span>物理 HBM / 可供 KV 字节</span><strong>${bytes(
              c.physical_hbm_bytes
            )} / ${bytes(
              c.available_kv_bytes
            )}</strong></div><div class="memory-line"><span>权重驻留 / 非 KV 预留</span><strong>${bytes(
              c.weight_resident_bytes
            )} / ${bytes(c.non_kv_reserve_bytes)}</strong></div>`
          : ""
      }<div class="memory-line"><span>最终已缓存 / 最终迁移保留</span><strong>${
        storage.ready_cached_blocks
      } / ${
        storage.migration_reserved_blocks
      } blocks</strong></div><div class="footnote">最终原生快照，非滑块时刻的实时显存。${
        !analytical
          ? "测试池大小不能作为硬件 HBM 上限。"
          : synthetic
          ? "该策略为合成会计验证，未使用旧指纹策略套用硬件测量。"
          : "策略与硬件 / 模型身份已绑定。"
      }</div><details><summary class="muted">每 rank KV / 权重驻留</summary><div class="scroll-table"><table class="hop-table"><tr><th>rank</th><th>层范围</th><th>权重</th><th>KV block</th></tr>${(
        c.rank_geometry || []
      )
        .map(
          (r) =>
            `<tr><td>${r.rank}</td><td>${r.layer_start}–${
              r.layer_end
            }</td><td>${bytes(r.weight_resident_bytes)}</td><td>${bytes(
              r.kv_block_bytes
            )}</td></tr>`
        )
        .join("")}</table></div></details></div>`;
    })
    .join("");
}
const checksText = {
  unified_clock: ['统一时钟与事件顺序'],
  shared_capacity: ['共享链路与容量池'],
  byte_conservation: ['逐跳字节守恒'],
  delivery_clock: [
    "通信完成后通知算子",
    "通信完成与依赖释放的时刻一致",
  ],
  service_capacity: [
    "逐段传输的带宽检查",
    "逐个带宽服务区间核对分配速率与容量、字节量；区间数不是请求数",
  ],
  transfer_identity: ["传输终态唯一", "每个 transfer ID 只有一个终态"],
  dependency_order: ["算子因果与依赖", "前驱完成与 launch gap 均已满足"],
  kernel_duration: ["计算 / 存储调用耗时", "每次执行与 AsterMind 模型预测的 ns 取整值对照；不是硬件实测"],
  communication_accounting: ["通信不重复计费", "原生 COMM 参考值实际计费为零"],
  communication_barrier: ["通信依赖与字节量", "逐流通信等待最慢交付；ProtoBridge 通信等待估算完成事件"],
  input_distribution: ["输入先于批次就绪", "token ID 分发完成后启动算子图"],
  request_delivery: ["请求与 token 交付", "计算、输出交付与 token 数量一致"],
  ground_forwarding: ["地面用户不转发 GPU 流量", "实际路径中没有地面用户中继"],
  migration_transaction: ["KV 事务完整性", "全部分片、写入与回滚的终态证据"],
};
function renderAudit() {
  const d = state.data,
    failures = d.audit.filter((c) => c.state === "fail"),
    warnings = d.audit.reduce((sum,c)=>sum+(c.warnings?.length||0),0);
  $("auditBadge").className = `badge ${failures.length ? "fail" : warnings ? "warn" : "pass"}`;
  $("auditBadge").textContent = failures.length
    ? `${failures.length} 类约束有违规`
    : warnings ? `${warnings.toLocaleString()} 条精度提示` : "记录一致性检查通过";
  $("checks").innerHTML =
    d.audit
      .map((c) => {
        const [title] = checksText[c.id] || [c.id];
        return `<div class="check ${
          c.state === "fail" ? "fail" : c.state === 'warn' ? 'warn' : ""
        }"><strong>${c.state === "pass" ? "✓" : c.state === 'warn' ? '△' : "✕"} ${esc(
          title
        )}</strong><span>${c.checked.toLocaleString()} 项核对</span>${
          c.violations.length
            ? `<details open><summary>${
                c.violations.length
              } 条违规证据</summary>${c.violations
                .slice(0, 8)
                .map((v) => `<p>${esc(v.evidence)}<br>${esc(v.detail)}</p>`)
                .join("")}</details>`
            : ""
        }${c.warnings?.length?`<details><summary>${c.warnings.length.toLocaleString()} 条时钟精度提示</summary>${c.warnings.slice(0,8).map(v=>`<p>${esc(v.evidence)}<br>${esc(v.detail)}</p>`).join('')}</details>`:''}</div>`;
      })
      .join("");
  $("sourceDetails").innerHTML = `<p>源报告：<code>${esc(
    d.source.name
  )}</code><br>SHA-256：<code>${esc(
    d.source.sha256
  )}</code><br>物理数据身份：<code>${esc(
    d.physical_identity || "静态显式资源"
  )}</code></p><p>时间语义：${esc(
    d.clock_note
  )}。图上容器为 owner 连接示意，不是轨道位置缩放图。逐跳表的字节是逻辑服务量，ns-3 PPP 线速计数单独记录。</p><p>本页 ${
    d.audit.length
  } 类核查由源记录重新计算；原始全成功审计：${
    d.original_audit?.passed
      ? "通过"
      : d.original_audit
      ? "未通过 / 故障场景不满足全成功条件"
      : "未附带"
  }。记录状态：${esc(d.status)}。</p><p>Scheduler：${Object.entries(d.cells)
    .map(
      ([id, c]) =>
        `${esc(id)} / ${esc(c.scheduler.version)} / Python manifest <code>${esc(
          c.scheduler.package_python_manifest_sha256
        )}</code>`
    )
    .join("<br>")}</p>`;
}
function updateCallOptions() {
  const query = $("search").value.toLowerCase();
  const rows = state.data.calls
    .map((c, i) => ({ c, i }))
    .filter(
      ({ c }) =>
        filterPass(c) &&
        (!query || `${c.id} ${c.kind} ${c.batch}`.toLowerCase().includes(query))
    );
  const shown = rows.slice(0, 150);
  if (state.selected >= 0 && !shown.some((r) => r.i === state.selected))
    shown.unshift({ c: state.data.calls[state.selected], i: state.selected });
  $("callSelect").innerHTML =
    shown
      .map(
        ({ c, i }) =>
          `<option value="${i}">${esc(
            c.device +
              " · " +
              c.id +
              (c.occurrence >= 0 ? " / #" + c.occurrence : "")
          )}</option>`
      )
      .join("") +
    (rows.length > 150
      ? `<option disabled>显示前 150 项 / 共 ${rows.length} 匹配；输入关键词定位</option>`
      : "");
  $("callSelect").value = String(state.selected);
}
function renderClock() {
  renderNow();
  renderTopology();
  drawTimeline();
  renderMigrations();
  for (const [i, el] of [...$("steps").querySelectorAll(".step-row")].entries())
    el.classList.toggle("live", active(state.data.steps[i]));
}
async function loadCase(id) {
  const epoch = ++state.epoch;
  state.playing = false;
  $("play").textContent = "▶ 播放";
  $("error").hidden = true;
  $("caseSubtitle").textContent = "读取并核对真实执行记录…";
  try {
    const item = catalog.find((c) => c.id === id) || catalog[0];
    const d = await get(item.file);
    if (epoch !== state.epoch) return;
    if (d.schema !== "orbitserve.native_replay.v1")
      throw Error("不支持的回放格式");
    state.data = d;
    state.limit = Math.min(d.horizon, Math.max(1e6, d.end * 1.06));
    state.view = [0, state.limit];
    transfers = new Map(d.transfers.map((t) => [t.id, t]));
    flowHops = new Map();
    for (const h of d.hops) {
      if (!flowHops.has(h.flow)) flowHops.set(h.flow, []);
      flowHops.get(h.flow).push(h);
    }
    resources = new Map(d.resources.map((r) => [r.id, r]));
    successors = d.calls.map(() => []);
    d.calls.forEach((c, i) =>
      c.deps.forEach((dep) => {
        if (dep >= 0) successors[dep].push(i);
      })
    );
    $("caseSelect").value = item.id;
    if($("batchSelect"))$("batchSelect").value=item.id;
    $("caseSubtitle").textContent = `${d.title} · ${communicationLabel(d)}`;
    $("cellFilter").innerHTML =
      '<option value="">全部 Cell</option>' +
      Object.keys(d.cells)
        .map((c) => `<option>${esc(c)}</option>`)
        .join("");
    $("deviceFilter").innerHTML =
      '<option value="">全部 GPU</option>' +
      d.devices
        .filter((n) => n.kind === "gpu")
        .map((n) => `<option>${esc(n.id)}</option>`)
        .join("");
    $("search").value = "";
    $("kindFilter").value = "";
    $("time").max = state.limit;
    $("timeInput").max = state.limit / 1e6;
    renderMetrics();
    renderMoments();
    renderMemory();
    renderAudit();
    state.selected = -1;
    updateCallOptions();
    state.step = 0;
    const longest = d.calls.reduce(
      (best, c, i) =>
        c.comm &&
        c.end - c.start > (d.calls[best]?.end - d.calls[best]?.start || 0)
          ? i
          : best,
      -1
    );
    state.selected = longest >= 0 ? longest : 0;
    selectCall(state.selected, { seek: false });
    const fault = d.events.find((e) => e.kind === "fault_injected");
    const t = fault
      ? fault.time + Math.min(1e8, (d.end - fault.time) / 3)
      : Math.min(d.end * 0.4, state.data.calls[state.selected]?.start || 0);
    setTime(t);
    if(d.page_scope?.kind==='batch'){
      setView(d.page_scope.start, d.page_scope.end);
      setTime(d.page_scope.start);
      $("caseSubtitle").textContent += ' · 当前批次算子与通信；校验与显存快照属于完整实验';
    }
    const caseUrl = new URL(location.href);
    caseUrl.searchParams.set("case", item.id);
    history.replaceState(null, "", caseUrl);
  } catch (error) {
    fail(error);
  }
}
$("caseSelect").addEventListener("change", () =>
  loadCase($("caseSelect").value)
);
$("reload").onclick = () => loadCase($("caseSelect").value);
$("time").oninput = () => setTime($("time").value);
$("timeInput").onchange = () => setTime(Number($("timeInput").value) * 1e6);
$("play").onclick = () => {
  if (!state.data) return;
  if (state.now >= state.limit) setTime(0);
  state.playing = !state.playing;
  $("play").textContent = state.playing ? "Ⅱ 暂停" : "▶ 播放";
  lastFrame = 0;
};
$("previous").onclick = () => {
  const row = state.data?.events.filter((e) => e.time < state.now - 1).at(-1);
  jump(row?.time ?? 0);
};
$("next").onclick = () => {
  const row = state.data?.events.find(
    (e) => e.time > state.now + 1 && e.time <= state.limit
  );
  jump(row?.time ?? state.limit);
};
$("moments").onclick = (event) => {
  const el = event.target.closest("[data-time]");
  if (el) jump(Number(el.dataset.time));
};
$("allTime").onclick = () => setView(0, state.limit);
$("focusTime").onclick = () => {
  const w = (state.view[1] - state.view[0]) * 0.35;
  setView(state.now - w / 2, state.now + w / 2);
};
for (const id of ["cellFilter", "deviceFilter", "kindFilter"])
  $(id).onchange = () => {
    updateCallOptions();
    drawTimeline();
  };
$("search").oninput = updateCallOptions;
$("callSelect").onchange = () =>
  selectCall(Number($("callSelect").value), { fit: true });
$("dependencies").onclick = (e) => {
  const el = e.target.closest("[data-call]");
  if (el) selectCall(Number(el.dataset.call), { fit: true });
};
$("steps").onclick = (e) => {
  const el = e.target.closest("[data-step]");
  if (el) selectStep(Number(el.dataset.step));
};
for (const id of ["flowDetails", "migrations"]) {
  $(id).addEventListener("click", (e) => {
    const el = e.target.closest("[data-flow]");
    if (el) {
      selectFlow(el.dataset.flow);
      if (id === "migrations")
        $("callDetails").scrollIntoView({ block: "start", behavior: "smooth" });
    }
  });
  $(id).addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const el = e.target.closest("[data-flow]");
      if (el) selectFlow(el.dataset.flow);
    }
  });
}
function inspectTopology(e) {
  const device = e.target.closest("[data-device]"),
    link = e.target.closest("[data-link]");
  if (device) {
    const n = state.data.devices.find((d) => d.id === device.dataset.device);
    $("deviceDetail").textContent = `${n.id} · ${n.satellite} · ${n.kind}${
      n.capacity_gbps != null
        ? " · 共享门容量 " + n.capacity_gbps + " Gb/s"
        : ""
    }${n.kind === "gpu" ? " · GPU 输入/输出端口分离，不能用于转发" : ""}`;
    if (n.kind === "gpu") {
      $("deviceFilter").value = n.id;
      updateCallOptions();
      drawTimeline();
    }
  } else if (link) {
    const row = state.data.links.find((d) => d.id === link.dataset.link);
    const matching = state.data.hops.filter(
      (h) =>
        [h.source, h.target].sort().join("|") ===
        [row.source, row.target].sort().join("|")
    );
    const current = matching.filter((h) => active(h));
    const last =
      current[0] || matching.filter((h) => h.end <= state.now).at(-1);
    $("deviceDetail").textContent = `${row.id} · ${row.source} ↔ ${
      row.target
    } · 配置端口上限 ${row.bandwidth_gbps} Gb/s。${
      last
        ? `${
            current.length
              ? `此刻 ${current.length} 条流；所示单流`
              : "最近一次单流"
          }服务：${rate(last.rate)} / 物理容量 ${rate(
            last.capacity
          )}，区间 ${ms(last.start)} → ${ms(last.end)}。`
        : "此刻之前没有可展示的实际服务区间。"
    }`;
  }
}
$("topology").addEventListener("click", inspectTopology);
$("topology").addEventListener("keydown", (e) => {
  if (e.key === "Enter") inspectTopology(e);
});
$("timeline").addEventListener("click", (event) => {
  const rect = $("timeline").getBoundingClientRect(),
    x = event.clientX - rect.left,
    y = event.clientY - rect.top;
  const hit = state.hits
    .filter(
      (h) => x >= h.x - 2 && x <= h.x + h.w + 2 && y >= h.y && y <= h.y + h.h
    )
    .at(-1);
  if (hit) selectCall(hit.index, { seek: true });
  else {
    const t =
      state.view[0] +
      ((x - 124) / (rect.width - 139)) * (state.view[1] - state.view[0]);
    jump(t);
  }
});
$("timeline").addEventListener("mousemove", (event) => {
  const rect = $("timeline").getBoundingClientRect(),
    x = event.clientX - rect.left,
    y = event.clientY - rect.top;
  const hit = state.hits
      .filter(
        (h) => x >= h.x - 2 && x <= h.x + h.w + 2 && y >= h.y && y <= h.y + h.h
      )
      .at(-1),
    tip = $("tooltip");
  tip.hidden = !hit;
  if (hit) {
    const c = state.data.calls[hit.index];
    tip.textContent = `${c.id}\n${c.device} / rank ${c.rank} · ${c.kind}\n${ms(
      c.start
    )} → ${ms(c.end)} · ${duration(c.end - c.start)}`;
    tip.style.left = Math.max(0, Math.min(x + 12, rect.width - 380)) + "px";
    tip.style.top = Math.max(0, y - 70) + "px";
  }
});
$("timeline").onmouseleave = () => {
  $("tooltip").hidden = true;
};
$("timeline").addEventListener(
  "wheel",
  (event) => {
    if (!state.data) return;
    event.preventDefault();
    const rect = $("timeline").getBoundingClientRect(),
      fraction = Math.max(
        0,
        Math.min(1, (event.clientX - rect.left - 124) / (rect.width - 139))
      ),
      [a, b] = state.view,
      center = a + fraction * (b - a),
      width = (b - a) * (event.deltaY > 0 ? 1.45 : 0.69);
    setView(center - fraction * width, center + (1 - fraction) * width);
  },
  { passive: false }
);
new ResizeObserver(() => drawTimeline()).observe($("timeline").parentElement);
function tick(now) {
  if (state.playing && state.data) {
    if (!lastFrame) lastFrame = now;
    const delta = now - lastFrame;
    lastFrame = now;
    state.now = Math.min(
      state.limit,
      state.now + delta * 1e6 * Number($("speed").value)
    );
    if (now - lastPaint > 65) {
      lastPaint = now;
      setTime(state.now);
    }
    if (state.now >= state.limit) {
      setTime(state.limit);
      state.playing = false;
      $("play").textContent = "▶ 播放";
    }
  }
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
const replayLoading=document.createElement('p');
replayLoading.setAttribute('role','status');replayLoading.textContent='正在读取计算批次…';
$("metrics").before(replayLoading);
try {
  let catalogReady=false;
  for(let attempt=0;attempt<120;attempt++){
    const response=await fetch('catalog.json'),value=await response.json();
    if(response.ok){catalog=value;catalogReady=true;break}
    if(!String(value.error||'').includes('正在整理'))throw Error(value.error||'回放索引读取失败');
    $("caseSubtitle").textContent='正在整理计算批次索引…';
    replayLoading.textContent='首次打开正在整理计算批次索引，完成后自动显示。';
    await new Promise(resolve=>setTimeout(resolve,3000));
  }
  if(!catalogReady)throw Error('回放索引仍在整理，请稍后重新加载');
  if (!catalog.length) throw Error("没有已导出的原生执行记录");
  $("caseSelect").innerHTML = catalog
    .map((c) => `<option value="${esc(c.id)}">${esc(c.title)}</option>`)
    .join("");
  if(catalog.some(c=>c.paged)){
    const label=document.createElement('label'),select=document.createElement('select');
    label.className='paged-batches';label.textContent='计算批次 ';select.id='batchSelect';
    select.setAttribute('aria-label','计算批次');select.innerHTML=$("caseSelect").innerHTML;
    select.onchange=()=>loadCase(select.value);label.append(select);$("metrics").before(label);
  }
  const initial =
    new URLSearchParams(location.search).get("case") ||
    new URLSearchParams(location.hash.slice(1)).get("case");
  await loadCase(initial || catalog[0].id);
} catch (error) {
  fail(error);
} finally {
  replayLoading.remove();
}
