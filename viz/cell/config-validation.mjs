import { fieldStage } from './config-layout.mjs?v=compact-2';
import {linkMode} from './link-mode.mjs';
import {resourceSyncError} from './resource-sync.mjs';

const resolve = (spec, defs) => spec?.$ref ? defs[spec.$ref.split('/').at(-1)] || {} : spec || {};
export function schemaAt(schema, path) {
  let spec = schema;
  for (const key of path) {
    spec = resolve(spec, schema.$defs);
    if (spec.anyOf) spec = resolve(spec.anyOf.find(s => s.type !== 'null'), schema.$defs);
    spec = typeof key === 'number' ? spec.prefixItems?.[key] || spec.items : spec.properties?.[key] || spec.additionalProperties;
  }
  return resolve(spec, schema.$defs);
}

export function valueError(value, raw, defs = {}) {
  const spec = resolve(raw, defs);
  if (spec.anyOf) return spec.anyOf.some(s => !valueError(value, s, defs)) ? '' : valueError(value, spec.anyOf.find(s => s.type !== 'null'), defs) || '值类型不正确';
  if (spec.const !== undefined && value !== spec.const) return `必须为 ${spec.const}`;
  if (spec.enum && !spec.enum.includes(value)) return '请选择有效选项';
  if (spec.type === 'null') return value === null ? '' : '必须为空';
  if (value == null) return '请填写此项';
  if (['integer', 'number'].includes(spec.type)) {
    if (typeof value !== 'number' || !Number.isFinite(value)) return '请输入有限数值';
    if (spec.type === 'integer' && !Number.isSafeInteger(value)) return '请输入有效整数';
    if (spec.minimum !== undefined && value < spec.minimum) return `不能小于 ${spec.minimum}`;
    if (spec.maximum !== undefined && value > spec.maximum) return `不能大于 ${spec.maximum}`;
    if (spec.exclusiveMinimum !== undefined && value <= spec.exclusiveMinimum) return `必须大于 ${spec.exclusiveMinimum}`;
    if (spec.exclusiveMaximum !== undefined && value >= spec.exclusiveMaximum) return `必须小于 ${spec.exclusiveMaximum}`;
    if (spec.multipleOf && Math.abs(value/spec.multipleOf - Math.round(value/spec.multipleOf)) > 1e-9) return `必须是 ${spec.multipleOf} 的倍数`;
  }
  if (spec.type === 'string') {
    if (typeof value !== 'string') return '请输入文本';
    if (spec.minLength && value.trim().length < spec.minLength) return `至少填写 ${spec.minLength} 个字符`;
    if (spec.maxLength && [...value].length > spec.maxLength) return `最多 ${spec.maxLength} 个字符`;
    if (spec.pattern && !new RegExp(spec.pattern, 'u').test(value)) return '格式不正确';
    if (spec.format === 'date-time') {
      const match = value.match(/^(\d{4})-(\d{2})-(\d{2})[Tt](\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:[Zz]|[+-](\d{2}):(\d{2}))$/);
      if (!match || !Number.isFinite(Date.parse(value))) return '请输入含时区的时间，如 2026-08-03T12:00:00Z';
      const [,y,m,d,h,min,s,oh='0',om='0'] = match.map(Number);
      if (m<1 || m>12 || d<1 || d>new Date(Date.UTC(y,m,0)).getUTCDate() || h>23 || min>59 || s>59 || oh>23 || om>59) return '日期或时间无效';
    }
  }
  if (spec.type === 'boolean' && typeof value !== 'boolean') return '请选择开关状态';
  if (spec.type === 'array') {
    if (!Array.isArray(value)) return '请输入列表';
    if (spec.minItems !== undefined && value.length < spec.minItems) return `至少需要 ${spec.minItems} 项`;
    if (spec.maxItems !== undefined && value.length > spec.maxItems) return `最多允许 ${spec.maxItems} 项`;
    if (spec.uniqueItems && new Set(value.map(v=>JSON.stringify(v))).size !== value.length) return '列表不能重复';
    for (const [i,item] of value.entries()) {
      const error = valueError(item, spec.prefixItems?.[i] || spec.items || {}, defs);
      if (error) return `第 ${i+1} 项：${error}`;
    }
  }
  if (spec.type === 'object') {
    if (!value || typeof value !== 'object' || Array.isArray(value)) return '必须是配置对象';
    for (const key of spec.required || []) if (value[key] === undefined) return `缺少 ${key}`;
    for (const [key,child] of Object.entries(value)) {
      const childSpec = spec.properties?.[key] || spec.additionalProperties;
      if (!childSpec && spec.additionalProperties === false) return `未知字段 ${key}`;
      if (typeof childSpec === 'object') { const error=valueError(child,childSpec,defs); if(error) return `${key}：${error}`; }
    }
  }
  return '';
}

export function fieldError(value, path, root, spec = {}, defs = {}) {
  const error = valueError(value, spec, defs); if (error) return error;
  const name=path.join('.'), key=path.at(-1), hardware=root.hardware, ids=hardware?.satellite_ids || [];
  // Dataclass-backed inputs have semantic bounds that JSON Schema does not
  // expose; mirror their native validators here, without changing engine rules.
  if(path[0]==='requests' && path[1]==='synthetic') {
    if(['num_requests','request_rate'].includes(key) && value<=0) return '必须大于 0';
    if(['input_lengths','output_lengths'].includes(key) && (!value.length || value.some(v=>!Number.isSafeInteger(v) || v<(key==='input_lengths'?1:0)))) return key==='input_lengths' ? '请输入正整数 Token 长度列表' : '请输入非负整数 Token 长度列表';
    if(key==='prefix_hit_fraction' && (value<0 || value>1)) return '比例范围为 0–1';
    if(key==='arrival_distribution' && !['poisson','constant'].includes(value)) return '请选择 poisson 或 constant';
  }
  if(path.includes('parallelism') && key.endsWith('_size') && value<1) return '并行规模必须为正整数';
  if(path[0]==='scheduler' && path.includes('vllm')) {
    if(key==='gpu_memory_utilization' && (value<=0 || value>1)) return '显存利用比例必须大于 0 且不超过 1';
    if(key==='max_model_len' && value!==null && value<1) return '模型长度必须为正整数';
    if(['dummy_token_id','mtp_num_speculative_tokens','mtp_fixed_accepted_tokens'].includes(key) && value<0) return '不能小于 0';
  }
  if(path.includes('kv_cache')) {
    if(['bytes_per_token','block_size_tokens'].includes(key) && value<1) return '必须为正整数';
    if(key==='capacity_bytes' && value<0) return '容量不能小于 0';
  }
  if(path.includes('pipeline_p2p') && ['source_stage','destination_stage'].includes(key) && value<0) return 'PP stage 不能小于 0';
  if(name==='deployment.router.dispatch_service_time_s' && value<0) return '调度时间不能小于 0';
  if(path[0]==='scheduler' && path.includes('sglang')) {
    if(['max_total_num_tokens','chunked_prefill_size','max_prefill_tokens','max_running_requests','schedule_conservativeness'].includes(key) && value<=0) return '必须大于 0';
    if(key==='mtp_num_speculative_tokens' && value<0) return '不能小于 0';
  }
  if(path.includes('mtp_acceptance_assumption')) {
    if(key==='proposal_depth' && value<1) return '推测深度必须为正整数';
    if(key==='root_seed' && value<0) return '随机种子不能小于 0';
    if(key==='mean_acceptance_rate' && (!String(value).trim() || !Number.isFinite(Number(value)) || Number(value)<0 || Number(value)>1)) return '接受比例范围为 0–1';
  }
  if(path.includes('graph_execution')) {
    if(key==='uniform_decode_query_len' && value<1) return 'Query 长度必须为正整数';
    if(key==='capture_sizes' && (value.some(v=>!Number.isSafeInteger(v) || v<1) || new Set(value).size!==value.length)) return 'Capture sizes 必须是无重复的正整数列表';
  }
  if(path[0]==='fso' && path.includes('modulation_pool') && path.length===5) {
    if(key===0 && (value<4 || !Number.isInteger(Math.log2(value)))) return '调制阶数必须为不小于 4 的 2 次幂';
    if(key===1 && !value.trim()) return '调制名称不能为空';
  }
  if (name === 'hardware.satellite_count' && root.topology?.source === 'formation' && (value<2 || value>35)) return '静态编队需要 2–35 颗卫星';
  if (name === 'hardware.satellite_ids') {
    if (!value.length || value.some(id=>typeof id!=='string' || !id.trim() || id!==id.trim())) return '卫星 ID 不能为空或包含首尾空格';
    if (new Set(value).size!==value.length) return '卫星 ID 不能重复';
    if (root.topology?.source==='formation' && (value.length<2 || value.length>35)) return '静态编队需要 2–35 个卫星 ID';
  }
  if (key==='local_index' || key==='gateway_local_index') {
    if (value >= hardware?.accelerators_per_satellite) return `卡号范围为 0–${hardware.accelerators_per_satellite-1}`;
  }
  if (key==='satellite_id' && !ids.includes(value)) return '请选择资源清单中的卫星';
  if (path[1]==='worker_pools' && key==='name') {
    if (!value.trim() || value!==value.trim() || ['__proto__','constructor','prototype'].includes(value)) return '名称不能为空或包含首尾空格';
    if (root.deployment.worker_pools.some((p,i)=>i!==path[2] && p.name===value)) return 'Pool 名称不能重复';
  }
  if (path.includes('links') && ['source','target'].includes(key)) {
    const link=path.slice(0,-1).reduce((v,k)=>v?.[k],root);
    if (!ids.includes(value)) return '请选择资源清单中的卫星';
    if (value===link?.[key==='source'?'target':'source']) return '链路两端不能是同一颗卫星';
  }
  if (name==='topology.inter_satellite.links') {
    const pairs=value.map(link=>JSON.stringify([link.source,link.target].sort()));
    if (new Set(pairs).size!==pairs.length) return '不能重复连接同一对卫星';
  }
  if (name==='fso.tx.baud_rate_options_ghz' && new Set(value).size!==value.length) return '波特率不能重复';
  if (path.includes('parallel_coordinate')) {
    const row=root.deployment.placements[path[2]], pool=root.deployment.worker_pools.find(p=>p.name===row.worker_pool);
    const size=pool?.parallelism?.[{pp:'pipeline_parallel_size',tp:'tensor_parallel_size',cp:'context_parallel_size',dp:'data_parallel_size'}[key]];
    if (size !== undefined && value>=size) return `此坐标范围为 0–${size-1}`;
  }
  return '';
}

// Validation of loaded JSON and committed values uses the same backend schema
// as individual controls. Cross-step completeness is checked at submission.
export function configurationErrors(root, schema, stage = null) {
  const issues=[];
  const add=(path,msg)=>{if(msg && (!stage || fieldStage(path)===stage)) issues.push({loc:path,msg});};
  function walk(value,raw,path) {
    let spec=resolve(raw,schema.$defs);
    if(spec.anyOf) { if(value===null && spec.anyOf.some(s=>s.type==='null')) return; spec=resolve(spec.anyOf.find(s=>s.type!== 'null'),schema.$defs); }
    if(spec.type==='object' && value && typeof value==='object' && !Array.isArray(value)) {
      for(const key of spec.required || []) if(value[key]===undefined) add([...path,key],'请填写此项');
      for(const [key,child] of Object.entries(value)) {
        const childSpec=spec.properties?.[key] || spec.additionalProperties;
        if(typeof childSpec==='object') walk(child,childSpec,[...path,key]);
        else if(spec.additionalProperties===false) add([...path,key],'未知字段');
      }
    } else if(spec.type==='array' && Array.isArray(value)) {
      const shallow={...spec,items:{},prefixItems:undefined}; add(path,fieldError(value,path,root,shallow,schema.$defs));
      value.forEach((v,i)=>walk(v,spec.prefixItems?.[i] || spec.items || {},[...path,i]));
    } else add(path,fieldError(value,path,root,spec,schema.$defs));
  }
  walk(root,schema,[]);
  if (root.hardware && root.deployment && resourceSyncError(root)) add(['hardware','satellite_count'],resourceSyncError(root));
  if(root.hardware && root.hardware.satellite_count!==root.hardware.satellite_ids?.length) add(['hardware','satellite_ids'],'ID 数量必须与卫星数量一致');
  if(root.topology?.source==='orbit' && !root.constellation?.tle_file) add(['constellation','tle_file'],'请先导入 TLE 文件');
  if(root.topology?.source==='formation' && !root.constellation?.formation) add(['constellation','formation'],'请设置静态编队');
  if(root.requests?.source==='synthetic' && !root.requests.synthetic) add(['requests','synthetic'],'请设置合成请求');
  if(root.requests?.source==='production' && !root.requests.production) add(['requests','production'],'请设置生产 Trace');
  if(root.requests?.source!=='synthetic' && root.requests && !root.requests.file) add(['requests','file'],'请先导入请求流文件');
  for(const [i,link] of (root.topology?.inter_satellite?.links || []).entries()) {
    if(link.mode==='fixed' && link.bandwidth_gbps==null) add(['topology','inter_satellite','links',i,'bandwidth_gbps'],'默认模式必须填写链路带宽');
  }
  if (root.topology?.inter_satellite && linkMode(root) === null) add(['topology','inter_satellite'],'请在星间连接中选择一个全局模式，统一所有链路');
  if(root.deployment?.placements && root.hardware) {
    const rows=root.deployment.placements, pools=root.deployment.worker_pools || [];
    const expected=pools.reduce((n,p)=>n+p.parallelism.tensor_parallel_size*p.parallelism.pipeline_parallel_size,0);
    const physical=root.hardware.satellite_count*root.hardware.accelerators_per_satellite;
    if(expected>physical) add(['deployment','worker_pools'],`当前只有 ${physical} 张物理卡，并行部署需要 ${expected} 张卡。请启用自动适配资源，或减少 TP / PP。`);
    if(rows.length!==expected) add(['deployment','placements'],'部署映射数量与并行规模不一致');
    const locations=new Set(), ranks=new Set();
    for(const [i,row] of rows.entries()) {
      const location=JSON.stringify([row.satellite_id,row.local_index]);
      const rank=JSON.stringify([row.worker_pool,row.parallel_coordinate]);
      if(locations.has(location)) add(['deployment','placements',i,'local_index'],'同一张算力卡不能重复分配');
      if(ranks.has(rank)) add(['deployment','placements',i,'parallel_coordinate'],'并行坐标不能重复');
      if(!pools.some(p=>p.name===row.worker_pool)) add(['deployment','placements',i,'worker_pool'],'请选择已有 Worker pool');
      locations.add(location);ranks.add(rank);
    }
  }
  return issues;
}

export function validationError(issues) {
  if (!issues.length) return null;
  const error=new Error(issues[0].msg); error.detail=issues; return error;
}
