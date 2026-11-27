import {modelDisplayName} from '../cell/config-labels.mjs';

const commonLabels = {
  hardware:'算力资源', constellation:'星座', topology:'拓扑', deployment:'模型部署', scheduler:'调度器',
  kv_offload:'KV 缓存', runtime_inputs:'运行参数', compute_mode:'计算模式', horizon_ns:'仿真时长',
  schema_version:'格式版本', runtime_network:'故障设置', additional_cell_links:'附加链路',
  disabled_physical_links:'停用链路', max_requests:'并发请求上限', max_model_len:'上下文上限',
  arrival_ns:'到达时间', retry_limit:'最多重试次数', primary:'首先比较', secondary:'条件相同时再比较',
};
const displayValues={detail_mode:'Detail Mode',fast_mode:'Fast Mode',least_loaded:'负载优先',nearest:'距离优先',estimated_arrival:'先比较负载，再比较到达时间',midlayer:'原生计算模型',load:'请求最少',distance:'路径最短',arrival:'最早到达'};

function node(tag, content, className) {
  const result=document.createElement(tag);
  if(content!=null) result.textContent=content;
  if(className) result.className=className;
  return result;
}

export function renderCellConfigProfile(container, config, labels) {
  const {model,compute_mode,hardware,topology,constellation,fso,deployment,scheduler}=config;
  const groups={model,compute_mode,hardware,topology,constellation,fso,deployment,scheduler,
    collective_policies:config.collective_policies,requests:config.requests};
  if(compute_mode==='fast_mode')groups.fast_roofline=config.fast_roofline;
  renderConfigProfile(container,groups,{...labels,fso:'光学链路',fast_roofline:'Roofline 参数配置'},[],true);
}

export function renderConfigProfile(container, config, labels={}, metrics=[], omitEmpty=false) {
  const previous=container.querySelector('select')?.value;
  container.replaceChildren();
  if(!config)return;
  const caption=key=>labels[key] || commonLabels[key] || key;
  const card=node('article',null,'hardware-card config-card');
  if(metrics.length) {
    const list=node('dl',null,'hardware-metrics');
    for(const [label,value] of metrics){
      const row=node('div');row.append(node('dt',label),node('dd',value));list.append(row);
    }
    card.append(list);
  }
  const scalar=value=>value===null || typeof value!=='object';
  const keys=Object.keys(config).filter(key=>config[key]!=null && !scalar(config[key]) && Object.keys(config[key]).length);
  const base=Object.fromEntries(Object.entries(config).filter(([,value])=>scalar(value) && value!=null));
  const select=node('select');select.setAttribute('aria-label','查看配置分组');
  if(Object.keys(base).length) select.add(new Option('基本参数',''));
  for(const key of keys)select.add(new Option(caption(key),key));
  if([...select.options].some(o=>o.value===previous))select.value=previous;
  const list=node('dl',null,'hardware-specs');
  const pages=node('div',null,'actions');
  function *rows(value, path=[]) {
    for(const [key,item] of Object.entries(value)){
      if(omitEmpty && (item==null || item==='' || (typeof item==='object'&&!Object.keys(item).length)))continue;
      const label=Array.isArray(value)?'第 '+(Number(key)+1)+' 项':caption(key);
      const name=[...path,typeof item==='number' && key.endsWith('_ns') ? label.replace(/(?:_ns|\s*\/\s*ns)$/i,'') : label];
      if(!scalar(item) && Object.keys(item).length){yield* rows(item,name);continue}
      let shown=item;
      if(item==null) shown='未设置';
      else if(typeof item==='object') shown='无';
      else if(typeof item==='boolean')shown=item?'开启':'关闭';
      else if(typeof item==='number' && key.endsWith('_ns'))shown=(item/1e9).toLocaleString('zh-CN',{maximumFractionDigits:9})+' 秒';
      else if(typeof item==='string' && key==='model')shown=modelDisplayName(item);
      else if(typeof item==='string' && ['cell_selection','compute_mode','primary','secondary'].includes(key))shown=displayValues[item]||item;
      yield [name.join(' / '),shown];
    }
  }
  let page=0;
  function show(){
    const source=select.value?config[select.value]:base, iterator=rows(source);
    let item=iterator.next();
    for(let i=0;i<page*24&&!item.done;i++)item=iterator.next();
    list.replaceChildren();
    for(let i=0;i<24&&!item.done;i++){
      const [key,value]=item.value,row=node('div');row.append(node('dt',key),node('dd',value));list.append(row);item=iterator.next();
    }
    pages.replaceChildren();
    if(page || !item.done) {
      for(const [label,step,disabled] of [['上一页',-1,!page],['下一页',1,item.done]]){
        const button=node('button',label);button.type='button';button.disabled=disabled;
        button.onclick=()=>{page+=step;show()};pages.append(button);
      }
    }
  }
  select.onchange=()=>{page=0;show()};card.append(select,list,pages);container.append(card);show();
}
