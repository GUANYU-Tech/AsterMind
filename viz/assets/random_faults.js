import {field} from './studio_common.js';

function normalizeGroup(settings,time,seed){
 const config=settings||{},step=Number(time.step_s)||1,horizon=Number(time.duration_s)||120;
 if(config.time_basis==='seconds'||!config.model)return {enabled:false,time_basis:'seconds',model:'exponential',seed,ratio:.01,
  start_s:0,end_s:horizon,mean_up_s:seed===43?2592000:86400,up_scale_s:86400,up_shape:1.5,...config,
  duration:{distribution:'fixed',seconds:60,min_s:30,max_s:120,mean_s:60,sigma:.6,...config.duration}};
 const {start_s,end_s,repair_probability,duration_steps,target_type,...saved}=config;
 const last=Math.round(horizon/step),end=Math.min(4,Math.max(1,last-1));
 return {enabled:false,model:'per_step',ratio:.01,seed,failure_probability:.01,mean_up_steps:20,mean_down_steps:3,up_scale_steps:20,up_shape:1.5,
  start_step:start_s==null?Math.min(2,end-1):start_s/step,end_step:end_s==null?end:end_s/step,...saved,
  duration:{distribution:'fixed',steps:duration_steps??2,min_steps:1,max_steps:4,mean_steps:3,sigma:.6,...saved.duration}};
}
export function normalizeRandomFaults(settings={},time={step_s:5,duration_s:120}){
 const grouped='nodes' in settings||'links' in settings;
 return {nodes:normalizeGroup(grouped?settings.nodes:settings.target_type==='node'?settings:{},time,43),
  links:normalizeGroup(grouped?settings.links:settings.target_type!=='node'?settings:{},time,42)};
}

export function renderRandomFaults(host,settings,time){
 host.replaceChildren();const refresh=[];
 for(const[key,title]of [['nodes','节点故障'],['links','链路故障']]){
  const config=settings[key],seconds=config.time_basis==='seconds',unit=seconds?'秒':'时间片',card=document.createElement('fieldset'),legend=document.createElement('legend'),grid=document.createElement('div');
  const target=key==='nodes'?'卫星':'链路';
  card.className='fault-card';legend.textContent=key==='nodes'?'卫星节点故障':'通信链路故障';grid.className='fields';
  const advanced=document.createElement('details'),caption=document.createElement('summary');
  caption.textContent='高级设置';
  const advancedGrid=document.createElement('div'),timeWindow=document.createElement('div');
  advancedGrid.className='fields';timeWindow.className='fields fault-window';
  advanced.append(caption,advancedGrid);
  card.append(legend,grid,advanced);host.append(card);
  if(!seconds){const convert=document.createElement('button');convert.type='button';convert.textContent='改为按秒设置';advanced.append(convert);convert.onclick=()=>{
   const step=Number(time.step_s)||1,p=config.failure_probability,up=config.model==='per_step'?(p>0&&p<1?-step/Math.log1p(-p):step):config.mean_up_steps*step;
   settings[key]=normalizeGroup({enabled:config.enabled,time_basis:'seconds',model:config.model==='per_step'?'exponential':config.model,seed:config.seed,ratio:config.ratio,start_s:config.start_step*step,end_s:config.end_step*step,mean_up_s:up,up_scale_s:config.up_scale_steps*step,up_shape:config.up_shape,duration:{distribution:'fixed',seconds:(config.duration.steps||config.mean_down_steps||1)*step}},time,config.seed);
   renderRandomFaults(host,settings,time);document.dispatchEvent(new Event('modelchange'));
  }}
  // Opening an older random-fault scenario preserves its saved sampling model.
  const randomModel=config.model==='fixed_ratio'?'exponential':config.model;
  const models=[[randomModel,'随机故障'],['fixed_ratio','指定时间故障']];
  const start=seconds?'start_s':'start_step',end=seconds?'end_s':'end_step',up=seconds?'mean_up_s':'mean_up_steps',scale=seconds?'up_scale_s':'up_scale_steps';
  const fixedDuration=seconds?'duration.seconds':'duration.steps',min=seconds?'duration.min_s':'duration.min_steps',max=seconds?'duration.max_s':'duration.max_steps',mean=seconds?'duration.mean_s':'duration.mean_steps';
  const inputs={},fields=[
   ['enabled','启用'+title,'checkbox'],['model','故障方式','text',models],['seed','随机种子','number'],
   [start,'开始时间（'+unit+'）','number'],[end,'停止产生新故障的时间（'+unit+'）','number'],
   ['ratio','停用的'+target+'占比（%）','number'],[up,'平均多久出故障（'+unit+'）','number'],
   [scale,'典型无故障时长（'+unit+'）','number'],['up_shape','风险变化系数','number'],
   ['duration.distribution','持续时间分布','text',[['fixed','固定'],['uniform','均匀分布'],['exponential','指数分布'],['lognormal','对数正态分布']]],
   [fixedDuration,'故障持续时间（'+unit+'）','number'],[min,'最短停用时长（'+unit+'）','number'],[max,'最长停用时长（'+unit+'）','number'],
   [mean,'平均故障持续时间（'+unit+'）','number'],['duration.sigma','停用时长波动幅度','number'],
  ];
  if(!seconds)fields.push(['failure_probability','每个可用'+target+'在一个时间片内出故障的概率（0.1 表示 10%）','number'],['mean_down_steps','平均停用多少个时间片','number']);
  for(const[path,label,type,options]of fields){
   const parent=[start,end].includes(path)?timeWindow:['seed','duration.distribution','duration.sigma'].includes(path)?advancedGrid:grid;
   const input=field(parent,config,path,label,type,options);inputs[path]=input;
   if(type!=='number')continue;
   input.min=path==='seed'?'':[start,'ratio','failure_probability'].includes(path)?'0':'0.000001';
   input.step=path==='seed'||!seconds&&[start,end,fixedDuration,min,max].includes(path)?'1':'any';
   if(path==='failure_probability')input.max='1';
   if(seconds&&[start,end].includes(path))input.max=time.duration_s;
  }
  const ratioInput=inputs.ratio;
  ratioInput.max='100';
  ratioInput.value=Number((config.ratio*100).toPrecision(12));
  ratioInput.oninput=ratioInput.onchange=()=>{
   config.ratio=Number(ratioInput.value)/100;
   document.dispatchEvent(new Event('modelchange'));
  };
  function update(){
   const model=config.model,dist=config.duration.distribution,fixed=model==='fixed_ratio',legacyExp=!seconds&&model==='exponential',show=(path,yes)=>{if(inputs[path]){inputs[path].parentElement.hidden=!yes;inputs[path].disabled=!yes}};
   show('ratio',fixed);show(up,model==='exponential');show(scale,model==='weibull');show('up_shape',model==='weibull');show('failure_probability',model==='per_step');show('mean_down_steps',legacyExp);
   show('duration.distribution',!fixed&&!legacyExp);show(fixedDuration,!fixed&&!legacyExp&&dist==='fixed');show(min,!fixed&&!legacyExp&&dist==='uniform');show(max,!fixed&&!legacyExp&&dist==='uniform');show(mean,!fixed&&!legacyExp&&['exponential','lognormal'].includes(dist));show('duration.sigma',!fixed&&!legacyExp&&dist==='lognormal');
   const windowParent=fixed?grid:advancedGrid;
   if(timeWindow.parentElement!==windowParent)windowParent.append(timeWindow);
   const startLabel=(fixed?'开始停用时间':'随机故障开始采样')+'（'+unit+'）';
   const endLabel=fixed?'恢复时间（'+unit+'）':'停止产生新故障的时间（'+unit+'）';
   inputs[start].setAttribute('aria-label',startLabel);inputs[start].parentElement.firstChild.textContent=startLabel;
   inputs[end].setAttribute('aria-label',endLabel);inputs[end].parentElement.firstChild.textContent=endLabel;
  }
  inputs.model.addEventListener('change',update);
  inputs['duration.distribution'].addEventListener('change',update);
  refresh.push(update);update();
 }
 return ()=>refresh.forEach(update=>update());
}
