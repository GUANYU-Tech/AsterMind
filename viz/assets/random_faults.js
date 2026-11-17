import {field} from './studio_common.js';

function normalizeGroup(settings,time,seed){
 const config=settings||{},step=Number(time.step_s)||1,horizon=Number(time.duration_s)||120;
 if(config.time_basis==='seconds'||!config.model)return {enabled:false,time_basis:'seconds',model:'exponential',seed,ratio:.01,
  start_s:0,end_s:horizon,mean_up_s:seed===43?2592000:86400,up_scale_s:86400,up_shape:1.5,...config,
  duration:{distribution:'exponential',seconds:60,min_s:30,max_s:120,mean_s:60,sigma:.6,...config.duration}};
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
  const config=settings[key],seconds=config.time_basis==='seconds',unit=seconds?'s':'片',card=document.createElement('fieldset'),legend=document.createElement('legend'),grid=document.createElement('div');
  card.style.cssText='min-width:0;border:1px solid #e2e8f2;border-radius:10px;padding:18px;margin:12px 0 20px;grid-column:1 / -1';legend.textContent=title+' · 独立模型与随机流';grid.className='fields form-grid';card.append(legend,grid);host.append(card);
  if(!seconds){const convert=document.createElement('button');convert.type='button';convert.textContent='将旧逐片模型转换为连续秒制';card.append(convert);convert.onclick=()=>{
   const step=Number(time.step_s)||1,p=config.failure_probability,up=config.model==='per_step'?(p>0&&p<1?-step/Math.log1p(-p):step):config.mean_up_steps*step;
   settings[key]=normalizeGroup({enabled:config.enabled,time_basis:'seconds',model:config.model==='per_step'?'exponential':config.model,seed:config.seed,ratio:config.ratio,start_s:config.start_step*step,end_s:config.end_step*step,mean_up_s:up,up_scale_s:config.up_scale_steps*step,up_shape:config.up_shape,duration:{distribution:'fixed',seconds:(config.duration.steps||config.mean_down_steps||1)*step}},time,config.seed);
   renderRandomFaults(host,settings,time);document.dispatchEvent(new Event('modelchange'));
  }}
  const models=seconds?[['exponential','指数健康期 · 连续更新过程'],['weibull','Weibull 健康期 · 连续更新过程'],['fixed_ratio','固定比例 · 秒级指定区间']]:[['per_step','旧 Bernoulli · 每片触发'],['exponential','旧指数 · 向上取整到片'],['weibull','旧 Weibull · 向上取整到片'],['fixed_ratio','旧固定比例 · 指定片区间']];
  const start=seconds?'start_s':'start_step',end=seconds?'end_s':'end_step',up=seconds?'mean_up_s':'mean_up_steps',scale=seconds?'up_scale_s':'up_scale_steps';
  const fixedDuration=seconds?'duration.seconds':'duration.steps',min=seconds?'duration.min_s':'duration.min_steps',max=seconds?'duration.max_s':'duration.max_steps',mean=seconds?'duration.mean_s':'duration.mean_steps';
  const inputs={},fields=[['enabled','启用'+title,'checkbox'],['model',title+'模型','text',models],['seed',title+' seed','number'],[start,'开始注入 / '+unit,'number'],[end,'停止注入（不含）/ '+unit,'number'],['ratio','一次抽取比例 (0–1)','number'],[up,'平均健康期 / '+unit,'number'],[scale,'Weibull 尺度 η / '+unit,'number'],['up_shape','Weibull 形状 β','number'],['duration.distribution','故障持续时间分布','text',[['fixed','固定时长'],['uniform','均匀分布'],['exponential','指数分布'],['lognormal','对数正态分布']]],[fixedDuration,'固定持续 / '+unit,'number'],[min,'最短持续 / '+unit,'number'],[max,'最长持续 / '+unit,'number'],[mean,'持续时间均值 / '+unit,'number'],['duration.sigma','对数空间 σ','number']];
  if(!seconds)fields.push(['failure_probability','每个健康对象每片故障概率','number'],['mean_down_steps','旧指数平均故障期 / 片','number']);
  for(const[path,label,type,options]of fields){const input=field(grid,config,path,label,type,options);inputs[path]=input;if(type==='number'){input.min=path==='seed'?'':path===start||path==='ratio'||path==='failure_probability'?'0':'0.000001';input.step=(!seconds&&[start,end,fixedDuration,min,max].includes(path)||path==='seed')?'1':'any';if(path==='ratio'||path==='failure_probability')input.max='1';if(seconds&&[start,end].includes(path))input.max=time.duration_s}}
  const note=document.createElement('p');note.style.cssText='grid-column:1 / -1;color:#8190a6;font-size:11px;line-height:1.8';grid.append(note);
  function update(){
   const model=config.model,dist=config.duration.distribution,fixed=model==='fixed_ratio',legacyExp=!seconds&&model==='exponential',show=(path,yes)=>{if(inputs[path]){inputs[path].parentElement.hidden=!yes;inputs[path].disabled=!yes}};
   show('ratio',fixed);show(up,model==='exponential');show(scale,model==='weibull');show('up_shape',model==='weibull');show('failure_probability',model==='per_step');show('mean_down_steps',legacyExp);
   show('duration.distribution',!fixed&&!legacyExp);show(fixedDuration,!fixed&&!legacyExp&&dist==='fixed');show(min,!fixed&&!legacyExp&&dist==='uniform');show(max,!fixed&&!legacyExp&&dist==='uniform');show(mean,!fixed&&!legacyExp&&['exponential','lognormal'].includes(dist));show('duration.sigma',!fixed&&!legacyExp&&dist==='lognormal');
   note.textContent=seconds?'发生间隔与持续时间按秒抽样，不对齐物理时间片。恢复超过时域时保留实际抽样时长。参数为每个对象的实验假设，需按测量数据标定。':'旧配置按时间片抽样和取整，保留原实验语义；可转换为秒制连续模型。每片 '+time.step_s+' s。';
  }
  Object.values(inputs).forEach(input=>input.addEventListener('change',update));refresh.push(update);update();
 }
 return ()=>refresh.forEach(update=>update());
}
