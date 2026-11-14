import {TRAFFIC_PALETTE,TRAFFIC_LEGEND} from './live_traffic_model.js';

export function attachLivePanelFeedback(container){
 if(!document.getElementById('live-panel-feedback-style')){
  const style=document.createElement('link');style.id='live-panel-feedback-style';style.rel='stylesheet';style.href='/assets/live_panel_feedback.css';document.head.append(style);
 }
 const legend=document.createElement('aside');legend.className='live-traffic-legend';legend.setAttribute('aria-label','传输状态图例');
 const heading=document.createElement('strong');heading.textContent='传输状态';legend.append(heading);
 const list=document.createElement('ul');
 for(const [key,label]of TRAFFIC_LEGEND){
  const item=document.createElement('li'),swatch=document.createElement('i'),text=document.createElement('span');
  swatch.setAttribute('aria-hidden','true');swatch.style.setProperty('--traffic-color','#'+TRAFFIC_PALETTE[key].toString(16).padStart(6,'0'));
  text.textContent=label;item.append(swatch,text);list.append(item);
 }
 legend.append(list);container.append(legend);

 const waiting=document.createElement('div');waiting.className='live-compute-wait';waiting.hidden=true;waiting.setAttribute('role','status');waiting.setAttribute('aria-live','polite');
 const spinner=document.createElement('span');spinner.className='live-wait-spinner';spinner.setAttribute('aria-hidden','true');
 const copy=document.createElement('div'),title=document.createElement('strong'),detail=document.createElement('span');copy.append(title,detail);
 const elapsed=document.createElement('span');elapsed.className='live-wait-elapsed';elapsed.setAttribute('aria-hidden','true');waiting.append(spinner,copy,elapsed);container.append(waiting);
 let phase=null,since=0,lastSecond=-1;
 return {
  update(feedback,now=performance.now()){
   if(!feedback){if(phase!==null){waiting.hidden=true;container.setAttribute('aria-busy','false');phase=null}return}
   if(phase!==feedback.phase){phase=feedback.phase;since=now;lastSecond=-1;title.textContent=feedback.title;detail.textContent=feedback.detail;waiting.hidden=false;container.setAttribute('aria-busy','true')}
   const seconds=Math.floor(Math.max(0,now-since)/1000);
   if(seconds!==lastSecond){lastSecond=seconds;elapsed.textContent=seconds+' s'}
  },
  dispose(){legend.remove();waiting.remove();container.removeAttribute('aria-busy')},
 };
}
