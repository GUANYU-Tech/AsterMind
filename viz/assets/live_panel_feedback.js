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

 // Execution status belongs to the page toolbar. The globe keeps its geometry
 // and traffic visible while the solver prepares the next batch.
 return {
  update(){},
  dispose(){legend.remove()},
 };
}
