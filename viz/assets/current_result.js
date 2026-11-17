import {get} from '/assets/studio_common.js';
const $=id=>document.getElementById(id);
const labels={running:'正在运行',paused:'已暂停',preparing:'正在准备',cancelled:'已停止',interrupted:'已中断',failed:'执行失败'};
function link(text,url,primary=false){const a=document.createElement('a');a.textContent=text;a.href=url;a.className='button'+(primary?' primary':'');$('actions').append(a)}
async function update(){
 try{
  const current=await get('/current-experiment');$('actions').replaceChildren();
  if(!current.id){$('heading').textContent='尚未启动当前实验';$('status').textContent='启动仿真后，结果会显示在这里。';link('配置实验 →','/configuration',true);return}
  const id=encodeURIComponent(current.id),job=await get('/runs/'+id+'/status');
  if(job.state==='completed'){location.replace('/runs/'+id+'/result?current=1');return}
  $('heading').textContent=labels[job.state]||'实验尚未完成';$('status').textContent=job.error||'结果将在本次实验完成后显示。';
  link('查看当前实验 →',job.runtime_url||'/runs/'+id+'/progress',true);
  if(['running','preparing','paused','stopping'].includes(job.state))setTimeout(update,1800);
 }catch(e){$('heading').textContent='当前实验暂时无法读取';$('status').textContent=e.message}
}
update();
