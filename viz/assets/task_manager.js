import {get,post} from './studio_common.js';
import {formatProgressLine,formatPhysicalSummary} from './progress_log.js';

export const stateLabels={running:'正在运行',stopping:'正在停止',cancelled:'已手动停止',completed:'已完成',failed:'失败',interrupted:'已中断',paused:'已暂停',aborted:'已提前结束',terminated:'已终止',unknown:'状态暂不可用'};
const validId=id=>typeof id==='string'&&/^[a-zA-Z0-9_-]+$/.test(id);
function runDate(id){const m=id.match(/-(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})(?:-(\d{6}))?$/);return m?{key:m.slice(1).join(''),text:`${m[1]}-${m[2]}-${m[3]} ${m[4]}:${m[5]}:${m[6]}`}:{key:'',text:'创建时间未记录'}}
function taskName(task){if(task.id.startsWith('native-scene-')&&(!task.title||task.title===task.id||['全部通信使用 FlowSim','卫星内通信使用 ProtoBridge','Cell 内通信使用 ProtoBridge'].includes(task.title)))return '完整场景仿真';if(task.title&&task.title!==task.id)return task.title.replaceAll(' · ',' / ');const grid=task.id.match(/^(?:starlink|comm)-([^-]+)-grid-(\d+)x(\d+)-/);return grid?`通信星座 ${grid[1]==='low'?'':grid[1]+' km '}（${grid[2]} 轨道 × ${grid[3]} 星）`:task.id.startsWith('qwen3-')?'计算 Cell 仿真':'仿真实验'}
const deletedTaskIds=new Set();
let taskRevision=0,dialogSerial=0;
export function rememberTask(id){
  if(!validId(id))return;
  try{const ids=JSON.parse(localStorage.getItem('orbitserve.tasks')||'[]');localStorage.setItem('orbitserve.tasks',JSON.stringify([...new Set([id,...ids.filter(validId)])].slice(0,100)))}catch{}
}
function remembered(){try{return JSON.parse(localStorage.getItem('orbitserve.tasks')||'[]').filter(validId)}catch{return []}}
function forgetTask(id){try{localStorage.setItem('orbitserve.tasks',JSON.stringify(remembered().filter(value=>value!==id)))}catch{}}
function notify(message){
  document.getElementById('taskNotice')?.remove();
  const notice=document.createElement('div');notice.id='taskNotice';notice.className='task-notice';notice.setAttribute('role','status');notice.textContent=message;document.body.append(notice);setTimeout(()=>notice.remove(),6000);
}
function taskDialog({task,title,description,confirmLabel,danger,execute}){
  const dialog=document.createElement('dialog');dialog.className='task-dialog';
  const heading=document.createElement('h2');heading.id='taskDialogTitle'+(++dialogSerial);heading.textContent=title;
  const text=document.createElement('p');text.id=heading.id+'Description';text.textContent=description;
  const id=document.createElement('code');id.className='task-dialog-id';id.textContent=task.id;
  const message=document.createElement('p');message.setAttribute('role','alert');message.className='error';
  const pending=document.createElement('p');pending.setAttribute('role','status');pending.hidden=true;
  const actions=document.createElement('div');actions.className='actions';
  const cancel=document.createElement('button');cancel.textContent='取消';cancel.type='button';
  const confirm=document.createElement('button');confirm.textContent=confirmLabel;confirm.type='button';confirm.className=danger?'danger':'primary';
  dialog.setAttribute('aria-labelledby',heading.id);dialog.setAttribute('aria-describedby',text.id);
  let working=false;
  cancel.onclick=()=>dialog.close();dialog.addEventListener('cancel',event=>{if(working)event.preventDefault()});
  dialog.addEventListener('close',()=>dialog.remove());
  confirm.onclick=async()=>{
    if(working)return;working=true;cancel.disabled=confirm.disabled=true;message.textContent='';pending.hidden=false;
    pending.textContent=task.can_stop_and_delete&&danger?'正在停止任务并清理本地文件…':'正在处理，请稍候…';dialog.setAttribute('aria-busy','true');
    try{await execute();dialog.close()}
    catch(error){message.textContent=error.message;message.tabIndex=-1;message.focus()}
    finally{working=false;cancel.disabled=confirm.disabled=false;pending.hidden=true;dialog.removeAttribute('aria-busy')}
  };
  actions.append(cancel,confirm);dialog.append(heading,text,id,message,pending,actions);document.body.append(dialog);dialog.showModal();cancel.focus();
}
async function boundedMap(values,action){const rows=[];let next=0;await Promise.all(Array.from({length:Math.min(4,values.length)},async()=>{while(next<values.length){const value=values[next++];rows.push(await action(value))}}));return rows}

export async function loadTasks(){
  // Discover jobs on older running consoles without restarting their workers.
  let seeds=[],partial=false,fallback=false;
  try{seeds=await get('/task-catalog')}catch{
    fallback=true;
    const [runs,models]=await Promise.allSettled([get('/run-catalog'),get('/physical-catalog')]);
    if(runs.status==='rejected'&&models.status==='rejected')throw Error('任务列表暂时无法读取，请刷新重试');
    partial=runs.status==='rejected'||models.status==='rejected';
    seeds=runs.status==='fulfilled'?runs.value:[];
    if(models.status==='fulfilled'){
      const jobs=await boundedMap(models.value,async model=>{
        try{const data=await get('/physical-data/'+encodeURIComponent(model.id));return data.run_id?{id:data.run_id,title:model.name}:null}catch{partial=true;return null}
      });
      seeds.push(...jobs.filter(Boolean));
    }
  }
  const byId=new Map(seeds.filter(row=>validId(row.id)&&!deletedTaskIds.has(row.id)).map(row=>[row.id,row]));
  if(fallback)for(const id of remembered())if(!byId.has(id))byId.set(id,{id,title:id});
  const tasks=await boundedMap([...byId.values()],async row=>{
    try{const job=await get('/runs/'+encodeURIComponent(row.id)+'/status');return {...row,...job,title:row.title||row.id}}
    catch(error){return {...row,state:'unknown',error:error.message}}
  });
  tasks.sort((a,b)=>runDate(b.id).key.localeCompare(runDate(a.id).key)||b.id.localeCompare(a.id));
  return {tasks:tasks.filter(task=>!deletedTaskIds.has(task.id)),partial};
}

export function renderTask(container,task){
  const row=document.createElement('article');row.className='row task-row';
  const heading=document.createElement('h3'),badge=document.createElement('span');
  badge.className='task-badge task-'+task.state;badge.textContent=stateLabels[task.state]||task.state;
  heading.append(badge,document.createTextNode(' '+taskName(task)));
  const id=document.createElement('small');id.textContent='创建时间 '+runDate(task.id).text;id.title='记录编号：'+task.id;
  const actions=document.createElement('div');actions.className='actions';
  const link=(label,suffix)=>{const a=document.createElement('a');a.className='button';a.href='/runs/'+encodeURIComponent(task.id)+'/'+suffix;a.textContent=label;actions.append(a)};
  link(task.state==='running'?'查看进度与日志':'查看状态与日志','progress?stay=1');
  if(task.state==='completed')link(task.preview_url?'查看物理场景':'查看结果',task.preview_url?'viewer':'result');
  if(task.state==='completed'&&task.runtime_url)link('场景回放','result?view=scene');
  const action=(label,verb,confirmation)=>{
    const button=document.createElement('button');button.textContent=label;
    if(verb==='delete')button.className='danger-outline';
    button.onclick=()=>taskDialog({task,title:label,description:confirmation,confirmLabel:verb==='delete'?(task.can_stop_and_delete?'停止并永久删除':'永久删除'):label,danger:verb==='delete',execute:async()=>{
      button.disabled=true;
      try{
        const job=await post('/runs/'+encodeURIComponent(task.id)+'/'+verb,verb==='delete'?{stop_running:!!task.can_stop_and_delete}:{});
        if(verb==='rerun'){rememberTask(job.id);location.href='/runs/'+encodeURIComponent(job.id)+'/progress?stay=1';return}
        taskRevision++;
        if(verb==='delete'){
          deletedTaskIds.add(task.id);forgetTask(task.id);
          if(location.pathname.startsWith('/runs/'+task.id+'/')){location.href='/experiments';return}
          row.remove();notify('任务及本地文件已删除');
        }else notify('任务已停止');
        document.dispatchEvent(new CustomEvent('taskschange',{detail:{id:task.id,verb,job}}));
      }finally{button.disabled=false}
    }});actions.append(button);
  };
  if(task.can_stop)action('停止任务','stop','停止此任务并保留已有文件？任务：'+task.id);
  if(task.can_rerun)action('重新运行','rerun','使用原始输入创建独立的新任务？原有文件将保留。');
  if(task.can_delete||task.can_stop_and_delete)action(task.can_stop_and_delete?'停止并删除':'删除记录与本地文件','delete',(task.can_stop_and_delete?'将先停止任务，等待计算进程退出，再永久删除任务输入、日志和结果。':'将永久删除此任务的输入、日志和结果。')+'删除后无法恢复。保存的场景配置会保留；仍被其他场景或任务引用的数据需先解除引用。');
  row.append(heading,id,actions);
  const last=(task.log_tail||'').trim().split('\n').filter(Boolean).at(-1);
  if(task.error||last){const detail=document.createElement('p');detail.textContent=(task.error||(task.preview_url&&task.checks?formatPhysicalSummary(task.checks):formatProgressLine(last))).slice(-400);if(task.error)detail.className='error';row.append(detail)}
  container.append(row);
}

export function mountTasks({container,status,refresh,filter,activeOnly=false}){
  let timer,busy=false,queued=false,previous=null,snapshot=null;
  function paint({tasks,partial}){
      tasks=tasks.filter(t=>!deletedTaskIds.has(t.id));
      const active=tasks.filter(t=>['running','stopping'].includes(t.state));
      const selected=activeOnly?'running':filter?.value||'all';
      const rows=tasks.filter(t=>selected==='all'||t.state===selected);
      const signature=JSON.stringify(rows);
      if(signature!==previous){container.replaceChildren();for(const task of rows)renderTask(container,task);previous=signature}
      status.textContent=(activeOnly?active.length+' 个正在运行的任务':tasks.length+' 个任务，'+active.length+' 个正在运行，当前显示 '+rows.length+' 个')+'。按创建时间倒序排列'+(partial?'；部分记录读取失败，可手动刷新':'');
      if(!rows.length&&container.childElementCount===0){const empty=document.createElement('p');empty.textContent=activeOnly?'当前未发现正在运行的任务。':'此筛选条件下没有任务。';container.append(empty)}
  }
  async function update(){
    clearTimeout(timer);if(busy){queued=true;return}busy=true;refresh.disabled=true;
    const revision=taskRevision;
    try{
      const loaded=await loadTasks();
      if(revision!==taskRevision){queued=true;return}
      snapshot=loaded;paint(snapshot);
    }catch(error){status.textContent=error.message+'；保留上次列表，稍后重试'}
    finally{busy=false;refresh.disabled=false;if(queued){queued=false;queueMicrotask(update)}else timer=setTimeout(update,5000)}
  }
  refresh.onclick=update;if(filter)filter.onchange=()=>{previous=null;update()};
  document.addEventListener('visibilitychange',()=>{if(!document.hidden)update()});
  document.addEventListener('taskschange',event=>{
    const change=event.detail;
    if(snapshot&&change){snapshot.tasks=change.verb==='delete'?snapshot.tasks.filter(t=>t.id!==change.id):snapshot.tasks.map(t=>t.id===change.id?{...t,...change.job}:t);paint(snapshot)}
    previous=null;update();
  });
  update();return update;
}
