// A result is loaded only after an explicit URL or user selection identifies it.
export async function setupResultSelection({get,select,button,status,navigate}){
 const placeholder=document.createElement('option');placeholder.value='';placeholder.textContent='请选择实验';
 select.replaceChildren(placeholder);select.disabled=true;button.disabled=true;
 const labels={completed:'已完成',running:'运行中',failed:'失败',cancelled:'已停止',interrupted:'已中断',stopping:'停止中'};
 const rows=await get('/task-catalog'),ids=new Set();
 for(const row of rows){
  if(!row.id)continue;ids.add(row.id);
  const option=document.createElement('option');option.value=row.id;
  option.textContent=`${row.title&&row.title!==row.id?row.title+' ':''}${row.id} ${labels[row.state]||row.state}`;
  select.append(option);
 }
 select.value='';select.disabled=!ids.size;
 status.textContent=ids.size?'选择一次实验，查看对应的性能、回放与分析。':'暂无实验记录，可以先新建任务。';
 select.onchange=()=>{button.disabled=!ids.has(select.value)};
 button.onclick=()=>{if(ids.has(select.value))navigate('/results?run='+encodeURIComponent(select.value))};
}
