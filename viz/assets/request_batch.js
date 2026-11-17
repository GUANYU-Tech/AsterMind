import {post} from '/assets/studio_common.js';

export function createRequestBatch({getExperiment,apply}){
 const $=id=>document.getElementById(id),dialog=$('requestBatchDialog');
 let kind='import',preview=null,configuration='',pending=false;
 const option=(value,text)=>{const item=document.createElement('option');item.value=value;item.textContent=text;return item};
 function reset(){preview=null;$('applyRequestBatch').disabled=true;$('requestBatchStatus').textContent='';$('requestBatchPreview').replaceChildren()}
 function controls(value){pending=value;for(const input of dialog.querySelectorAll('input,textarea,select'))input.disabled=value;for(const id of ['previewRequestBatch','closeRequestBatch'])$(id).disabled=value;$('applyRequestBatch').disabled=value||!preview}
 function open(nextKind){
  const experiment=getExperiment();if(!experiment)throw Error('请先生成部署配置');kind=nextKind;reset();
  $('requestBatchTitle').textContent=kind==='import'?'批量导入':'批量生成';$('requestImportFields').hidden=kind!=='import';$('requestGenerateFields').hidden=kind!=='generate';
  const grounds=new Set(experiment.ground_owners),devices=experiment.physical.devices.filter(row=>grounds.has(row.satellite)&&row.kind==='terminal');
  $('requestBatchGround').replaceChildren(option('','所有地面节点轮换'),...devices.map(row=>option(row.id,row.satellite)));
  $('requestBatchCell').replaceChildren(option('','自动选择'),...experiment.cells.map(row=>option(row.id,row.id)));
  $('requestBatchWindow').max=String(experiment.horizon_ns/1e9);$('requestBatchWindow').value=Math.min(60,experiment.horizon_ns/1e9);
  $('requestBatchMode').value=kind==='generate'?'replace':'append';dialog.showModal();
 }
 dialog.addEventListener('cancel',event=>{if(pending)event.preventDefault()});
 $('closeRequestBatch').onclick=()=>dialog.close();
 for(const input of dialog.querySelectorAll('input,textarea,select'))input.addEventListener('input',reset);
 $('requestFile').onchange=async()=>{reset();const file=$('requestFile').files[0];if(!file)return;try{if(file.size>8_000_000)throw Error('请求文件不能超过 8 MB');controls(true);const extension=file.name.split('.').pop().toLowerCase();$('requestFormat').value=['jsonl','ndjson'].includes(extension)?'jsonl':extension==='json'?'json':'csv';$('requestImportContent').value=new TextDecoder('utf-8',{fatal:true}).decode(await file.arrayBuffer())}catch(error){$('requestBatchStatus').textContent=error instanceof TypeError?'请求文件必须使用 UTF-8 编码':error.message}finally{controls(false)}};
 $('previewRequestBatch').onclick=async()=>{
  if(pending)return;reset();
  try{
   if(kind==='generate')for(const input of $('requestGenerateFields').querySelectorAll('input,select'))if(!input.checkValidity()){input.reportValidity();throw Error(input.validationMessage)}
   const experiment=getExperiment();configuration=JSON.stringify(experiment);controls(true);$('requestBatchStatus').textContent='正在检查请求…';
   const value={experiment,kind,mode:$('requestBatchMode').value};
   if(kind==='import')Object.assign(value,{format:$('requestFormat').value,content:$('requestImportContent').value});
   else value.options={count:Number($('requestBatchCount').value),window_s:Number($('requestBatchWindow').value),distribution:$('requestBatchDistribution').value,seed:Number($('requestBatchSeed').value),input_tokens:Number($('requestBatchInput').value),output_tokens:Number($('requestBatchOutput').value),user_device:$('requestBatchGround').value||null,cell:$('requestBatchCell').value||null};
   const result=await post('/native-scene-requests',value);
   if(configuration!==JSON.stringify(getExperiment()))throw Error('场景已变化，请重新检查请求');
   preview=result;$('requestBatchStatus').textContent=`本批 ${result.imported_count.toLocaleString()} 条 / 应用后共 ${result.total_count.toLocaleString()} 条`;
   const table=document.createElement('table'),head=document.createElement('tr');for(const text of ['请求 ID','到达 / s','地面接入','输入 / 输出 token']){const th=document.createElement('th');th.textContent=text;head.append(th)}table.append(head);
   const shown=result.mode==='append'?result.requests.slice(-result.imported_count):result.requests;
   for(const row of shown.slice(0,5)){const tr=document.createElement('tr');for(const text of [row.id,row.arrival_ns/1e9,row.user_device,`${row.input_tokens} / ${row.output_tokens}`]){const td=document.createElement('td');td.textContent=text;tr.append(td)}table.append(tr)}$('requestBatchPreview').replaceChildren(table);
  }catch(error){preview=null;$('requestBatchStatus').textContent=error.message}
  finally{controls(false)}
 };
 $('applyRequestBatch').onclick=()=>{if(!preview||pending)return;if(configuration!==JSON.stringify(getExperiment())){reset();$('requestBatchStatus').textContent='场景已变化，请重新检查请求';return}apply(preview.requests);dialog.close()};
 return {open};
}
