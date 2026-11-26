import {post} from '/assets/studio_common.js';

export function importedShell(template,analysis,id){
  const shell=structuredClone(template);
  shell.shell_id=id;shell.source=structuredClone(analysis.suggested_source);
  for(const key of ['radio_profile','hardware_link_profiles','radio_capacity_bps','capacity_assumptions','optical_terminal_bandwidth_bps','terminal_assumptions'])delete shell[key];
  shell.isl_terminals=3;shell.sgl_terminals=8;shell.topology=analysis.inferred_grid_shape?'honeycomb':'explicit';shell.topology_repair='none';
  shell.wrap_planes=analysis.inferred_grid_shape?.[0]%2===0;
  shell.wrap_slots=analysis.inferred_grid_shape?.[1]%2===0;
  return shell;
}

export async function inspectTleShells(file,config,template){
  const data=await post('/inspect-tle',{tle_text:await file.text()});
  const {communication_shells,compute_formations}=config.constellations;
  const used=new Set(communication_shells.map(shell=>shell.shell_id));
  const catalogs=new Set();
  for(const shell of [...communication_shells,...compute_formations]){
    if(shell.source.type!=='tle')continue;
    for(const line of (shell.source.tle_text||'').split('\n'))if(line.startsWith('1 '))catalogs.add(line.slice(2,7).trim());
  }
  let index=1;
  const analyses=data.shells??[data];
  const additions=analyses.map(analysis=>{
    for(const line of analysis.suggested_source.tle_text.split('\n')){
      if(line.startsWith('1 ') && catalogs.has(line.slice(2,7).trim()))throw Error('目录号已存在。请在对应壳层中更新 TLE。');
    }
    while(used.has('tle-'+index))index++;
    return importedShell(template,analysis,'tle-'+index++);
  });
  return {additions,analyses,satelliteCount:data.satellite_count};
}

export function showTleResult(card,result){
  const {additions,analyses,satelliteCount}=result;
  const status=document.createElement('p');
  status.setAttribute('role','status');
  status.textContent='已追加 '+additions.length+' 个通信壳层，共 '+satelliteCount+' 颗卫星。';
  if(analyses.some(a=>!a.inferred_grid_shape))status.textContent+=' 部分壳层未识别到规则网格，请检查星间拓扑。';
  card.querySelector('h3').after(status);
  const details=document.createElement('details');
  const caption=document.createElement('summary');caption.textContent='TLE 壳层分组详情';details.append(caption);
  analyses.forEach((analysis,i)=>{
    if(!analysis.inferred_grid_shape)return;
    const row=document.createElement('p');
    row.textContent=additions[i].shell_id+'：推断网格 '+analysis.inferred_grid_shape.join('×')+'，已匹配 '+analysis.grid_placed_count+' 颗，未匹配 '+analysis.grid_unplaced_count+' 颗。';
    details.append(row);
  });
  if(details.childElementCount>1)status.after(details);
}
