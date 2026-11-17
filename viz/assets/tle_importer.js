import {get,post} from '/assets/studio_common.js';

export function importedShell(template,analysis){
  const shell=structuredClone(template);
  shell.shell_id='tle-'+Date.now();shell.source=structuredClone(analysis.suggested_source);
  for(const key of ['radio_profile','hardware_link_profiles','radio_capacity_bps','capacity_assumptions','optical_terminal_bandwidth_bps','terminal_assumptions'])delete shell[key];
  shell.isl_terminals=3;shell.sgl_terminals=8;shell.topology='honeycomb';shell.topology_repair='none';
  shell.wrap_planes=analysis.inferred_grid_shape[0]%2===0;
  shell.wrap_slots=analysis.inferred_grid_shape[1]%2===0;
  return shell;
}

export function initTleImporter({readConfig,replaceConfig,readShell,render}){
  const $=id=>document.getElementById(id);let catalog=[];
  const status=message=>$('tleStatus').textContent=message;
  $('tleFile').onchange=()=>{
    const file=$('tleFile').files[0];$('importTle').disabled=!file;
    status(file?'已选择 '+file.name+'，点击“导入文件并生成通信星座”。':'请选择单壳层 .tle 或 .txt 文件。');
  };
  $('importTle').onclick=async()=>{
    const file=$('tleFile').files[0];if(!file)return;
    try{
      $('importTle').disabled=true;status('正在校验 TLE 并分析共同历元的轨道面和槽位…');
      const analysis=await post('/inspect-tle',{tle_text:await file.text()});
      const value=structuredClone(readConfig());
      value.constellations.communication_shells=[importedShell(readShell(),analysis)];
      value.time.epoch_utc=analysis.simulation_epoch_utc;
      value.network.faults=[];value.network.explicit_pairs=[];value.route_probes=[];
      delete value.network.nodes;delete value.network.communication_shells;
      value.network.ground_roles=value.network.ground_roles?.includes('compute')?['communication','compute']:['communication'];
      replaceConfig(value);render();
      status('已从 '+file.name+' 生成 '+analysis.satellite_count+' 颗真实通信卫星的配置；推断网格 '+analysis.inferred_grid_shape.join('×')+'，匹配 '+analysis.grid_placed_count+' 颗、未匹配 '+analysis.grid_unplaced_count+' 颗、空位 '+analysis.nominal_grid_empty_positions+'。全部实测卫星保留；点击“保存并演示物理场景”生成 3D 回放。硬件使用可修改的工程假设。');
    }catch(e){status('导入失败：'+e.message)}finally{$('importTle').disabled=false}
  };
  function showSample(){
    const row=catalog.find(r=>r.id===$('tleSample').value);
    $('loadTleSample').disabled=!row;$('downloadTle').hidden=!row;
    if(row){$('downloadTle').href='/tle-files/'+row.id+'.tle';$('downloadTle').download=row.id+'.tle';
      $('tleComparison').textContent='已筛选 '+row.selected_count+' 颗 / 历史参考 '+row.nominal_reference.planes+'×'+row.nominal_reference.satellites_per_plane+'='+row.nominal_reference.total+' 颗；实测推断网格 '+row.inferred_grid_shape.join('×')+'，匹配 '+row.grid_placed_count+'、未匹配 '+row.grid_unplaced_count+'、空位 '+row.nominal_grid_empty_positions+'。历史差值不等于故障数；样例已包含来源与筛选规则。';}
  }
  $('tleSample').onchange=showSample;
  $('loadTleSample').onclick=async()=>{
    try{
      $('loadTleSample').disabled=true;
      const doc=await get('/tle-samples/'+$('tleSample').value);
      replaceConfig(doc.physical_network);$('name').value=doc.summary.title;render();
      status('已载入本地 CelesTrak 单壳层：'+doc.summary.selected_count+' 颗通信卫星；不是全部 '+doc.summary.raw_catalog_count+' 颗。可直接保存演示，也可下载 .tle 文件后通过上方文件入口重新导入。');
    }catch(e){status(e.message)}finally{$('loadTleSample').disabled=false}
  };
  get('/tle-samples').then(rows=>{
    catalog=rows;$('tleSample').replaceChildren();
    for(const row of rows){const option=document.createElement('option');option.value=row.id;option.textContent=row.title;$('tleSample').append(option)}
    if(!rows.length)status('暂无已筛选的本地样例；仍可导入自己的单壳层 TLE。');showSample();
  }).catch(e=>status('本地 TLE 样例读取失败：'+e.message));
  const configured=readConfig().constellations.communication_shells.filter(s=>s.source.type==='tle');
  if(configured.length){const count=configured.reduce((n,s)=>n+(s.source.tle_text||'').split('\n').filter(l=>l.startsWith('1 ')).length,0);status('当前模型已保存 '+configured.length+' 个 TLE 通信壳层，共 '+count+' 颗。轨道数据已保存在模型内，无需重复选择文件；也可从这里替换。')}
}
