import {post} from '/assets/studio_common.js';

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
  const $=id=>document.getElementById(id);
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
      status('已从 '+file.name+' 生成 '+analysis.satellite_count+' 颗通信卫星的配置；推断网格 '+analysis.inferred_grid_shape.join('×')+'，匹配 '+analysis.grid_placed_count+' 颗、未匹配 '+analysis.grid_unplaced_count+' 颗、空位 '+analysis.nominal_grid_empty_positions+'。可在“通信星座”中查看并配置硬件。');
    }catch(e){status('导入失败：'+e.message)}finally{$('importTle').disabled=false}
  };
  const configured=readConfig().constellations.communication_shells.filter(s=>s.source.type==='tle');
  if(configured.length){const count=configured.reduce((n,s)=>n+(s.source.tle_text||'').split('\n').filter(l=>l.startsWith('1 ')).length,0);status('当前模型已保存 '+configured.length+' 个 TLE 通信壳层，共 '+count+' 颗。轨道数据已保存在模型内，无需重复选择文件；也可从这里替换。')}
}
