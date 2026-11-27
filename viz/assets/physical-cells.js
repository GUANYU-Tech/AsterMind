import {sceneCellDraft,sceneCellLayout} from './scene-cell-parameters.js';

const pad = (number, width) => String(number).padStart(width, '0');

export function cellCount({source}) {
  if(source.type==='tle')return Object.keys(source.cell_members||{}).length;
  const count=source.planes*source.clusters_per_plane;
  return [source.planes,source.clusters_per_plane,count].every(n=>Number.isSafeInteger(n)&&n>0)?count:0;
}

export function cellKeys(formation,start=0,end=cellCount(formation)) {
  const {source}=formation;
  if(source.type==='tle')return Object.keys(source.cell_members||{}).slice(start,end);
  return Array.from({length:Math.min(end,cellCount(formation))-start},(_,offset)=>{
    const index=start+offset, slots=source.clusters_per_plane;
    return 'p'+pad(Math.floor(index/slots),3)+'-s'+pad(index%slots,4);
  });
}

export function cellIndex(formation,key) {
  const {source}=formation;
  if(source.type==='tle')return Object.keys(source.cell_members||{}).indexOf(key);
  const match=/^p(\d+)-s(\d+)$/.exec(key);
  if(!match)return -1;
  const plane=Number(match[1]), slot=Number(match[2]);
  return plane<source.planes && slot<source.clusters_per_plane && key==='p'+pad(plane,3)+'-s'+pad(slot,4)
    ? plane*source.clusters_per_plane+slot : -1;
}

export function satelliteCount(formation) {
  if(formation.source.type==='tle')return Object.values(formation.source.cell_members||{}).reduce((n,group)=>n+group.length,0);
  const base=formation.cluster.satellites_per_cluster;
  let count=cellCount(formation)*base;
  for(const [id,entry] of Object.entries(formation.cell_overrides||{})){
    if(cellIndex(formation,id.slice(formation.name.length+1))>=0)count+=(entry.satellites_per_cluster??base)-base;
  }
  return count;
}

// Only new UI presets opt into static geometry. Stored datasets retain their motion model.
export function newPhysicalPreset(network) {
  for (const formation of network.constellations.compute_formations) {
    if (formation.source.type !== 'tle' && !formation.cluster.motion_model) {
      formation.cluster.motion_model = 'static_formation';
      delete formation.cluster.grid_shape;
    }
  }
  return network;
}

export function cellGeometry(formation, key) {
  const base = formation.cluster;
  const geometry = {topology:formation.topology || 'ring'};
  if (formation.source.type !== 'tle') Object.assign(geometry, {
    satellites_per_cluster:base.satellites_per_cluster,
    ...(base.motion_model ? {motion_model:base.motion_model} : {}),
    ...(base.grid_shape ? {grid_shape:base.grid_shape} : {}),
    formation_pitch_m:base.formation_pitch_m, altitude_km:formation.source.altitude_km,
  });
  return {...geometry,...formation.cell_overrides?.[formation.name+'-'+key]};
}

export function memberIds(formation, key, geometry) {
  if (formation.source.type === 'tle') return (formation.source.cell_members[key] || []).map(n=>formation.name+'-'+n);
  return Array.from({length:geometry.satellites_per_cluster}, (_, n)=>formation.name+'-'+key+'-m'+pad(n,3));
}

// Older Cell layouts also stored a request generator. Saved experiment.requests
// already contains the actual requests, including any later scene edits.
export function removeLegacyCellWorkload(layout) {
  if (!layout) return;
  delete layout.workload_ground;
  if (layout.configuration) delete layout.configuration.requests;
}

// Store templates by member position so a hand-assigned rank follows the same member in each Cell.
export function bindLayout(layout, members) {
  if (!layout) return undefined;
  const previous=Object.keys(layout.satellite_gpus);
  const result=structuredClone(layout);
  if (result.configuration) {
    const config = result.configuration;
    const remap = id => {
      const target = members[previous.indexOf(id)];
      if (!target) throw Error('配置引用了此编队中不存在的卫星');
      return target;
    };
    config.hardware.satellite_ids = [...members];
    config.hardware.satellite_count = members.length;
    for (const row of config.deployment.placements) row.satellite_id = remap(row.satellite_id);
    if (!layout.custom_links) {
      config.topology.inter_satellite.links=config.topology.inter_satellite.links.filter(link=>
        members[previous.indexOf(link.source)] && members[previous.indexOf(link.target)]);
    }
    for (const link of config.topology.inter_satellite.links) {
      link.source = remap(link.source);
      link.target = remap(link.target);
    }
  }
  result.satellite_gpus=Object.fromEntries(members.map(id=>[id,layout.deployment.gpus_per_satellite]));
  result.rank_devices=layout.rank_devices.map(address=>{
    const [sat, slot]=address.split(/\/(?=gpu\d+$)/);
    const target=members[previous.indexOf(sat)];
    if (!target) throw Error('Rank 映射超出当前编队的卫星数量，请重新配置。');
    return target+'/'+slot;
  });
  return result;
}

export function applyCell(formation, key, geometry, layout, all) {
  const settings = formation.compute_parameters ??= {cells:{}};
  if (all) {
    if (formation.source.type !== 'tle') {
      const {satellites_per_cluster,grid_shape,formation_pitch_m}=geometry;
      Object.assign(formation.cluster,{satellites_per_cluster,formation_pitch_m});
      if(geometry.motion_model==='static_formation'){
        formation.cluster.motion_model='static_formation';
        delete formation.cluster.grid_shape;
      }else if(grid_shape)formation.cluster.grid_shape=grid_shape;
      formation.source.altitude_km=geometry.altitude_km;
    }
    formation.topology=geometry.topology;
    delete formation.cell_overrides;
    settings.template=structuredClone(layout);
    settings.cells={};
  } else {
    const overrides=structuredClone(geometry);
    if(overrides.motion_model===formation.cluster.motion_model)delete overrides.motion_model;
    (formation.cell_overrides??={})[formation.name+'-'+key]=overrides;
    (settings.cells??={})[key]=structuredClone(layout);
  }
}

export function cellLayout(formation,key,members) {
  const settings=formation.compute_parameters;
  return bindLayout(settings?.cells?.[key] || settings?.template,members);
}

// The standalone editor needs a complete configuration, including derived link fields.
export function inheritCellBudget(config,profile,links=config.topology.inter_satellite.links) {
  const mode=profile.capacity_model==='fixed'?'fixed':'islsim';
  config.topology.inter_satellite.links=links.map(link=>({
    ...link,mode,bandwidth_gbps:mode==='fixed'?profile.bandwidth_bps/1e9:null,
  }));
  if(mode==='islsim' && profile.islsim_config)config.fso=structuredClone(profile.islsim_config);
}

export function physicalPayload(network, mode, epStrategy, capabilities, linkBandwidthFactors={}) {
  const physical_network=structuredClone(network);
  const formations={};
  const spec = physical_network.network;
  const profiles = new Map();
  for (const formation of physical_network.constellations.compute_formations) {
    const prefix=formation.name+'-';
    if (formation.cell_overrides) formation.cell_overrides=Object.fromEntries(Object.entries(formation.cell_overrides).filter(([id])=>id.startsWith(prefix)&&cellIndex(formation,id.slice(prefix.length))>=0));
    if (formation.compute_parameters) {
      const settings=formation.compute_parameters;
      settings.cells=Object.fromEntries(Object.entries(settings.cells||{}).filter(([key])=>cellIndex(formation,key)>=0));
      for (const layout of [settings.template,...Object.values(settings.cells)]) {
        if (!layout?.configuration) continue;
        delete layout.configuration.fast_link_contention;
        if(spec?.optical_link_model)inheritCellBudget(layout.configuration,spec.link_profiles.cell_isl);
        if (!capabilities||layout.configuration.compute_mode===mode) continue;
        const members=layout.configuration.hardware.satellite_ids;
        const draft=sceneCellDraft(capabilities,members,mode,layout);
        Object.assign(layout,sceneCellLayout(draft,capabilities,members),{configuration:draft});
      }
      if(settings.cell_names)settings.cell_names=Object.fromEntries(Object.entries(settings.cell_names).filter(([key])=>cellIndex(formation,key)>=0));
      formations[formation.name]=settings;
      for (const key of cellKeys(formation)) {
        const saved = settings.cells[key] || settings.template;
        if (!saved?.custom_links) continue;
        const ids = memberIds(formation,key,cellGeometry(formation,key));
        const config = bindLayout(saved,ids).configuration;
        const links = config.topology.inter_satellite.links.map(link => {
          if(spec.optical_link_model)return {source:link.source,target:link.target,profile_id:'cell_isl'};
          const profile = {...spec.link_profiles.cell_isl, capacity_model:link.mode || 'islsim'};
          if (profile.capacity_model === 'fixed') {
            profile.bandwidth_bps = link.bandwidth_gbps * 1e9;
            delete profile.islsim_config;
            delete profile.islsim_source_sha256;
          } else profile.islsim_config = config.fso;
          const signature = JSON.stringify(profile);
          if (!profiles.has(signature)) {
            let suffix=profiles.size;
            while (Object.hasOwn(spec.link_profiles,'editor-cell-'+suffix)) suffix++;
            const name = 'editor-cell-' + suffix;
            spec.link_profiles[name] = profile;
            profiles.set(signature,name);
          }
          return {source:link.source,target:link.target,profile_id:profiles.get(signature)};
        });
        (spec.cell_link_overrides ??= {})[formation.name+'-'+key] = links;
      }
      delete formation.compute_parameters;
    }
  }
  if (spec?.optical_link_model) {
    const profiles=spec.link_profiles;
    for (const kind of ['communication_isl','cross_layer','cell_isl','inter_cell']) {
      if (spec.optical_link_model==='fixed') {
        delete profiles[kind].islsim_config;
        delete profiles[kind].islsim_source_sha256;
      }
    }
    const directions=['user_uplink','user_downlink','feeder_uplink','feeder_downlink'];
    profiles.sgl.bandwidth_bps=Math.max(...directions.map(key=>profiles[key].bandwidth_bps));
    for (const k of directions) profiles[k].setup_time_s=profiles.sgl.setup_time_s;
  }
  const compute_configuration={mode,ep_strategy:epStrategy,formations};
  if (mode==='fast_mode') compute_configuration.link_bandwidth_factors=linkBandwidthFactors;
  return {physical_network,compute_configuration};
}

export function loadPhysicalCompute(doc) {
  const spec=doc.physical_network.network;
  const overrides=spec?.cell_link_overrides;
  const derivedProfiles=new Set();
  for(const formation of doc.physical_network.constellations?.compute_formations || []) {
    const settings=doc.compute_configuration?.formations?.[formation.name];
    if(!settings)continue;
    formation.compute_parameters=structuredClone(settings);
    const {template,cells}=formation.compute_parameters;
    removeLegacyCellWorkload(template);
    Object.values(cells||{}).forEach(removeLegacyCellWorkload);
    if(!overrides)continue;
    for(const id of Object.keys(overrides)){
      if(!id.startsWith(formation.name+'-'))continue;
      const key=id.slice(formation.name.length+1);
      if(cellIndex(formation,key)<0 || !(settings.cells?.[key]||settings.template)?.custom_links)continue;
      for(const edge of overrides[id])if(edge.profile_id.startsWith('editor-cell-'))derivedProfiles.add(edge.profile_id);
      delete overrides[id];
    }
  }
  if(overrides){
    for(const links of Object.values(overrides))for(const edge of links)derivedProfiles.delete(edge.profile_id);
    for(const name of derivedProfiles)delete spec.link_profiles[name];
    if(!Object.keys(overrides).length)delete spec.cell_link_overrides;
  }
}

export function deploymentFromPhysical(doc, membersByCell) {
  loadPhysicalCompute(doc);
  const layouts={};
  for(const formation of doc.physical_network.constellations?.compute_formations || []) {
    for(const key of cellKeys(formation)) {
      const id=formation.name+'-'+key;
      if (!membersByCell[id]) continue;
      const layout=cellLayout(formation,key,memberIds(formation,key,cellGeometry(formation,key)));
      if (layout) layouts[id]=layout;
    }
  }
  return layouts;
}
