import {cellCount,cellIndex,cellGeometry,cellKeys,cellLayout,memberIds,applyCell,bindLayout} from './physical-cells.js';

import {formationNames} from '../cell/config-labels.mjs';

const pages=new WeakMap();
const el=(tag,text)=>{const node=document.createElement(tag);if(text!==undefined)node.textContent=text;return node};
function button(text,click){const node=el('button',text);node.type='button';node.onclick=click;return node}

function editCell(formation,key,capabilities,scene,render) {
  const id=formation.name+'-'+key,geometry=cellGeometry(formation,key);
  const layout=cellLayout(formation,key,memberIds(formation,key,geometry));
  const dialog=el('dialog');dialog.className='cell-workspace-dialog';
  dialog.setAttribute('aria-label','编辑算力编队 '+id);
  const header=el('header');const title=el('strong',formation.compute_parameters?.cell_names?.[key] || id);
  title.title=id;
  const cancel=button('取消',()=>dialog.close());header.append(title,cancel);
  const frame=el('iframe');frame.title='算力编队编辑器';
  const footer=el('footer');const status=el('span','正在打开编队编辑器…');status.setAttribute('role','status');
  let submit;
  async function commit(all){
    apply.disabled=template.disabled=true;
    frame.inert=true;
    try{await submit(all)}catch(e){status.textContent=e.message}
    finally{frame.inert=false;apply.disabled=template.disabled=false}
  }
  const apply=button('应用到此 Cell',()=>commit(false));apply.className='primary';
  const count=cellCount(formation);
  const template=button('设为模板并应用到全部 '+count+' 个 Cell',()=>commit(true));
  apply.disabled=template.disabled=true;footer.append(status,template,apply);
  frame.formationSession={formation:structuredClone(formation),key,id,geometry,layout,capabilities,...scene,
    explicitPairs:scene.cellLinkOverrides?.[id]?.map(edge=>[edge.source,edge.target]),
    status:message=>status.textContent=message,
    ready:callback=>{submit=callback;apply.disabled=template.disabled=false;status.textContent=''},
    apply:(geometry,layout,all)=>{
      if (all && formation.source.type==='tle') {
        for (const other of cellKeys(formation)) bindLayout(layout,memberIds(formation,other,cellGeometry(formation,other)));
      }
      if (!dialog.open) return;
      if(scene.cellLinkOverrides){
        const keys=all?cellKeys(formation):[key];
        for(const memberKey of keys)delete scene.cellLinkOverrides[formation.name+'-'+memberKey];
      }
      applyCell(formation,key,geometry,layout,all);
      dialog.close();render();
    },
  };
  frame.src='/cell/?context=physical';
  dialog.append(header,frame,footer);
  dialog.addEventListener('close',()=>dialog.remove(),{once:true});
  document.body.append(dialog);dialog.showModal();
}

export function showCellList(host,formation,capabilities,mode,render,focusId) {
  const count=cellCount(formation), size=12;
  const focus=focusId?.startsWith(formation.name+'-')?cellIndex(formation,focusId.slice(formation.name.length+1)):-1;
  const page=Math.max(0,Math.min(Math.ceil(count/size)-1,focus>=0?Math.floor(focus/size):pages.get(formation)||0));
  host.replaceChildren(el('h4','算力编队 · '+count+' 个 Cell'));
  const list=el('div');list.className='formation-cells';host.append(list);
  for(const [i,key] of cellKeys(formation,page*size,(page+1)*size).entries()){
    const id=formation.name+'-'+key, geometry=cellGeometry(formation,key);
    const settings=formation.compute_parameters;
    const layout=settings?.cells?.[key] || settings?.template;
    const count=formation.source.type==='tle'?formation.source.cell_members[key].length:geometry.satellites_per_cluster;
    const row=el('div');row.className='formation-cell-row';
    const identity=el('div'), name=el('input');
    name.type='text';name.placeholder='Cell '+(page*size+i+1);
    name.value=settings?.cell_names?.[key] || '';
    name.setAttribute('aria-label','编队名称');name.title=id;
    const machineId=el('small',id);
    machineId.hidden=Boolean(name.value);
    name.onchange=()=>{
      const params=formation.compute_parameters??={cells:{}};
      const names=params.cell_names??={};
      name.value=name.value.trim();
      if(name.value)names[key]=name.value;else delete names[key];
      machineId.hidden=Boolean(name.value);
      document.dispatchEvent(new Event('modelchange'));
    };
    identity.append(name,machineId);
    const hardware=layout?.deployment.hardware_id || capabilities.default_configuration.hardware.default_hardware_id;
    const shape=formation.source.type==='tle'?'TLE':geometry.motion_model==='static_formation'
      ? formationNames[geometry.topology] : geometry.motion_model==='reference_fixed_lvlh'?'自定义相对位置':(geometry.grid_shape || count+'x1')+' 网格';
    const summary=el('span');
    const parts=[count+' 颗卫星',capabilities.su_profiles[hardware]?.name || hardware,(layout?.deployment.gpus_per_satellite??1)+' 卡/星',
      shape,layout?.deployment.model?.split('/').at(-1)?.replace(/\.json$/,''),
      settings?.cells?.[key] || formation.cell_overrides?.[id]?'单独配置':settings?.template?'星座模板':'默认配置'].filter(Boolean);
    parts.forEach((part,index)=>summary.append(index ? ' · ' : '',el('span',part)));
    const edit=button('编辑编队',()=>{try{editCell(formation,key,capabilities,mode(),render)}catch(e){document.getElementById('status').textContent=e.message}});
    edit.setAttribute('aria-label','编辑编队 '+id);
    row.append(identity,summary,edit);list.append(row);
  }
  if(count>size){
    const pager=el('div');pager.className='actions';
    const step=delta=>{pages.set(formation,page+delta);showCellList(host,formation,capabilities,mode,render)};
    const back=button('上一页',()=>step(-1)),next=button('下一页',()=>step(1));
    back.disabled=page===0;next.disabled=(page+1)*size>=count;
    pager.append(back,el('span',(page+1)+' / '+Math.ceil(count/size)),next);host.append(pager);
  }
}
