import {renderConfigForm} from '/cell/config-form.mjs?v=validation-1';
import {matchingShape,shapeForCount} from './formation_size.js';

// Reuse the Cell resource controls while preserving the full scene's grid geometry.
export function editFormation(cell, name, apply, cellId=null) {
  const dialog = document.createElement('dialog');
  dialog.className = 'formation-dialog';
  const form = document.createElement('form');
  const heading = document.createElement('h2');
  heading.textContent = name + ' 编队资源';
  const fields = document.createElement('div');
  const actions = document.createElement('div'); actions.className = 'actions';
  const cancel = document.createElement('button'); cancel.type = 'button'; cancel.textContent = '取消';
  const save = document.createElement('button'); save.type = 'submit'; save.textContent = cellId?'应用到此 Cell':'应用到算力星座';
  actions.append(cancel, save); form.append(heading, fields, actions); dialog.append(form);
  const synthetic = cell.source.type !== 'tle';
  const scoped = cellId && cell.cell_overrides?.[cellId];
  const count = scoped?.satellites_per_cluster ?? cell.cluster.satellites_per_cluster;
  const editableCount = synthetic && cell.cluster.motion_model !== 'reference_fixed_lvlh';
  const draft = {
    satellite_count: count,
    ...(synthetic ? {grid_shape: scoped?.grid_shape ?? shapeForCount(cell.cluster.grid_shape,count),
      spacing_km: (scoped?.formation_pitch_m ?? cell.cluster.formation_pitch_m) / 1000,
      altitude_km: scoped?.altitude_km ?? cell.source.altitude_km} : {}),
    topology: scoped?.topology ?? cell.topology,
  };
  const schema = {properties:{
    satellite_count:{type:'integer', minimum:1, readOnly:!editableCount},
    grid_shape:{type:'string'},
    spacing_km:{type:'number', exclusiveMinimum:0},
    altitude_km:{type:'number', exclusiveMinimum:0},
    topology:{type:'string', enum:['ring','chain','fully_connected']},
  }};
  renderConfigForm(fields, draft, Object.keys(draft), schema, () => {}, {
    labels:{topology:'编队内部连接',satellite_count:'每编队算力卫星数',grid_shape:'编队网格尺寸'},
    onEdit(path,value) {
      if(path[0]==='satellite_count' && synthetic) {
        draft.grid_shape=shapeForCount(draft.grid_shape,value);
        const input=fields.querySelector('#field-grid_shape');if(input)input.value=draft.grid_shape;
      }
    },
  });
  const distance = fields.querySelector('#field-spacing_km');
  const keepoutError = '编队距离必须大于安全间距';
  form.addEventListener('input', () => {
    if (distance?.validationMessage === keepoutError) distance.setCustomValidity('');
  });
  cancel.onclick = () => dialog.close();
  form.onsubmit = event => {
    event.preventDefault();
    if (synthetic && !matchingShape(draft.grid_shape,draft.satellite_count)) {
      const shape=fields.querySelector('#field-grid_shape');
      shape.setCustomValidity('网格行列乘积须等于卫星数');
      shape.reportValidity();return;
    }
    if (synthetic && draft.satellite_count > 1 && draft.spacing_km * 1000 <= (cell.cluster.d_keepout_m || 0)) {
      distance.setCustomValidity(keepoutError);
      distance.reportValidity(); return;
    }
    if(cellId) {
      cell.cell_overrides??={};
      cell.cell_overrides[cellId]=synthetic?{
        satellites_per_cluster:draft.satellite_count,grid_shape:shapeForCount(draft.grid_shape,draft.satellite_count),
        formation_pitch_m:draft.spacing_km*1000,altitude_km:draft.altitude_km,topology:draft.topology,
      }:{topology:draft.topology};
      dialog.close();apply();return;
    }
    cell.cluster.satellites_per_cluster = draft.satellite_count;
    if (synthetic) {
      cell.cluster.grid_shape = shapeForCount(draft.grid_shape, draft.satellite_count);
      cell.cluster.formation_pitch_m = draft.spacing_km * 1000;
      cell.source.altitude_km = draft.altitude_km;
    }
    cell.topology = draft.topology;
    dialog.close(); apply();
  };
  dialog.addEventListener('close', () => dialog.remove(), {once:true});
  document.body.append(dialog); dialog.showModal();
}
