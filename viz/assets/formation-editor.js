import {renderConfigForm} from '/cell/config-form.mjs?v=validation-1';
import {shapeForCount} from './formation_size.js';

// Reuse the Cell resource controls while preserving the full scene's grid geometry.
export function editFormation(cell, name, apply) {
  const dialog = document.createElement('dialog');
  dialog.className = 'formation-dialog';
  const form = document.createElement('form');
  const heading = document.createElement('h2');
  heading.textContent = name + ' 编队资源';
  const fields = document.createElement('div');
  const actions = document.createElement('div'); actions.className = 'actions';
  const cancel = document.createElement('button'); cancel.type = 'button'; cancel.textContent = '取消';
  const save = document.createElement('button'); save.type = 'submit'; save.textContent = '应用到算力星座';
  actions.append(cancel, save); form.append(heading, fields, actions); dialog.append(form);
  const synthetic = cell.source.type !== 'tle';
  const draft = {
    satellite_count: cell.cluster.satellites_per_cluster,
    ...(synthetic ? {spacing_km: cell.cluster.formation_pitch_m / 1000, altitude_km: cell.source.altitude_km} : {}),
    topology: cell.topology,
  };
  const schema = {properties:{
    satellite_count:{type:'integer', minimum:1},
    spacing_km:{type:'number', exclusiveMinimum:0},
    altitude_km:{type:'number', exclusiveMinimum:0},
    topology:{type:'string', enum:['ring','fully_connected']},
  }};
  renderConfigForm(fields, draft, Object.keys(draft), schema, () => {}, {
    labels:{topology:'编队内部连接',satellite_count:'每编队算力卫星数'},
  });
  const distance = fields.querySelector('#field-spacing_km');
  const keepoutError = '编队距离必须大于安全间距';
  form.addEventListener('input', () => {
    if (distance?.validationMessage === keepoutError) distance.setCustomValidity('');
  });
  cancel.onclick = () => dialog.close();
  form.onsubmit = event => {
    event.preventDefault();
    if (synthetic && draft.satellite_count > 1 && draft.spacing_km * 1000 <= (cell.cluster.d_keepout_m || 0)) {
      distance.setCustomValidity(keepoutError);
      distance.reportValidity(); return;
    }
    cell.cluster.satellites_per_cluster = draft.satellite_count;
    if (synthetic) {
      cell.cluster.grid_shape = shapeForCount(cell.cluster.grid_shape, draft.satellite_count);
      cell.cluster.formation_pitch_m = draft.spacing_km * 1000;
      cell.source.altitude_km = draft.altitude_km;
    }
    cell.topology = draft.topology;
    dialog.close(); apply();
  };
  dialog.addEventListener('close', () => dialog.remove(), {once:true});
  document.body.append(dialog); dialog.showModal();
}
