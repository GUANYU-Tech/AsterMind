import {get, fillOptions} from './studio_common.js';
import {enhanceSelects} from './design-controls.js';
const $ = id => document.getElementById(id);
function updateSelection() {
  const hasModel = !!$('models').value, hasScene = !!$('scenes').value;
  document.querySelector('.configuration-grid').dataset.phase = hasScene ? 'ready' : hasModel ? 'deployment' : 'physical';
  $('editModel').disabled = !$('models').value;
  $('editScene').disabled = !$('scenes').value;
  $('createScene').hidden = !$('models').value;
  $('configurePhysical').hidden = !!$('models').value;
  $('openLive').disabled = !$('scenes').value;
}
Promise.all([get('/physical-catalog'), get('/scene-catalog')]).then(([models, scenes]) => {
  fillOptions($('models'), models); fillOptions($('scenes'), scenes);
  enhanceSelects(document.querySelector('.configuration-grid'));
  $('models').disabled = !models.length; $('scenes').disabled = !scenes.length;
  updateSelection();
}).catch(error => {
  for (const id of ['models', 'scenes']) fillOptions($(id), []);
  $('status').textContent = '读取配置失败：' + error.message + '。请刷新后重试。';
});
for (const id of ['models', 'scenes']) $(id).addEventListener('change', updateSelection);
$('editModel').onclick = () => location.assign('/physical?id=' + encodeURIComponent($('models').value));
$('editScene').onclick = () => location.assign('/scene?id=' + encodeURIComponent($('scenes').value));
$('openLive').onclick = () => location.assign('/live');
