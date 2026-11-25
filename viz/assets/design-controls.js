// Presentation adapters: the original form inputs and their handlers remain authoritative.
let fieldSequence = 0;

export function enhanceSelects(root) {
  if (!globalThis.CSS?.supports?.('appearance', 'base-select')) return;
  for (const select of root.querySelectorAll('select')) {
    if (select.querySelector('selectedcontent')) continue;
    const button = document.createElement('button'); button.type = 'button'; button.tabIndex = -1; button.className = 'design-select-button';
    button.append(document.createElement('selectedcontent')); select.prepend(button);
  }
}

function captionFor(input) {
  return input.getAttribute('aria-label') || input.closest('label')?.firstChild?.textContent?.trim() || '参数';
}

function fieldContainer(input, caption) {
  const label = input.closest('label');
  if (!label || label.querySelectorAll('input,select,textarea').length !== 1) return null;
  const wrap = document.createElement('div'); wrap.className = 'design-field';
  const title = document.createElement('label'); title.textContent = caption;
  input.id ||= `physical-field-${++fieldSequence}`; title.htmlFor = input.id;
  label.replaceWith(wrap); wrap.append(title, input);
  return {wrap, title};
}

export function enhancePhysicalFields(root) {
  enhanceSelects(root);
  for (const input of root.querySelectorAll('input[type=checkbox]:not([data-design-control]):not(#linkContention)')) {
    const caption = captionFor(input), container = fieldContainer(input, caption);
    if (!container) continue;
    const {wrap, title} = container; input.dataset.designControl = 'toggle';
    input.classList.add('design-source'); input.tabIndex = -1; input.setAttribute('aria-hidden', 'true');
    title.id = `${input.id}-caption`; title.removeAttribute('for');
    const group = document.createElement('div'); group.className = 'design-segments';
    group.setAttribute('role', 'group'); group.setAttribute('aria-labelledby', title.id);
    for (const [value, text] of [[true, '打开'], [false, '关闭']]) {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = text;
      button.disabled = input.disabled; button.setAttribute('aria-pressed', String(input.checked === value));
      button.onclick = () => {
        input.checked = value;
        input.dispatchEvent(new Event('input', {bubbles:true}));
        input.dispatchEvent(new Event('change', {bubbles:true}));
        for (const item of group.children) item.setAttribute('aria-pressed', String(item === button));
      };
      group.append(button);
    }
    wrap.append(group);
  }
  for (const input of root.querySelectorAll('.fields input[type=number]:not([data-design-control])')) {
    if (input.closest('#time')) continue;
    const caption = captionFor(input), container = fieldContainer(input, caption);
    if (!container) continue;
    input.dataset.designControl = 'range';
    const row = document.createElement('div'); row.className = 'design-number-range';
    const slider = document.createElement('input'); slider.type = 'range';
    slider.setAttribute('aria-label', `${caption}滑块`);
    const integer = /轨道面数|每面.*数|每 cell 卫星数|终端数|关联总上限|用户节点上限|信关站上限/.test(caption);
    const angle = /°|相位/.test(caption);
    const refresh = () => {
      const value = input.value === '' ? 0 : Number(input.value);
      if (!Number.isFinite(value)) return;
      // Slider bounds are a convenient editing window, not extra model constraints.
      slider.min = input.min || String(angle ? Math.min(-180,value) : Math.min(0,value));
      slider.max = input.max || String(Math.max(angle ? 360 : /高度 km/.test(caption) ? 1200 : integer ? 160 : 10, value * 2));
      slider.step = input.step && input.step !== 'any' ? input.step : integer ? '1' : 'any';
      slider.value = String(value); slider.disabled = input.disabled;
      paint();
    };
    const paint = () => {
      const span = Number(slider.max)-Number(slider.min);
      slider.style.setProperty('--range-progress', `${span ? (Number(slider.value)-Number(slider.min))/span*100 : 0}%`);
    };
    slider.addEventListener('input', () => {
      input.value = String(Number(Number(slider.value).toPrecision(7))); paint();
      input.dispatchEvent(new Event('input', {bubbles:true}));
    });
    slider.addEventListener('change', () => input.dispatchEvent(new Event('change', {bubbles:true})));
    input.addEventListener('change', refresh);
    input.addEventListener('design-value-change', refresh);
    input.addEventListener('input', () => { if (document.activeElement !== slider) refresh(); });
    row.append(input, slider); container.wrap.append(row); refresh();
  }
}
