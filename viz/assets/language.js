/* Localize presentation without reloading forms, tasks or the WebGL scene. */
(() => {
  const STORAGE_KEY = 'astermind.ui.language';
  const attrs = ['aria-label', 'aria-description', 'title', 'placeholder', 'alt'];
  const excluded = 'script,style,template,[translate="no"],[data-raw]';
  const skip = excluded + ',textarea,code';
  const hasChinese = /\p{Script=Han}/u;
  const textState = new WeakMap();
  const attributeState = new WeakMap();
  const fileLabels = new WeakMap();
  let language = 'zh-CN';
  let english;
  let requestVersion = 0;

  const switcher = document.createElement('div');
  switcher.className = 'language-switch';
  switcher.setAttribute('role','group');
  switcher.setAttribute('translate','no');
  const buttons = new Map();
  for (const [locale, label, title] of [['zh-CN', '中文', '简体中文'], ['en', 'EN', 'English']]) {
    const button = document.createElement('button');
    button.type = 'button';
    button.lang = locale;
    button.textContent = label;
    button.setAttribute('aria-label', title);
    button.onclick = () => choose(locale);
    buttons.set(locale, button);
    switcher.append(button);
  }

  function localize(value, previous, multiline = false) {
    const source = previous?.shown === value ? previous.source : value;
    if (!hasChinese.test(source)) return;
    const shown = language === 'en' ? english.translate(source,multiline) : source;
    return {source, shown};
  }

  function updateText(node) {
    if (!node.parentElement || node.parentElement.closest(skip)) return;
    const state = localize(node.data,textState.get(node),Boolean(node.parentElement.closest('pre')));
    if (!state) return;
    textState.set(node,state);
    if (node.data !== state.shown) node.data = state.shown;
  }

  function updateAttribute(node, name) {
    if (node.closest('[translate="no"],[data-raw]')) return;
    const value = node.getAttribute(name);
    if (value === null) return;
    let saved = attributeState.get(node);
    const state = localize(value, saved?.get(name));
    if (!state) return;
    if (!saved) attributeState.set(node,saved=new Map());
    saved.set(name,state);
    if (state.shown !== value) node.setAttribute(name,state.shown);
  }

  function scan(node) {
    if (node.nodeType === Node.TEXT_NODE) { updateText(node); return; }
    if (node.nodeType !== Node.ELEMENT_NODE || node.closest(excluded)) return;
    for (const name of attrs) updateAttribute(node,name);
    if (node.matches('textarea,code')) return;
    for (const child of node.childNodes) scan(child);
  }

  function fileInputs(root) {
    if (!(root instanceof Element)) return;
    const selector = 'input[type=file]:not(.import-row input):not([data-raw])';
    const inputs = root.matches(selector) ? [root] : root.querySelectorAll(selector);
    for (const input of inputs) {
      if (fileLabels.has(input)) continue;
      const control = document.createElement('span');
      const action = document.createElement('span');
      const filename = document.createElement('span');
      control.className = 'localized-file';
      action.className = 'file-action';
      filename.className = 'file-name';
      action.textContent = '选择文件';
      action.setAttribute('aria-hidden','true');
      filename.setAttribute('aria-hidden','true');
      const sync = () => {
        const actual = Array.from(input.files, file => file.name).join(', ');
        if (actual) filename.setAttribute('translate','no');
        else filename.removeAttribute('translate');
        filename.textContent = actual || '未选择文件';
      };
      fileLabels.set(input,control);
      input.before(control);
      control.append(action,filename,input);
      input.addEventListener('change',sync);
      sync();
      scan(control);
    }
  }

  let catalogRequest;
  const observer = new MutationObserver(records => {
    for (const record of records) {
      if (record.type === 'childList') for (const node of record.addedNodes) fileInputs(node);
      if (language === 'en') {
        if (record.type === 'characterData') updateText(record.target);
        else if (record.type === 'attributes') updateAttribute(record.target, record.attributeName);
        else for (const node of record.addedNodes) scan(node);
      }
    }
    observer.takeRecords();
  });

  async function choose(locale, persist = true) {
    const version = ++requestVersion;
    if (locale === 'en' && !english) {
      switcher.setAttribute('aria-busy','true');
      try {
        english = await (catalogRequest ??= import('/assets/language-en.js')).catch(error => {
          catalogRequest = undefined;
          throw error;
        });
      } catch (error) {
        console.error('Unable to load the English language catalog',error);
        return;
      } finally {
        if (version === requestVersion) switcher.removeAttribute('aria-busy');
      }
      if (version !== requestVersion) return;
    }
    switcher.removeAttribute('aria-busy');
    language = locale;
    document.documentElement.lang = locale;
    switcher.setAttribute('aria-label',locale === 'en' ? 'Interface language' : '界面语言');
    for (const [id,button] of buttons) button.setAttribute('aria-pressed',String(locale === id));
    if (persist) {
      try { localStorage.setItem(STORAGE_KEY,locale); } catch { /* private storage */ }
    }
    observer.disconnect();
    if (english) scan(document.documentElement);
    observer.observe(document.documentElement,{
      subtree:true,childList:true,characterData:true,attributes:true,attributeFilter:attrs,
    });
  }

  if (new URLSearchParams(location.search).get('embedded') !== '1') {
    const header = document.querySelector('.studio-topbar, .topbar, body > header, main > nav');
    if (header) header.append(switcher);
    else {
      switcher.classList.add('language-switch-floating');
      document.body.prepend(switcher);
    }
  }
  const savedLanguage = () => {
    try { return localStorage.getItem(STORAGE_KEY) === 'en' ? 'en' : 'zh-CN'; }
    catch { return 'zh-CN'; }
  };
  window.addEventListener('storage', event => {
    if (event.key === STORAGE_KEY || event.key === null) choose(savedLanguage(),false);
  });
  window.addEventListener('pageshow', () => {
    if (savedLanguage() !== language) choose(savedLanguage(),false);
  });
  fileInputs(document.body);
  choose('zh-CN',false);
  if (savedLanguage() === 'en') choose('en',false);
})();
