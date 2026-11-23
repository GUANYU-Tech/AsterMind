// Visualize actual page input. This module never sends input to the OS or page.
const params = new URL(location.href).searchParams;
if (params.get('capture') === '1') sessionStorage.setItem('orbitserve.webCapture', '1');
if (params.get('capture') === '0') sessionStorage.removeItem('orbitserve.webCapture');
if (sessionStorage.getItem('orbitserve.webCapture') === '1') {
  document.documentElement.classList.add('web-capture-mode');
  const style = document.createElement('style');
  style.textContent = `
    html.web-capture-mode,html.web-capture-mode body{width:2688px!important;min-width:2688px!important}
    html.web-capture-mode .studio-topbar{width:2688px!important}
    html.web-capture-mode .studio-sidebar{height:1280px;bottom:auto!important}
    html.web-capture-mode .studio-topbar{height:92px;padding:0 39px;gap:28px}
    html.web-capture-mode .studio-logo{font-size:23px;gap:17px}
    html.web-capture-mode .studio-topbar>span{font-size:15px}
    html.web-capture-mode .studio-sidebar{inset:92px auto 0 0;width:286px;padding:42px 20px 28px;gap:7px}
    html.web-capture-mode .studio-sidebar a{font-size:17px;padding:17px 20px;gap:17px}
    html.web-capture-mode .studio-sidebar a b{font-size:18px;width:24px}
    html.web-capture-mode .studio-nav-label{font-size:14px;padding:0 20px;margin:21px 0 11px}
    html.web-capture-mode .studio-sidefoot{font-size:14px;padding-top:25px}
    html.web-capture-mode .studio-sidefoot strong{font-size:15px}
    html.web-capture-mode.studio-shell main{width:2402px!important;max-width:none;margin:92px 0 0 286px;padding:39px 48px 50px}
    html.web-capture-mode main{font-size:18px}
    html.web-capture-mode h1{font-size:42px!important}
    html.web-capture-mode h2{font-size:25px!important}
    html.web-capture-mode h3{font-size:21px!important}
    html.web-capture-mode .hero h2{font-size:58px!important;line-height:1.45}
    html.web-capture-mode .hero{min-height:480px;padding:55px}
    html.web-capture-mode p,html.web-capture-mode label,html.web-capture-mode .hint{font-size:17px!important}
    html.web-capture-mode button,html.web-capture-mode select,html.web-capture-mode input{font-size:18px!important;padding:14px 18px!important}
    html.web-capture-mode input[type=checkbox]{padding:0!important;width:20px;height:20px}
    html.web-capture-mode .panel{padding:30px;border-radius:14px}
    html.web-capture-mode .workspace-grid{grid-template-columns:minmax(0,1fr) 448px}
    html.web-capture-mode #globe{height:784px}
    html.web-capture-mode .clock strong{font-size:43px}
    html.web-capture-mode .metrics strong{font-size:38px}
    html.web-capture-mode .metrics small,html.web-capture-mode .metrics span{font-size:15px}
    html.web-capture-mode .notice,html.web-capture-mode .event-detail,html.web-capture-mode table{font-size:17px}
    .web-capture-pointer{position:fixed;left:0;top:0;width:26px;height:34px;z-index:2147483647;pointer-events:none;filter:drop-shadow(0 2px 3px #000b);transition:transform .3s ease-out}
    .web-capture-pointer svg{width:26px;height:34px}
    .web-capture-click{position:fixed;width:46px;height:46px;margin:-23px;border:3px solid #a3ffe0;border-radius:50%;z-index:2147483646;pointer-events:none;box-shadow:0 0 0 7px #a3ffe025;animation:web-capture-click 1.8s both}
    .web-capture-badge{position:fixed;right:18px;bottom:12px;z-index:2147483645;background:#0b191eea;border:1px solid #52776b;border-radius:6px;padding:6px 11px;color:#d0f2e4;font:12px 'Microsoft YaHei',sans-serif;pointer-events:none}
    html.web-capture-mode .web-capture-badge{display:none}
    @keyframes web-capture-click{0%,45%{opacity:1;transform:scale(.72)}100%{opacity:0;transform:scale(1.65)}}`;
  document.head.append(style);
  const cursor = document.createElement('div');
  cursor.className = 'web-capture-pointer';
  cursor.setAttribute('aria-hidden', 'true');
  cursor.innerHTML = '<svg viewBox="0 0 30 38"><path d="M3 2 L3 29 L10 23 L16 35 L21 32 L15 21 L25 20 Z" fill="#f2fff8" stroke="#0d2920" stroke-width="2"/><path d="M5 5 L5 24 L10 19 L17 21 Z" fill="#7cd8c0"/></svg>';
  const badge = document.createElement('span');
  badge.className = 'web-capture-badge';
  badge.textContent = '网页交互录制 等待点击';
  const log = document.createElement('script');
  log.id = 'web-capture-events';
  log.type = 'application/json';
  let events = JSON.parse(sessionStorage.getItem('orbitserve.webCaptureEvents') || '[]');
  log.textContent = JSON.stringify(events);
  document.body.append(cursor, badge, log);
  let position = JSON.parse(sessionStorage.getItem('orbitserve.webCapturePosition') || '[300,180]');
  const zoom = parseFloat(getComputedStyle(document.documentElement).zoom) || 1;
  cursor.style.transform = `translate(${position[0]/zoom}px,${position[1]/zoom}px)`;
  function move(event) {
    position = [event.clientX, event.clientY];
    cursor.style.transform = `translate(${position[0]/zoom}px,${position[1]/zoom}px)`;
  }
  function record(event) {
    const target = event.target instanceof Element ? event.target.closest('button,a,select,input,summary,canvas') || event.target : null;
    const name = (target?.getAttribute('aria-label') || target?.getAttribute('title') || target?.textContent || target?.id || event.type).trim().replace(/\s+/g, ' ').slice(0, 90);
    const entry = {time: Date.now(), type:event.type, x:position[0], y:position[1], target:target?.id || '', name, path:location.pathname, trusted:event.isTrusted, zoom, dpr:devicePixelRatio, scrollY, rect:target?.getBoundingClientRect().toJSON()};
    if (event.type === 'change' && target?.tagName === 'SELECT') entry.selected = target.selectedOptions[0]?.textContent;
    events.push(entry);
    log.textContent = JSON.stringify(events);
    sessionStorage.setItem('orbitserve.webCaptureEvents', log.textContent);
    sessionStorage.setItem('orbitserve.webCapturePosition', JSON.stringify(position));
    badge.textContent = `网页交互录制 ${event.type === 'change' ? '选择' : '点击'} ${name.slice(0, 24)}`;
  }
  document.addEventListener('pointermove', move, {passive:true,capture:true});
  document.addEventListener('pointerdown', event => {
    move(event);
    const ring = document.createElement('span');
    ring.className = 'web-capture-click';
    ring.style.left = event.clientX/zoom + 'px';
    ring.style.top = event.clientY/zoom + 'px';
    ring.setAttribute('aria-hidden', 'true');
    document.body.append(ring);
    setTimeout(() => ring.remove(), 1900);
    record(event);
  }, {passive:true,capture:true});
  document.addEventListener('change', record, {passive:true,capture:true});
}
