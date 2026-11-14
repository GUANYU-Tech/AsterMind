// Share navigation and appearance without touching page simulation state.
// Recording has ended. Clear its page-only pointer modes on every visit.
for (const key of ['orbitserve.webCapture', 'orbitserve.webCapturePosition',
                   'orbitserve.webCaptureEvents', 'orbitserve.demoPointer']) {
  sessionStorage.removeItem(key);
}
document.documentElement.classList.remove('web-capture-mode');
document.querySelectorAll('.web-capture-pointer,.web-capture-click,.web-capture-badge,'+
                          '.demo-pointer,.demo-click,.demo-pointer-label')
  .forEach(element => element.remove());
if (!document.querySelector('[data-studio-theme]')) {
  const theme=document.createElement('link');
  theme.rel='stylesheet';theme.href='/assets/workspace.css';theme.dataset.studioTheme='';document.head.append(theme);
  document.body.classList.add('studio-shell');
  const links=[['01','场景概览','/workspace'],['02','实验配置','/configuration'],['03','在线仿真','/live'],['04','实验结果','/results'],['05','运行记录','/experiments']];
  const shell=document.createElement('div');
  shell.innerHTML=`<header class="studio-topbar"><a class="studio-logo" href="/"><span class="studio-mark" aria-hidden="true">✧</span>Astermind <em>星鉴</em></a><span>仿真任务工作台</span><a class="studio-local" href="/">切换仿真工作区 ↗</a></header><aside class="studio-sidebar" aria-label="工作台导航"><div class="studio-nav-label">完整场景仿真</div>${links.map(([n,label,href])=>`<a href="${href}"${location.pathname===href.split('?')[0]?' aria-current="page"':''}><b aria-hidden="true">${n}</b>${label}</a>`).join('')}<div class="studio-sidefoot"><a href="https://qcnaaemidt0k.feishu.cn/wiki/CwKmwM1VhiOD19kbv8UcNM5RnTq" target="_blank" rel="noopener noreferrer">技术文档 ↗</a></div></aside>`;
  document.body.prepend(...shell.childNodes);
}

// Keep ordinary workspace navigation in its current browser tab, including
// dynamically rendered task links. Modified clicks retain browser behavior.
document.addEventListener('click', event => {
  if (event.defaultPrevented || event.button !== 0 || event.ctrlKey ||
      event.metaKey || event.shiftKey || event.altKey) return;
  const link = event.target instanceof Element ? event.target.closest('a[href]') : null;
  if (!link || link.hasAttribute('download') || link.hasAttribute('data-view')) return;
  const destination = new URL(link.href, location.href);
  if(destination.pathname===location.pathname&&destination.search===location.search&&destination.hash)return;
  if (destination.origin !== location.origin ||
      !['http:', 'https:'].includes(destination.protocol)) return;
  event.preventDefault();
  location.assign(destination.href);
}, {capture: true});
