'use strict';
const copy = JSON.parse(document.querySelector('#page-copy').textContent);
const report = document.getElementById('pdf');
const reportLink = document.getElementById('report-link');
function syncReport() {
  const open = location.hash === '#pdf';
  reportLink.setAttribute('aria-expanded', String(open));
  if (open && !report.querySelector('iframe')) {
    report.append(document.getElementById('report-viewer').content.cloneNode(true));
  }
}
window.addEventListener('hashchange', syncReport);
syncReport();
const menu = document.querySelector('#menu');
const nav = document.querySelector('#nav');
function closeMenu() {
  menu.setAttribute('aria-expanded', 'false');
  nav.classList.remove('open');
}
menu.addEventListener('click', () => {
  const open = menu.getAttribute('aria-expanded') !== 'true';
  menu.setAttribute('aria-expanded', String(open));
  nav.classList.toggle('open', open);
});
nav.querySelectorAll('a').forEach(link => link.addEventListener('click', closeMenu));
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && menu.getAttribute('aria-expanded') === 'true') {
    closeMenu(); menu.focus();
  }
});
const journey = document.querySelector('.journey');
const tabs = [...journey.querySelectorAll('[data-step]')];
const nodes = [...journey.querySelectorAll('[data-node]')];
const panel = document.querySelector('#journey-panel');
const playButton = document.querySelector('#play-stages');
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const stageDuration = 7000;
let current = 0;
let playing = !reducedMotion.matches;
let inView = false;
let progress;

function syncPlayback() {
  if (playing && inView && !document.hidden) progress.play();
  else progress.pause();
  playButton.classList.toggle('paused', !playing);
  playButton.title = playing ? copy.pause : copy.play;
  playButton.setAttribute('aria-label', playButton.title);
}

function selectStage(index) {
  if (progress) progress.cancel();
  current = index;
  tabs.forEach((tab, i) => {
    tab.setAttribute('aria-selected', String(i === index));
    tab.tabIndex = i === index ? 0 : -1;
  });
  panel.setAttribute('aria-labelledby', tabs[index].id);
  ['#stage-kicker', '#stage-title', '#stage-body'].forEach((selector, i) => {
    panel.querySelector(selector).textContent = copy.stages[index][i];
  });
  nodes.forEach((node, i) => {
    node.classList.toggle('active', i === index);
    node.setAttribute('aria-pressed', String(i === index));
  });
  progress = tabs[index].querySelector('.stage-track i').animate(
    [{ transform: 'scaleX(0)' }, { transform: 'scaleX(1)' }],
    { duration: stageDuration, fill: 'forwards' }
  );
  progress.onfinish = () => selectStage((current + 1) % tabs.length);
  syncPlayback();
}

tabs.forEach((tab, index) => {
  function chooseStage() {
    playing = false;
    selectStage(index);
  }
  nodes[index].addEventListener('click', chooseStage);
  tab.addEventListener('click', chooseStage);
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else return;
    event.preventDefault();
    tabs[next].focus();
    tabs[next].click();
  });
});
playButton.addEventListener('click', () => {
  playing = !playing;
  syncPlayback();
});
document.querySelector('#restart-stages').addEventListener('click', () => {
  playing = true;
  selectStage(0);
});
selectStage(0);
journey.querySelector('.journey-controls').hidden = false;
new IntersectionObserver(([entry]) => {
  inView = entry.isIntersecting;
  syncPlayback();
}).observe(journey);
document.addEventListener('visibilitychange', syncPlayback);
reducedMotion.addEventListener('change', event => {
  playing = !event.matches;
  syncPlayback();
});
let copyReset;
document.querySelector('#copy-code').addEventListener('click', async () => {
  const button = document.querySelector('#copy-code');
  const status = document.querySelector('#copy-status');
  clearTimeout(copyReset);
  try {
    await navigator.clipboard.writeText(document.querySelector('#setup-code').textContent);
    button.textContent = copy.copied;
    status.textContent = copy.copystatus;
    copyReset = setTimeout(() => { button.textContent = copy.copy; status.textContent = ''; }, 2200);
  } catch {
    button.textContent = copy.copyfail;
    status.textContent = copy.copyfailstatus;
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#setup-code'));
    const selection = window.getSelection();
    selection.removeAllRanges(); selection.addRange(range);
  }
});
