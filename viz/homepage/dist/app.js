'use strict';
const copy = JSON.parse(document.querySelector('#page-copy').textContent);
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
let current = 0;
const tabs = [...document.querySelectorAll('[data-step]')];
const panel = document.querySelector('#journey-panel');
function selectStage(index, focus = false) {
  current = index;
  tabs.forEach((tab, i) => {
    tab.setAttribute('aria-selected', String(i === index));
    tab.tabIndex = i === index ? 0 : -1;
  });
  panel.setAttribute('aria-labelledby', `step-${index}`);
  ['#stage-kicker', '#stage-title', '#stage-body'].forEach((selector, i) => {
    document.querySelector(selector).textContent = copy.stages[index][i];
  });
  document.querySelectorAll('[data-node]').forEach(node => {
    node.classList.toggle('active', Number(node.dataset.node) === index);
  });
  document.querySelector('#next-step').textContent = index === 3 ? copy.restart : copy.next;
  if (focus) tabs[index].focus();
}
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => selectStage(index));
  tab.addEventListener('keydown', event => {
    let next;
    if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
    else if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
    else if (event.key === 'Home') next = 0;
    else if (event.key === 'End') next = tabs.length - 1;
    else return;
    event.preventDefault(); selectStage(next, true);
  });
});
document.querySelector('#next-step').addEventListener('click', () => selectStage((current + 1) % tabs.length));
selectStage(0);
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
