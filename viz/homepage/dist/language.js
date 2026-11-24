/* The homepage and every simulation view share the same origin-wide preference. */
(() => {
  const key = 'astermind.language';
  const chinese = 'zh-CN';
  const english = 'en';
  const valid = value => value === chinese || value === english;
  function preference() {
    try { return localStorage.getItem(key) ?? localStorage.getItem('astermind.ui.language'); } catch { return null; }
  }
  function remember(value) {
    try { localStorage.setItem(key, value); } catch { /* Storage may be disabled. */ }
  }

  const selector = document.getElementById('language');
  const original = document.documentElement.lang;
  let current = valid(preference()) ? preference() : chinese;
  const url = new URL(location.href);
  const requested = url.searchParams.get('lang');
  if (valid(requested)) {
    current = requested;
    remember(current);
    url.searchParams.delete('lang');
    history.replaceState(history.state, '', url);
  } else if (original === english) {
    current = english;
    remember(current);
  }

  function homeURL(lang) {
    const link = new URL(selector.href, location.href);
    link.search = location.search;
    link.hash = location.hash;
    link.searchParams.set('lang', lang);
    return link;
  }

  function synchronize(value) {
    current = valid(value) ? value : chinese;
    if (current !== original) location.replace(homeURL(current));
  }

  window.addEventListener('storage', event => {
    if (event.key === key || event.key === null) synchronize(preference());
  });
  window.addEventListener('pageshow', event => {
    if (event.persisted) synchronize(preference());
  });

  // An explicit choice also applies when the link opens in a new tab.
  const target = original === english ? chinese : english;
  const updateLink = () => { selector.href = homeURL(target).href; };
  updateLink();
  window.addEventListener('hashchange', updateLink);
  synchronize(current);
})();
