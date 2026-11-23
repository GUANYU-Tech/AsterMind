import {AsterMindScene} from '/cell/scene.js?v=profile-1';

const canvas = document.querySelector('#overviewScene');
try {
  const scene = new AsterMindScene(canvas);
  canvas.closest('figure').setAttribute('data-rendered', '');
  window.addEventListener('pagehide', () => scene.renderer.setAnimationLoop(null));
  window.addEventListener('pageshow', event => {
    if (event.persisted) scene.renderer.setAnimationLoop(() => scene.animate());
  });
} catch (error) {
  console.warn('三维场景无法显示', error);
}
