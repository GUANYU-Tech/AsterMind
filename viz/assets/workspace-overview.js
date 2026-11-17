import * as THREE from './three.module.js';

// An illustration only: no simulation, task polling or editable scene state.
const container = document.getElementById('overviewScene');
const figure = container.closest('figure');
let renderer;
try {
  renderer = new THREE.WebGLRenderer({antialias: true, alpha: true});
} catch {
  // The SVG remains visible on devices without WebGL.
}

if (renderer) {
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.3;
  container.append(renderer.domElement);
  const scene = new THREE.Scene();
  const camera = new THREE.OrthographicCamera(-5, 5, 3.6, -3.6, .1, 30);
  camera.position.set(0, 0, 12);
  const key = new THREE.DirectionalLight(0xf5ffe9, 3.1);
  key.position.set(-5, 6, 9);
  scene.add(key, new THREE.AmbientLight(0xb6d0d1, 1.1));

  const earthMaterial = new THREE.MeshStandardMaterial({color: 0xc5d8cc, roughness: .93});
  const earth = new THREE.Mesh(new THREE.SphereGeometry(2, 80, 56), earthMaterial);
  earth.rotation.y = 2.2;
  const earthSystem = new THREE.Group();
  earthSystem.add(earth);
  scene.add(earthSystem);
  const atmosphere = new THREE.Mesh(new THREE.SphereGeometry(2.035, 64, 48), new THREE.ShaderMaterial({
    vertexShader: 'varying vec3 n;varying vec3 v;void main(){vec4 p=modelViewMatrix*vec4(position,1.0);n=normalize(normalMatrix*normal);v=normalize(-p.xyz);gl_Position=projectionMatrix*p;}',
    fragmentShader: 'varying vec3 n;varying vec3 v;void main(){float a=pow(1.0-max(dot(normalize(n),normalize(v)),0.0),4.0);gl_FragColor=vec4(0.43,0.69,0.65,a*.42);}',
    transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
  }));
  scene.add(atmosphere);

  const bodyMaterial = new THREE.MeshStandardMaterial({color: 0xe1e6d5, metalness: .3, roughness: .55});
  const panelMaterial = new THREE.MeshStandardMaterial({color: 0x387b87, metalness: .15, roughness: .7, emissive: 0x173c45, emissiveIntensity: .6});
  const panelGridMaterial = new THREE.LineBasicMaterial({color: 0xa9d2cf, transparent: true, opacity: .6});
  function satellite(position, angle) {
    const model = new THREE.Group();
    model.add(new THREE.Mesh(new THREE.BoxGeometry(.17, .21, .17), bodyMaterial));
    for (const side of [-1, 1]) {
      const wing = new THREE.Mesh(new THREE.BoxGeometry(.3, .17, .025), panelMaterial);
      wing.position.x = side * .27;
      model.add(wing);
      const points = [];
      for (let x = -.15; x <= .151; x += .1) points.push(new THREE.Vector3(side * .27 + x, -.085, .016), new THREE.Vector3(side * .27 + x, .085, .016));
      points.push(new THREE.Vector3(side * .27 - .15, 0, .016), new THREE.Vector3(side * .27 + .15, 0, .016));
      model.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(points), panelGridMaterial));
    }
    model.position.copy(position);
    model.rotation.set(.18, -.2, angle);
    scene.add(model);
    return model;
  }
  const satellites = [];
  for (const [radius, tilt, roll, period, phases] of [
    [3.13, .95, -.34, 75, [.1, 1.3, 2.7, 4.1, 5.5]],
    [3.04, -.55, 1.06, 95, [.2, 3.2, 5.7]],
  ]) {
    const rotation = new THREE.Euler(tilt, 0, roll, 'ZXY');
    const at = phase => new THREE.Vector3(Math.cos(phase) * radius, Math.sin(phase) * radius, 0).applyEuler(rotation);
    const orbit = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(Array.from({length: 192}, (_, i) => at(i / 192 * Math.PI * 2))), new THREE.LineBasicMaterial({color: 0x88b3a2, transparent: true, opacity: .48}));
    scene.add(orbit);
    for (const phase of phases) satellites.push({model: satellite(at(phase), roll), phase, radius, rotation, speed: Math.PI * 2 / period});
  }

  const stationMaterial = new THREE.MeshStandardMaterial({color: 0xd5e5aa, metalness: .15, roughness: .7, side: THREE.DoubleSide});
  const groundMaterial = new THREE.MeshBasicMaterial({color: 0xc5e78b});
  const stations = [];
  for (const direction of [new THREE.Vector3(-.46, .55, .7), new THREE.Vector3(-.33, -.43, .84)]) {
    const normal = direction.normalize();
    const station = new THREE.Group();
    station.position.copy(normal).multiplyScalar(2.025);
    station.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), normal);
    const footprint = new THREE.Mesh(new THREE.RingGeometry(.16, .18, 40), groundMaterial);
    footprint.rotation.x = -Math.PI / 2;
    station.add(footprint);
    const base = new THREE.Mesh(new THREE.CylinderGeometry(.065, .11, .13, 12), stationMaterial);
    base.position.y = .065;
    station.add(base);
    const dish = new THREE.Mesh(new THREE.SphereGeometry(.16, 24, 14, 0, Math.PI * 2, Math.PI * .58, Math.PI * .42), stationMaterial);
    dish.position.y = .27;
    station.add(dish);
    const feed = new THREE.Mesh(new THREE.CylinderGeometry(.009, .009, .18, 6), stationMaterial);
    feed.position.y = .27;
    station.add(feed);
    earthSystem.add(station);
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(6), 3));
    geometry.setAttribute('lineDistance', new THREE.Float32BufferAttribute([0, 0], 1));
    const link = new THREE.Line(geometry, new THREE.LineDashedMaterial({color: 0xc5e78b, dashSize: .06, gapSize: .045, transparent: true, opacity: .8}));
    link.frustumCulled = false;
    scene.add(link);
    stations.push({normal, link});
  }

  const up = new THREE.Vector3(0, 1, 0);
  const worldNormal = new THREE.Vector3(), start = new THREE.Vector3(), delta = new THREE.Vector3();
  function updateMotion(seconds) {
    // Keep rotation perceptible in the smaller, secondary illustration.
    earthSystem.rotation.y = seconds * Math.PI * 2 / 120;
    for (const satellite of satellites) {
      const phase = satellite.phase + seconds * satellite.speed;
      satellite.model.position.set(Math.cos(phase) * satellite.radius, Math.sin(phase) * satellite.radius, 0).applyEuler(satellite.rotation);
    }
    for (const {normal, link} of stations) {
      worldNormal.copy(normal).applyAxisAngle(up, earthSystem.rotation.y);
      start.copy(worldNormal).multiplyScalar(2.325);
      let nearest = null, distance = Infinity;
      for (const {model} of satellites) {
        delta.copy(model.position).sub(start);
        const squared = delta.lengthSq();
        if (delta.dot(worldNormal) > 0 && squared < distance) {
          nearest = model.position;
          distance = squared;
        }
      }
      // Reconnect as satellites pass overhead; never draw through the Earth.
      link.visible = nearest !== null;
      if (!nearest) continue;
      const positions = link.geometry.getAttribute('position');
      positions.setXYZ(0, start.x, start.y, start.z);
      positions.setXYZ(1, nearest.x, nearest.y, nearest.z);
      positions.needsUpdate = true;
      const distances = link.geometry.getAttribute('lineDistance');
      distances.setX(1, Math.sqrt(distance));
      distances.needsUpdate = true;
    }
  }
  updateMotion(0);

  let disposed = false;
  let textureReady = false;
  const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
  const motionButton = document.getElementById('overviewMotion');
  let paused = motionPreference.matches;
  let frame = 0, previousTime = null, elapsed = 0;
  function animate(time) {
    frame = 0;
    if (disposed || paused || document.hidden || renderer.getContext().isContextLost()) return;
    elapsed += previousTime === null ? 0 : Math.min((time - previousTime) / 1000, .1);
    previousTime = time;
    updateMotion(elapsed);
    render();
    frame = requestAnimationFrame(animate);
  }
  function syncMotion() {
    cancelAnimationFrame(frame);
    frame = 0;
    previousTime = null;
    motionButton.textContent = paused ? '播放动画' : '暂停动画';
    if (!disposed && !paused && !document.hidden && textureReady && !renderer.getContext().isContextLost()) frame = requestAnimationFrame(animate);
  }
  function preferenceChanged() {
    paused = motionPreference.matches;
    syncMotion();
  }
  motionButton.addEventListener('click', () => {
    paused = !paused;
    syncMotion();
  });
  motionPreference.addEventListener('change', preferenceChanged);
  document.addEventListener('visibilitychange', syncMotion);
  window.addEventListener('pageshow', syncMotion);
  function render() {
    if (disposed || renderer.getContext().isContextLost()) return;
    renderer.render(scene, camera);
    if (textureReady && !figure.hasAttribute('data-rendered')) figure.setAttribute('data-rendered', '');
  }
  const texture = new THREE.TextureLoader().load('/assets/earth.jpg', () => {
    if (disposed) return;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
    earthMaterial.map = texture;
    earthMaterial.needsUpdate = true;
    textureReady = true;
    motionButton.hidden = false;
    render();
    syncMotion();
  }, undefined, () => { /* Keep the illustrated fallback if the texture fails. */ });
  function resize() {
    const {width, height} = container.getBoundingClientRect();
    if (!width || !height || disposed) return;
    // Fit the complete constellation, including solar panels, on narrow screens.
    const aspect = width / height;
    const halfHeight = Math.max(3.55, 3.85 / aspect);
    camera.left = -halfHeight * aspect;
    camera.right = halfHeight * aspect;
    camera.top = halfHeight;
    camera.bottom = -halfHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height);
    render();
  }
  const observer = new ResizeObserver(resize);
  observer.observe(container);
  resize();
  renderer.domElement.addEventListener('webglcontextlost', event => {
    event.preventDefault();
    figure.removeAttribute('data-rendered');
    motionButton.hidden = true;
    syncMotion();
  });
  renderer.domElement.addEventListener('webglcontextrestored', () => {
    motionButton.hidden = !textureReady;
    render();
    syncMotion();
  });
  window.addEventListener('pagehide', event => {
    cancelAnimationFrame(frame);
    frame = 0;
    previousTime = null;
    if (event.persisted) return;
    disposed = true;
    document.removeEventListener('visibilitychange', syncMotion);
    window.removeEventListener('pageshow', syncMotion);
    motionPreference.removeEventListener('change', preferenceChanged);
    observer.disconnect();
    const geometries = new Set(), materials = new Set();
    scene.traverse(object => {
      if (object.geometry) geometries.add(object.geometry);
      if (object.material) materials.add(object.material);
    });
    geometries.forEach(geometry => geometry.dispose());
    materials.forEach(material => material.dispose());
    texture.dispose();
    renderer.dispose();
  });
}
