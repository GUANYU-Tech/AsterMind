import * as THREE from "three";
import { OrbitControls } from "./vendor/OrbitControls.js?v=compact-2";
import { sceneData } from "./scene-data.mjs?v=compact-2";

import { formationDisplay, orbitDisplay, focusedViewDistance } from "./scene-geometry.mjs?v=scale-3";

const TAU = Math.PI * 2;
const DEG = Math.PI / 180;
const VISUAL_EARTH_ROTATION_SECONDS = 450;
const VISUAL_ORBIT_TIME_SCALE = 30;
const HOME_CAMERA_POSITION = new THREE.Vector3(0.46, 0.72, 7.76);
const HOME_CAMERA_TARGET = new THREE.Vector3(0, 0.08, 0);
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

export const sceneState = {
  stage: "home", nodeCount: 2, layout: "ring", spacingKm: 2, altitudeKm: 535,
  geometrySource: "formation", orbitPositions: null,
  selectedHardwareSatellite: 0, linkBuildStart: null,
  explicitEdges: [],
};

const clusterData = [
  {
    id: "COMPUTE CELL",
    name: "计算 Cell",
    altitude: "520 km",
    status: "535 km",
    radius: 2.38,
    inclination: 47 * DEG,
    ascendingNode: 18 * DEG,
    phase: 0.18,
    periodSeconds: 5680,
  },
  {
    id: "COMPUTE CELL",
    name: "计算 Cell",
    altitude: "548 km",
    status: "548 km",
    radius: 2.44,
    inclination: 62 * DEG,
    ascendingNode: 104 * DEG,
    phase: 2.36,
    periodSeconds: 5740,
  },
  {
    id: "COMPUTE CELL",
    name: "计算 Cell",
    altitude: "505 km",
    status: "505 km",
    radius: 2.34,
    inclination: 33 * DEG,
    ascendingNode: 226 * DEG,
    phase: 4.52,
    periodSeconds: 5620,
  },
  {
    id: "COMPUTE CELL",
    name: "计算 Cell",
    altitude: "560 km",
    status: "560 km",
    radius: 2.47,
    inclination: 71 * DEG,
    ascendingNode: 286 * DEG,
    phase: 5.74,
    periodSeconds: 5780,
  },
  {
    id: "COMPUTE CELL",
    name: "计算 Cell",
    altitude: "532 km",
    status: "532 km",
    radius: 2.4,
    inclination: 52 * DEG,
    ascendingNode: 318 * DEG,
    phase: 1.24,
    periodSeconds: 5700,
  },
  {
    id: "COMPUTE CELL",
    name: "计算 Cell",
    altitude: "542 km",
    status: "542 km",
    radius: 2.42,
    inclination: 38 * DEG,
    ascendingNode: 154 * DEG,
    phase: 3.18,
    periodSeconds: 5720,
  },
  {
    id: "COMPUTE CELL",
    name: "计算 Cell",
    altitude: "518 km",
    status: "518 km",
    radius: 2.37,
    inclination: 67 * DEG,
    ascendingNode: 72 * DEG,
    phase: 4.08,
    periodSeconds: 5660,
  },
  {
    id: "COMPUTE CELL",
    name: "计算 Cell",
    altitude: "555 km",
    status: "555 km",
    radius: 2.46,
    inclination: 43 * DEG,
    ascendingNode: 248 * DEG,
    phase: 5.12,
    periodSeconds: 5760,
  },
];

const groundStationData = [
  { name: "密云接收站", lat: 40.37, lon: 116.85 },
  { name: "喀什接收站", lat: 39.47, lon: 75.99 },
  { name: "三亚接收站", lat: 18.25, lon: 109.51 },
];

const stageOrder = ["formation", "hardware", "terminal", "inference", "analysis"];
const BLUE_LINK = 0x69c8ff;
const RED_LINK = 0xc5e78b;

function createFallbackEarthTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 1024;
  canvas.height = 512;
  const context = canvas.getContext("2d");
  const ocean = context.createLinearGradient(0, 0, 0, 512);
  ocean.addColorStop(0, "#193457");
  ocean.addColorStop(0.48, "#0f2846");
  ocean.addColorStop(1, "#07172a");
  context.fillStyle = ocean;
  context.fillRect(0, 0, 1024, 512);
  context.globalAlpha = 0.7;
  context.fillStyle = "#4b654f";
  [
    [630, 140, 120, 55],
    [725, 215, 80, 105],
    [550, 280, 72, 130],
    [300, 165, 105, 80],
    [230, 260, 65, 145],
  ].forEach(([x, y, rx, ry]) => {
    context.beginPath();
    context.ellipse(x, y, rx, ry, 0.2, 0, TAU);
    context.fill();
  });
  return new THREE.CanvasTexture(canvas);
}

function latLonToVector3(lat, lon, radius) {
  const phi = (90 - lat) * DEG;
  const theta = (lon + 180) * DEG;
  return new THREE.Vector3(
    -radius * Math.sin(phi) * Math.cos(theta),
    radius * Math.cos(phi),
    radius * Math.sin(phi) * Math.sin(theta),
  );
}

function createGroundStationBillboard() {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const context = canvas.getContext("2d");
  context.strokeStyle = "#d8fffb";
  context.lineWidth = 6;
  context.beginPath();
  context.arc(32, 32, 18, 0, TAU);
  context.stroke();
  context.fillStyle = "#79d7d0";
  context.beginPath();
  context.arc(32, 32, 5, 0, TAU);
  context.fill();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({
    map: texture,
    color: 0x79d7d0,
    transparent: true,
    depthTest: true,
    depthWrite: false,
  });
  const marker = new THREE.Sprite(material);
  marker.position.y = 0.075;
  marker.scale.set(0.15, 0.15, 1);
  marker.renderOrder = 21;
  return marker;
}

function disposeObject(object) {
  object.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      const materials = Array.isArray(child.material) ? child.material : [child.material];
      materials.forEach((material) => material.dispose());
    }
  });
}

function createFlowMaterial(color, opacity = 0.82, speed = 0.52) {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uTime: { value: 0 },
      uColor: { value: new THREE.Color(color) },
      uOpacity: { value: opacity },
      uSpeed: { value: speed },
    },
    vertexShader: `
      attribute float aProgress;
      varying float vProgress;
      void main() {
        vProgress = aProgress;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying float vProgress;
      uniform float uTime;
      uniform vec3 uColor;
      uniform float uOpacity;
      uniform float uSpeed;
      void main() {
        float phase = fract(vProgress * 7.0 - uTime * uSpeed);
        float leading = smoothstep(0.0, 0.045, phase);
        float trailing = 1.0 - smoothstep(0.46, 0.58, phase);
        float head = 1.0 - smoothstep(0.0, 0.16, phase);
        float edgeFade = smoothstep(0.0, 0.08, vProgress) * (1.0 - smoothstep(0.92, 1.0, vProgress));
        float alpha = (leading * trailing + head * 0.55) * edgeFade * uOpacity;
        if (alpha < 0.015) discard;
        gl_FragColor = vec4(mix(uColor, vec3(1.0), head * 0.28), alpha);
      }
    `,
  });
}

function createFlowLine(color, opacity = 0.82, speed = 0.52, segments = 72) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array((segments + 1) * 3), 3));
  const progress = new Float32Array(segments + 1);
  for (let index = 0; index <= segments; index += 1) progress[index] = index / segments;
  geometry.setAttribute("aProgress", new THREE.BufferAttribute(progress, 1));
  const line = new THREE.Line(geometry, createFlowMaterial(color, opacity, speed));
  line.frustumCulled = false;
  line.userData.segments = segments;
  return line;
}

function createFlowParticle(color) {
  return new THREE.Mesh(
    new THREE.SphereGeometry(0.018, 12, 8),
    new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity: 0.96,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );
}

function updateArcGeometry(line, start, end, lift = 0.55, radial = true) {
  const segments = line.userData.segments;
  const positions = line.geometry.attributes.position.array;
  const midpoint = start.clone().add(end).multiplyScalar(0.5);

  if (radial && midpoint.lengthSq() > 0.0001) {
    const baseRadius = Math.max(start.length(), end.length());
    midpoint.normalize().multiplyScalar(baseRadius + lift);
  } else {
    midpoint.y += lift;
  }

  const curve = new THREE.QuadraticBezierCurve3(start, midpoint, end);
  line.userData.curve = curve;
  for (let index = 0; index <= segments; index += 1) {
    const point = curve.getPoint(index / segments);
    const offset = index * 3;
    positions[offset] = point.x;
    positions[offset + 1] = point.y;
    positions[offset + 2] = point.z;
  }
  line.geometry.attributes.position.needsUpdate = true;
  line.geometry.computeBoundingSphere();
}

function createLaserBeam(start, end, lift = 0.14, opacity = 0.72) {
  const midpoint = start.clone().add(end).multiplyScalar(0.5);
  midpoint.y += lift;
  const curve = new THREE.QuadraticBezierCurve3(start, midpoint, end);
  return new THREE.Mesh(
    new THREE.TubeGeometry(curve, 32, 0.006, 5, false),
    new THREE.MeshBasicMaterial({
      color: 0xc5e78b,
      transparent: true,
      opacity,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    }),
  );
}

function createLinkHitMesh(start, end, linkIndex) {
  const midpoint = start.clone().add(end).multiplyScalar(0.5);
  const curve = new THREE.QuadraticBezierCurve3(start, midpoint, end);
  const hitMesh = new THREE.Mesh(
    new THREE.TubeGeometry(curve, 12, 0.075, 5, false),
    new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0,
      depthWrite: false,
      colorWrite: false,
    }),
  );
  hitMesh.userData.linkIndex = linkIndex;
  return hitMesh;
}

function createNodeHalo(color) {
  const material = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: { uColor: { value: new THREE.Color(color) } },
    vertexShader: `
      varying vec2 vUv;
      void main() {
        vUv = uv;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying vec2 vUv;
      uniform vec3 uColor;
      void main() {
        float d = distance(vUv, vec2(0.5));
        float alpha = smoothstep(0.5, 0.05, d) * 0.46;
        gl_FragColor = vec4(uColor, alpha);
      }
    `,
  });
  return new THREE.Sprite(material);
}

function createAtmosphereMaterial(opacity = 0.42, power = 2.45) {
  return new THREE.ShaderMaterial({
    side: THREE.FrontSide,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uOpacity: { value: opacity },
      uPower: { value: power },
    },
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vWorldPosition;
      void main() {
        vNormal = normalize(mat3(modelMatrix) * normal);
        vec4 worldPosition = modelMatrix * vec4(position, 1.0);
        vWorldPosition = worldPosition.xyz;
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      varying vec3 vNormal;
      varying vec3 vWorldPosition;
      uniform float uOpacity;
      uniform float uPower;
      void main() {
        vec3 viewDirection = normalize(cameraPosition - vWorldPosition);
        float fresnel = pow(clamp(1.0 - dot(vNormal, viewDirection), 0.0, 1.0), uPower);
        float veil = smoothstep(0.08, 0.92, fresnel);
        float polarShift = clamp(vNormal.y * 0.5 + 0.5, 0.0, 1.0);
        vec3 cyan = vec3(0.16, 0.70, 0.68);
        vec3 violet = vec3(0.36, 0.42, 0.84);
        vec3 color = mix(cyan, violet, polarShift * 0.42 + fresnel * 0.22);
        gl_FragColor = vec4(color, veil * uOpacity);
      }
    `,
  });
}

function makeConceptSatellite(index) {
  const group = new THREE.Group();
  const visual = new THREE.Group();
  const shellMaterial = new THREE.MeshStandardMaterial({
    color: 0xe8edef,
    roughness: 0.34,
    metalness: 0.56,
    emissive: 0x25363a,
    emissiveIntensity: 0.18,
  });
  const panelMaterial = new THREE.MeshStandardMaterial({
    color: 0x2d9995,
    roughness: 0.45,
    metalness: 0.38,
    emissive: 0x0d4f4d,
    emissiveIntensity: 0.72,
  });
  const edgeMaterial = new THREE.LineBasicMaterial({
    color: 0xaef0e7,
    transparent: true,
    opacity: 0.48,
  });

  const bus = new THREE.Mesh(new THREE.BoxGeometry(0.11, 0.095, 0.105), shellMaterial);
  const busEdges = new THREE.LineSegments(
    new THREE.EdgesGeometry(bus.geometry),
    edgeMaterial,
  );
  const leftWing = new THREE.Mesh(new THREE.BoxGeometry(0.17, 0.014, 0.066), panelMaterial);
  leftWing.position.x = -0.14;
  const rightWing = leftWing.clone();
  rightWing.position.x = 0.14;
  visual.add(bus, busEdges, leftWing, rightWing);
  visual.rotation.set(0.14 * Math.sin(index), 0.32 + index * 0.4, index * 0.18);

  const halo = createNodeHalo(0x49bcb4);
  halo.scale.set(0.32, 0.32, 1);
  halo.material.uniforms.uColor.value.set(0x3ba59f);
  const hitArea = new THREE.Mesh(
    new THREE.SphereGeometry(0.32, 12, 8),
    new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0,
      depthWrite: false,
    }),
  );
  hitArea.scale.setScalar(2);

  group.add(halo, visual, hitArea);
  group.userData.halo = halo;
  group.userData.visual = visual;
  group.userData.panelMaterial = panelMaterial;
  group.userData.shellMaterial = shellMaterial;
  group.userData.hitArea = hitArea;
  return group;
}

function makeSatellite(index) {
  const group = new THREE.Group();
  const shellMaterial = new THREE.MeshStandardMaterial({
    color: 0x868d95,
    roughness: 0.28,
    metalness: 0.84,
  });
  const darkMaterial = new THREE.MeshStandardMaterial({
    color: 0x11141a,
    roughness: 0.5,
    metalness: 0.65,
  });
  const panelMaterial = new THREE.MeshStandardMaterial({
    color: 0x171620,
    roughness: 0.32,
    metalness: 0.68,
    emissive: 0x0b0917,
    emissiveIntensity: 0.42,
  });
  const payloadMaterial = new THREE.MeshStandardMaterial({
    color: index === 2 ? 0x8275dc : 0x37334f,
    emissive: index === 2 ? 0x4a33b0 : 0x17142b,
    emissiveIntensity: index === 2 ? 1.25 : 0.56,
    roughness: 0.3,
    metalness: 0.62,
  });
  const goldMaterial = new THREE.MeshStandardMaterial({
    color: 0x9b713a,
    roughness: 0.32,
    metalness: 0.82,
  });

  const bus = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.44, 0.35), darkMaterial);
  const frontFrame = new THREE.Mesh(new THREE.BoxGeometry(0.285, 0.32, 0.025), shellMaterial);
  frontFrame.position.z = 0.188;
  const payload = new THREE.Mesh(new THREE.BoxGeometry(0.23, 0.265, 0.035), payloadMaterial);
  payload.position.z = 0.205;

  const topCap = new THREE.Mesh(new THREE.BoxGeometry(0.37, 0.12, 0.38), shellMaterial);
  topCap.position.y = 0.28;
  const topPlate = new THREE.Mesh(new THREE.BoxGeometry(0.31, 0.018, 0.32), goldMaterial);
  topPlate.position.y = 0.35;

  const radiator = new THREE.Mesh(new THREE.BoxGeometry(0.028, 0.31, 0.29), shellMaterial);
  radiator.position.x = -0.184;
  const radiatorB = radiator.clone();
  radiatorB.position.x = 0.184;

  for (let finIndex = 0; finIndex < 5; finIndex += 1) {
    const y = -0.12 + finIndex * 0.06;
    const fin = new THREE.Mesh(new THREE.BoxGeometry(0.025, 0.014, 0.3), goldMaterial);
    fin.position.set(-0.205, y, 0);
    const finB = fin.clone();
    finB.position.x = 0.205;
    group.add(fin, finB);
  }

  const leftBoom = new THREE.Mesh(new THREE.BoxGeometry(0.22, 0.026, 0.035), goldMaterial);
  leftBoom.position.x = -0.29;
  const rightBoom = leftBoom.clone();
  rightBoom.position.x = 0.29;

  const leftPanel = new THREE.Mesh(new THREE.BoxGeometry(0.48, 0.025, 0.22), panelMaterial);
  leftPanel.position.x = -0.62;
  const rightPanel = leftPanel.clone();
  rightPanel.position.x = 0.62;

  [-0.76, -0.60, -0.46, 0.46, 0.60, 0.76].forEach((x) => {
    const grid = new THREE.Mesh(new THREE.BoxGeometry(0.008, 0.03, 0.225), shellMaterial);
    grid.position.x = x;
    group.add(grid);
  });
  [-0.075, 0, 0.075].forEach((z) => {
    const leftGrid = new THREE.Mesh(new THREE.BoxGeometry(0.49, 0.03, 0.006), shellMaterial);
    leftGrid.position.set(-0.62, 0.001, z);
    const rightGrid = leftGrid.clone();
    rightGrid.position.x = 0.62;
    group.add(leftGrid, rightGrid);
  });

  for (let strip = 0; strip < 5; strip += 1) {
    const payloadStrip = new THREE.Mesh(
      new THREE.BoxGeometry(0.19, 0.012, 0.006),
      new THREE.MeshBasicMaterial({
        color: index === 2 ? 0xb6a8ff : 0x5d587c,
        transparent: true,
        opacity: 0.72,
      }),
    );
    payloadStrip.position.set(0, -0.1 + strip * 0.05, 0.228);
    group.add(payloadStrip);
  }

  const terminalMaterial = new THREE.MeshStandardMaterial({
    color: 0x4a2a24,
    emissive: 0xff4b38,
    emissiveIntensity: 1.65,
    roughness: 0.22,
    metalness: 0.74,
  });
  const terminalBase = new THREE.Mesh(
    new THREE.CylinderGeometry(0.075, 0.095, 0.09, 20),
    goldMaterial,
  );
  terminalBase.rotation.x = Math.PI / 2;
  terminalBase.position.set(0, -0.005, 0.255);
  const terminal = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.065, 0.07, 20), terminalMaterial);
  terminal.rotation.x = Math.PI / 2;
  terminal.position.set(0, -0.005, 0.32);

  group.add(
    bus,
    frontFrame,
    payload,
    topCap,
    topPlate,
    radiator,
    radiatorB,
    leftBoom,
    rightBoom,
    leftPanel,
    rightPanel,
    terminalBase,
    terminal,
  );
  group.userData.payloadMaterial = payloadMaterial;
  group.userData.terminalMaterial = terminalMaterial;
  group.userData.index = index;
  group.scale.setScalar(0.94);
  return group;
}

export class AsterMindScene {
  constructor(canvas, callbacks) {
    this.canvas = canvas;
    this.callbacks = callbacks;
    this.clock = new THREE.Clock();
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x101413);
    this.scene.fog = new THREE.FogExp2(0x101413, 0.014);

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.05, 80);
    this.camera.position.copy(HOME_CAMERA_POSITION);

    this.renderer = new THREE.WebGLRenderer({
      canvas,
      antialias: true,
      alpha: false,
      powerPreference: "high-performance",
    });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.28;

    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.055;
    this.controls.enablePan = false;
    this.controls.minDistance = 6.7;
    this.controls.maxDistance = 9.4;
    this.controls.rotateSpeed = 0.34;
    this.controls.target.copy(HOME_CAMERA_TARGET);

    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.pointerDown = new THREE.Vector2();
    this.hoveredNode = null;
    this.hoveredStation = null;
    this.clusterMeshes = [];
    this.clusterGroups = [];
    this.orbitGuides = [];
    this.groundStations = [];
    this.groundStationMeshes = [];
    this.flowLines = [];
    this.flowParticles = [];
    this.localFlowLines = [];
    this.localBaseLines = [];
    this.linkHitMeshes = [];
    this.localSatelliteMeshes = [];
    this.satellites = [];
    this.hoveredLink = null;
    this.linkClickTimer = null;
    this.configAnchor = new THREE.Vector3(0, 0, 2.42);
    this.configRadial = new THREE.Vector3(0, 0, 1);
    this.configBaseQuaternion = new THREE.Quaternion();
    this.configViewRadial = HOME_CAMERA_POSITION.clone().normalize();
    this.playbackBaseDate = Date.now();
    this.lastFrameTime = performance.now();
    this.stage = "home";
    this.transitionStart = 0;
    this.transitionDuration = 0;
    this.cameraTransition = null;
    this.currentViewOffset = 0;
    this.homeReturnPending = false;

    this.buildLighting();
    this.buildStars();
    this.buildPlanet();
    this.buildOrbitalNodes();
    this.buildConfigScene();
    this.bindEvents();
    this.resize();

    this.renderer.setAnimationLoop(() => this.animate());
  }

  buildLighting() {
    const ambient = new THREE.HemisphereLight(0xb8d8e2, 0x0b1820, 0.92);
    const sun = new THREE.DirectionalLight(0xffffff, 4.25);
    sun.position.set(-5, 3.5, 6);
    const rim = new THREE.DirectionalLight(0x6f9fe0, 1.55);
    rim.position.set(5, -1, -4);
    const coralFill = new THREE.PointLight(0xff735f, 1.4, 12);
    coralFill.position.set(8, 2, 5);
    this.scene.add(ambient, sun, rim, coralFill);
  }

  buildStars() {
    const count = 500;
    const positions = new Float32Array(count * 3);
    for (let index = 0; index < count; index += 1) {
      const radius = 12 + Math.random() * 24;
      const theta = Math.random() * TAU;
      const phi = Math.acos(2 * Math.random() - 1);
      positions[index * 3] = radius * Math.sin(phi) * Math.cos(theta);
      positions[index * 3 + 1] = radius * Math.cos(phi);
      positions[index * 3 + 2] = radius * Math.sin(phi) * Math.sin(theta);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    const material = new THREE.PointsMaterial({
      color: 0xaeb8c8,
      size: 0.015,
      transparent: true,
      opacity: 0.54,
      depthWrite: false,
    });
    this.stars = new THREE.Points(geometry, material);
    this.scene.add(this.stars);
  }

  buildPlanet() {
    this.earthTilt = new THREE.Group();
    this.earthTilt.rotation.z = -23.4 * DEG;
    this.earthSpin = new THREE.Group();
    this.earthTilt.add(this.earthSpin);
    this.scene.add(this.earthTilt);

    const geometry = new THREE.SphereGeometry(2, 96, 64);
    const fallbackTexture = createFallbackEarthTexture();
    const earthMaterial = new THREE.MeshStandardMaterial({
      map: fallbackTexture,
      roughness: 0.68,
      metalness: 0,
      emissive: new THREE.Color(0x071726),
      emissiveIntensity: 0.28,
    });
    this.earthMaterial = earthMaterial;
    this.earth = new THREE.Mesh(geometry, earthMaterial);
    this.earthBaseRotation = 2.75;
    this.earthSpin.rotation.y = this.earthBaseRotation;
    this.earthSpin.add(this.earth);

    const textureLoader = new THREE.TextureLoader();
    textureLoader.load(
      "./assets/earth-blue-marble.jpg",
      (texture) => {
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy());
        earthMaterial.map?.dispose();
        earthMaterial.map = texture;
        earthMaterial.needsUpdate = true;
      },
      undefined,
      () => {},
    );

    const atmosphere = new THREE.Mesh(
      new THREE.SphereGeometry(2.035, 96, 64),
      createAtmosphereMaterial(0.62, 4.4),
    );
    this.earthTilt.add(atmosphere);

    const outerAtmosphere = new THREE.Mesh(
      new THREE.SphereGeometry(2.13, 96, 64),
      createAtmosphereMaterial(0.11, 2.2),
    );
    this.earthTilt.add(outerAtmosphere);

    const latitudeMaterial = new THREE.MeshBasicMaterial({
      color: 0x76aeb6,
      transparent: true,
      opacity: 0.038,
      wireframe: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const latitudeShell = new THREE.Mesh(new THREE.SphereGeometry(2.025, 48, 24), latitudeMaterial);
    this.earthSpin.add(latitudeShell);

    groundStationData.forEach((station) => {
      const group = new THREE.Group();
      group.position.copy(latLonToVector3(station.lat, station.lon, 2.04));
      group.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        group.position.clone().normalize(),
      );

      const mast = new THREE.Mesh(
        new THREE.ConeGeometry(0.018, 0.07, 10),
        new THREE.MeshBasicMaterial({
          color: 0x79d7d0,
          depthTest: true,
          depthWrite: false,
        }),
      );
      mast.position.y = 0.04;
      mast.renderOrder = 20;
      const marker = createGroundStationBillboard();
      const hitArea = new THREE.Mesh(
        new THREE.SphereGeometry(0.15, 12, 8),
        new THREE.MeshBasicMaterial({
          transparent: true,
          opacity: 0,
          depthWrite: false,
        }),
      );
      hitArea.position.y = 0.035;
      hitArea.userData.station = station;
      group.add(mast, marker, hitArea);
      group.userData.station = station;
      group.userData.mastMaterial = mast.material;
      group.userData.markerMaterial = marker.material;
      this.earthSpin.add(group);
      this.groundStations.push(group);
      this.groundStationMeshes.push(hitArea);
    });
  }

  buildOrbitalNodes() {
    clusterData.forEach((data, index) => {
      const group = makeConceptSatellite(index);
      group.userData.hitArea.scale.setScalar(.8);
      const hitArea = group.userData.hitArea;
      hitArea.userData.cluster = data;
      group.userData.data = data;
      this.scene.add(group);
      this.clusterGroups.push(group);
      this.clusterMeshes.push(hitArea);

      const action = document.createElement("button");
      action.type = "button";
      action.className = "orbit-node-action";
      action.setAttribute("aria-label", `配置计算 Cell ${this.clusterGroups.length}`);
      action.title = `计算 Cell ${this.clusterGroups.length}`;
      action.onclick = () => this.callbacks.onSelect(data);
      document.getElementById("orbit-node-actions")?.append(action);
      group.userData.action = action;

      const orbitPoints = [];
      for (let pointIndex = 0; pointIndex <= 160; pointIndex += 1) {
        const angle = (pointIndex / 160) * TAU;
        orbitPoints.push(new THREE.Vector3(
          Math.cos(angle) * data.radius,
          Math.sin(angle) * data.radius,
          0,
        ));
      }
      const orbitGeometry = new THREE.BufferGeometry().setFromPoints(orbitPoints);
      const orbitMaterial = new THREE.LineDashedMaterial({
        color: 0x59b8b4,
        transparent: true,
        opacity: 0.22,
        dashSize: 0.055,
        gapSize: 0.13,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const orbitGuide = new THREE.Line(orbitGeometry, orbitMaterial);
      orbitGuide.computeLineDistances();
      orbitGuide.rotation.x = data.inclination;
      orbitGuide.rotation.y = data.ascendingNode;
      this.scene.add(orbitGuide);
      this.orbitGuides.push(orbitGuide);
    });

    [
      [0, 1], [1, 4], [4, 2], [2, 6],
      [6, 3], [3, 7], [7, 5], [5, 0],
      [1, 6], [4, 7],
    ].forEach(([a, b], index) => {
      const line = createFlowLine(BLUE_LINK, 0.76, 0.44 + (index % 3) * 0.035);
      line.userData.kind = "node-node";
      line.userData.a = a;
      line.userData.b = b;
      this.scene.add(line);
      this.flowLines.push(line);
    });

    [
      [0, 0],
      [2, 2],
      [5, 1],
      [7, 0],
    ].forEach(([clusterIndex, stationIndex]) => {
      const line = createFlowLine(RED_LINK, 0.8, 0.58);
      line.userData.kind = "node-ground";
      line.userData.clusterIndex = clusterIndex;
      line.userData.stationIndex = stationIndex;
      this.scene.add(line);
      this.flowLines.push(line);
    });

    [
      { line: this.flowLines.find((item) => item.userData.kind === "node-node"), color: BLUE_LINK, offset: 0.08 },
      { line: this.flowLines.find((item) => item.userData.kind === "node-ground"), color: RED_LINK, offset: 0.52 },
    ].forEach(({ line, color, offset }) => {
      if (!line) return;
      const particle = createFlowParticle(color);
      particle.userData.line = line;
      particle.userData.offset = offset;
      particle.visible = false;
      this.scene.add(particle);
      this.flowParticles.push(particle);
    });
  }

  buildConfigScene() {
    this.configRoot = new THREE.Group();
    this.configRoot.position.copy(this.configAnchor);
    this.scene.add(this.configRoot);

    this.configStage = new THREE.Mesh(
      new THREE.RingGeometry(1.7, 1.704, 96),
      new THREE.MeshBasicMaterial({
        color: 0x8275dc,
        transparent: true,
        opacity: 0.14,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
      }),
    );
    this.configStage.rotation.x = Math.PI / 2;
    this.configStage.visible = false;
    this.configRoot.add(this.configStage);

    this.satelliteRoot = new THREE.Group();
    this.localLinksRoot = new THREE.Group();
    this.configRoot.add(this.satelliteRoot, this.localLinksRoot);
    this.configRoot.visible = false;
    this.altitudeGuide = new THREE.Line(new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({color: 0xc5e78b, transparent: true, opacity: .48}));
    this.altitudeGuide.visible = false;
    this.scene.add(this.altitudeGuide);
    this.rebuildFormation();
  }

  prepareConfigAnchor(cluster) {
    this.configViewRadial.copy(this.camera.position).normalize();
    const view = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,0,1), this.configViewRadial);
    const group = this.clusterGroups.find(item => item.userData.data === cluster);
    const screen = group ? group.position.clone().normalize().applyQuaternion(view.clone().invert()) : new THREE.Vector3(.9,.4,0);
    const lateral = new THREE.Vector2(screen.x, screen.y);
    if (lateral.length() < .1) lateral.set(.9,.4);
    lateral.normalize();
    // Keep the formation near the limb, so changing height visibly changes
    // surface clearance instead of only changing perspective magnification.
    const angle = 60 * DEG;
    this.configRadial.set(lateral.x*Math.sin(angle), lateral.y*Math.sin(angle), Math.cos(angle)).applyQuaternion(view);
    this.configBaseQuaternion.setFromUnitVectors(new THREE.Vector3(0,0,1), this.configRadial);
    this.applyFormationFrame();
    this.orientSatelliteVisuals();
  }

  orientSatelliteVisuals() {
    // Choose a readable initial spacecraft attitude independently of the Cell's
    // location. Keep it fixed during editing and subsequent manual camera moves.
    const view = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,0,1), this.configViewRadial);
    const localView = this.configRoot.quaternion.clone().invert().multiply(view);
    this.satellites.forEach((satellite, index) => {
      const attitude = new THREE.Quaternion().setFromEuler(new THREE.Euler(.95, .15 + (index % 4) * .04, .25));
      satellite.userData.visual.quaternion.copy(localView).multiply(attitude);
    });
  }

  displayModel() {
    if (sceneState.geometrySource === "orbit" && sceneState.orbitPositions) {
      return {positions:sceneState.orbitPositions, orbitRadius:0, satelliteScale:.3};
    }
    return formationDisplay(sceneState.nodeCount, sceneState.layout, sceneState.spacingKm, sceneState.altitudeKm);
  }

  applyFormationFrame() {
    const model = this.displayModel();
    if (sceneState.geometrySource === "orbit") {
      this.configRoot.position.set(0,0,0);
      this.configRoot.quaternion.identity();
      this.altitudeGuide.visible = false;
    } else {
      this.configAnchor.copy(this.configRadial).multiplyScalar(model.orbitRadius);
      this.configRoot.position.copy(this.configAnchor);
      this.configRoot.quaternion.copy(this.configBaseQuaternion);
      this.altitudeGuide.geometry.setFromPoints([
        this.configRadial.clone().multiplyScalar(2.03), this.configAnchor.clone(),
      ]);
      this.altitudeGuide.visible = !["home", "analysis"].includes(this.stage);
    }
  }

  defaultFormationEdges(positions) {
    return sceneState.explicitEdges;
  }

  rebuildFormation() {
    this.satellites.forEach((satellite) => {
      this.satelliteRoot.remove(satellite);
      disposeObject(satellite);
    });
    this.localFlowLines.forEach((line) => {
      this.localLinksRoot.remove(line);
      line.geometry.dispose();
      line.material.dispose();
    });
    this.localBaseLines.forEach((line) => {
      this.localLinksRoot.remove(line);
      line.geometry.dispose();
      line.material.dispose();
    });
    this.linkHitMeshes.forEach((mesh) => {
      this.localLinksRoot.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    });
    this.satellites = [];
    this.localFlowLines = [];
    this.localBaseLines = [];
    this.linkHitMeshes = [];
    this.localSatelliteMeshes = [];
    this.hoveredLink = null;

    const model = this.displayModel();
    const positions = sceneState.geometrySource === "orbit" && !sceneState.orbitPositions ? [] : model.positions.map(position => new THREE.Vector3(...position));
    this.applyFormationFrame();
    positions.forEach((position, index) => {
      const satellite = makeConceptSatellite(index);
      satellite.position.copy(position);
      satellite.userData.basePosition = position.clone();
      satellite.scale.setScalar(model.satelliteScale);
      satellite.userData.hitArea.userData.formationNodeIndex = index;
      satellite.userData.hitArea.scale.setScalar(.55);
      this.satelliteRoot.add(satellite);
      this.satellites.push(satellite);
      this.localSatelliteMeshes.push(satellite.userData.hitArea);
    });
    this.orientSatelliteVisuals();

    const edges = this.defaultFormationEdges(positions);

    edges.filter(([a,b]) => this.satellites[a] && this.satellites[b] && a !== b).forEach(([index, next], linkIndex) => {
      const line = createFlowLine(RED_LINK, 0.9, 0.58, 48);
      updateArcGeometry(
        line,
        this.satellites[index].position,
        this.satellites[next].position,
        0,
        false,
      );
      const baseLine = createLaserBeam(
        this.satellites[index].position,
        this.satellites[next].position,
        0,
        0.1,
      );
      const hitMesh = createLinkHitMesh(
        this.satellites[index].position,
        this.satellites[next].position,
        linkIndex,
      );
      hitMesh.userData.edge = [index, next];
      hitMesh.userData.beam = baseLine;
      hitMesh.userData.flowLine = line;
      this.localLinksRoot.add(baseLine);
      this.localBaseLines.push(baseLine);
      this.localLinksRoot.add(line);
      this.localFlowLines.push(line);
      this.localLinksRoot.add(hitMesh);
      this.linkHitMeshes.push(hitMesh);
    });

    this.updateStrategyVisual();
    this.setHardwareSatelliteSelection(sceneState.selectedHardwareSatellite);
    const host = document.querySelector("#node-labels");
    if (host) {
      host.replaceChildren();
      this.nodeLabels = this.satellites.map((satellite, index) => {
        const label = document.createElement("span");
        label.className = "node-label"; label.textContent = sceneState.nodeIds?.[index] || `sat-${index}`;
        host.append(label); return {satellite,label};
      });
    }
  }

  updateStrategyVisual() {
    this.satellites.forEach((satellite) => {
      const color = new THREE.Color(0x2d9995);
      const material = satellite.userData.panelMaterial || satellite.userData.payloadMaterial;
      material.color.copy(color);
      material.emissive.copy(color).multiplyScalar(0.32);
    });
  }

  bindEvents() {
    this.canvas.addEventListener("pointerdown", (event) => {
      this.pointerDown.set(event.clientX, event.clientY);
    });

    this.canvas.addEventListener("pointermove", (event) => {
      const rect = this.canvas.getBoundingClientRect();
      this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      this.scene.updateMatrixWorld(true);
      this.camera.updateMatrixWorld(true);
      this.raycaster.setFromCamera(this.pointer, this.camera);

      if (this.stage === "home") {
        const nextNode =
          this.raycaster.intersectObjects(this.clusterMeshes, false)[0]?.object || null;
        const stationHit = nextNode
          ? null
          : this.raycaster.intersectObjects(this.groundStationMeshes, false)[0] || null;
        const earthHit = stationHit
          ? this.raycaster.intersectObject(this.earth, false)[0] || null
          : null;
        const nextStation = stationHit && (!earthHit || stationHit.distance < earthHit.distance)
          ? stationHit.object
          : null;
        if (nextNode !== this.hoveredNode) {
          if (this.hoveredNode) this.setNodeHover(this.hoveredNode, false);
          this.hoveredNode = nextNode;
          if (nextNode) this.setNodeHover(nextNode, true);
        }
        if (nextStation !== this.hoveredStation) {
          if (this.hoveredStation) this.setStationHover(this.hoveredStation, false);
          this.hoveredStation = nextStation;
          if (nextStation) this.setStationHover(nextStation, true);
        }
        this.canvas.classList.toggle("is-hovering", Boolean(nextNode || nextStation));
        if (nextNode) {
          this.callbacks.onHover(nextNode.userData.cluster);
        } else if (nextStation) {
          this.callbacks.onHoverStation(nextStation.userData.station);
        } else {
          this.callbacks.onHover(null);
        }
        return;
      }

      if (this.hoveredNode) {
        this.setNodeHover(this.hoveredNode, false);
        this.hoveredNode = null;
        this.callbacks.onHover(null);
      }
      if (this.hoveredStation) {
        this.setStationHover(this.hoveredStation, false);
        this.hoveredStation = null;
        this.callbacks.onHoverStation(null);
      }

      const linkInteractive = this.stage === "hardware"
        || this.stage === "terminal";
      const nextLink = linkInteractive
        ? this.raycaster.intersectObjects(this.linkHitMeshes, false)[0]?.object || null
        : null;
      const nextFormationNode = this.stage === "formation"
        || this.stage === "hardware"
        || this.stage === "terminal"
        ? this.raycaster.intersectObjects(this.localSatelliteMeshes, false)[0]?.object || null
        : null;
      if (nextLink !== this.hoveredLink) {
        if (this.hoveredLink) this.setLinkHover(this.hoveredLink, false);
        this.hoveredLink = nextLink;
        if (nextLink) this.setLinkHover(nextLink, true);
      }
      this.canvas.classList.toggle("is-hovering", Boolean(nextLink || nextFormationNode));
    });

    this.canvas.addEventListener("click", (event) => {
      const movement = Math.hypot(
        event.clientX - this.pointerDown.x,
        event.clientY - this.pointerDown.y,
      );
      if (movement >= 12) return;
      const rect = this.canvas.getBoundingClientRect();
      this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      this.scene.updateMatrixWorld(true);
      this.camera.updateMatrixWorld(true);
      this.raycaster.setFromCamera(this.pointer, this.camera);
      if (this.stage === "home") {
        const hit = this.raycaster.intersectObjects(this.clusterMeshes, false)[0];
        if (hit?.object.userData.cluster) {
          this.callbacks.onSelect(hit.object.userData.cluster);
        }
      } else if (this.stage === "formation") {
        const satelliteHit = this.raycaster.intersectObjects(this.localSatelliteMeshes, false)[0];
        if (satelliteHit?.object) {
          this.callbacks.onFormationNodeSelect(
            satelliteHit.object.userData.formationNodeIndex,
          );
        }
      } else if (this.stage === "hardware" || this.stage === "terminal") {
        const satelliteHit = this.raycaster.intersectObjects(this.localSatelliteMeshes, false)[0];
        if (satelliteHit?.object) {
          const satelliteIndex = satelliteHit.object.userData.formationNodeIndex;
          if (this.stage === "hardware") {
            this.callbacks.onHardwareNodeSelect(satelliteIndex);
          } else {
            this.callbacks.onTerminalNodeSelect(satelliteIndex);
          }
          return;
        }
        const linkHit = this.raycaster.intersectObjects(this.linkHitMeshes, false)[0];
        if (linkHit?.object) {
          window.clearTimeout(this.linkClickTimer);
          const { linkIndex, edge } = linkHit.object.userData;
          this.linkClickTimer = window.setTimeout(() => {
            this.callbacks.onLinkSelect(linkIndex, edge);
          }, 230);
          return;
        }
      }
    });

    this.canvas.addEventListener("dblclick", (event) => {
      if (this.stage !== "hardware" && this.stage !== "terminal") return;
      event.preventDefault();
      window.clearTimeout(this.linkClickTimer);
      const rect = this.canvas.getBoundingClientRect();
      this.pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      this.pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      this.scene.updateMatrixWorld(true);
      this.camera.updateMatrixWorld(true);
      this.raycaster.setFromCamera(this.pointer, this.camera);
      const linkHit = this.raycaster.intersectObjects(this.linkHitMeshes, false)[0];
      if (linkHit?.object.userData.edge) {
        this.callbacks.onLinkRemove(linkHit.object.userData.edge);
      }
    });

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(this.canvas);
  }

  setNodeHover(node, active) {
    const group = node.parent;
    group.userData.halo.scale.setScalar(active ? 0.44 : 0.32);
    group.userData.panelMaterial.emissive.set(active ? 0x486437 : 0x0d4f4d);
    group.userData.panelMaterial.color.set(active ? 0xc5e78b : 0x2d9995);
    group.userData.shellMaterial.emissiveIntensity = active ? 0.52 : 0.18;
  }

  setStationHover(node, active) {
    const group = node.parent;
    group.scale.setScalar(active ? 1.45 : 1);
    group.userData.mastMaterial.color.set(active ? 0xffffff : 0x79d7d0);
    group.userData.markerMaterial.color.set(active ? 0xffffff : 0x79d7d0);
  }

  setLinkHover(hitMesh, active) {
    const beam = hitMesh.userData.beam;
    const flowLine = hitMesh.userData.flowLine;
    beam.material.opacity = active ? 0.4 : 0.1;
    beam.material.color.set(active ? 0xffb3a5 : 0xff735f);
    flowLine.material.uniforms.uOpacity.value = active ? 1 : 0.9;
    flowLine.material.uniforms.uColor.value.set(active ? 0xffd3ca : 0xff735f);
  }

  setStage(stage) {
    const previousStage = this.stage;
    this.earthTilt.visible = true;
    const isConfigStage = (value) =>
      value === "formation" || value === "hardware"
      || value === "terminal" || value === "inference";
    const preserveConfigView = isConfigStage(previousStage) && isConfigStage(stage);
    const target = preserveConfigView
      ? {
          position: this.camera.position.clone(),
          lookAt: this.controls.target.clone(),
        }
      : this.getCameraTarget(stage);
    this.stage = stage;
    this.transitionStart = performance.now();
    this.transitionDuration = preserveConfigView
      ? 220
      : stage === "formation"
        ? 1500
        : stage === "home"
          ? 1250
          : 520;
    const fromDirection = this.camera.position.clone().normalize();
    const toDirection = target.position.clone().normalize();
    this.cameraTransition = {
      fromPosition: this.camera.position.clone(),
      fromTarget: this.controls.target.clone(),
      toPosition: target.position,
      toTarget: target.lookAt,
      fromDirection,
      toDirection,
      fromRadius: this.camera.position.length(),
      toRadius: target.position.length(),
      directionRotation: new THREE.Quaternion().setFromUnitVectors(
        fromDirection,
        toDirection,
      ),
      fromViewOffset: this.currentViewOffset,
      toViewOffset: isConfigStage(stage) ? this.getConfigViewOffset() : 0,
    };
    this.homeReturnPending = stage === "home";
    this.configRoot.visible = stage !== "home" && stage !== "analysis";
    this.localLinksRoot.visible = stage !== "home" && stage !== "analysis";
    this.clusterGroups.forEach((group) => {
      group.visible = stage === "home";
    });
    this.orbitGuides.forEach((guide) => {
      guide.visible = stage === "home";
    });
    this.controls.enabled = false;
    this.controls.enableDamping = false;
    if (this.hoveredLink) {
      this.setLinkHover(this.hoveredLink, false);
      this.hoveredLink = null;
    }
    this.canvas.classList.remove("is-hovering");
    this.updateStrategyVisual();
    this.setHardwareSatelliteSelection(sceneState.selectedHardwareSatellite);
    this.altitudeGuide.visible = isConfigStage(stage) && sceneState.geometrySource !== "orbit";
    if (preserveConfigView) {
      this.cameraTransition = null;
      this.controls.enabled = true;
      this.controls.enableDamping = true;
    } else if (isConfigStage(stage)) this.resetConfigView();
    if (stage === "home") {
      this.configRoot.visible = false;
      this.callbacks.onHover(null);
      this.callbacks.onHoverStation(null);
    }
  }

  setHardwareSatelliteSelection(index) {
    this.satellites.forEach((satellite, satelliteIndex) => {
      const active = (this.stage === "hardware" && satelliteIndex === index)
        || (this.stage === "terminal" && satelliteIndex === sceneState.linkBuildStart);
      satellite.userData.halo.scale.setScalar(active ? 0.42 : 0.32);
      satellite.userData.halo.material.uniforms.uColor.value.set(
        active && this.stage === "terminal" ? 0x79d7d0 : active ? 0xff9a86 : 0x3ba59f,
      );
    });
  }

  updateConfig() {
    const model = this.displayModel();
    const signature = JSON.stringify([model.positions, model.orbitRadius, sceneState.geometrySource, sceneState.explicitEdges]);
    if (signature !== this.formationSignature) {
      this.formationSignature = signature;
      this.rebuildFormation();
    }
    // Names/card inventory/link prediction do not own the camera or orientation.
    (this.nodeLabels || []).forEach(({label}, index) => {
      label.textContent = sceneState.nodeIds?.[index] || "";
    });
  }

  resetConfigView(preserveDirection = false) {
    if (["home", "analysis"].includes(this.stage)) return;
    this.cameraTransition = null;
    this.currentViewOffset = 0;
    this.camera.clearViewOffset();
    this.configRoot.updateMatrixWorld(true);
    const direction = preserveDirection ? this.camera.position.clone().sub(this.controls.target).normalize() : this.configViewRadial;
    const inverse = new THREE.Quaternion().setFromUnitVectors(direction, new THREE.Vector3(0,0,1));
    const world = this.satellites.map(satellite => satellite.getWorldPosition(new THREE.Vector3()));
    const target = world.reduce((center,point)=>center.add(point), new THREE.Vector3()).divideScalar(world.length || 1);
    const points = world.map(point=>point.clone().sub(target).applyQuaternion(inverse).toArray());
    const distance = focusedViewDistance(points, this.camera.aspect, this.camera.fov);
    this.camera.far = Math.max(80, distance * 6);
    this.camera.position.copy(target).addScaledVector(direction,distance);
    this.controls.target.copy(target);
    this.controls.minDistance = .65;
    this.controls.maxDistance = Math.max(14,distance*3);
    this.controls.enabled = true;
    this.controls.enableDamping = true;
    this.controls.enableZoom = true;
    this.controls.enableRotate = true;
    this.controls.rotateSpeed = .4;
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  updateStrategy() {
    this.updateStrategyVisual();
  }

  updateClusterPositions() {
    const liveSeconds = prefersReducedMotion
      ? 0
      : (Date.now() - this.playbackBaseDate) / 1000;
    const visualOrbitTime =
      this.playbackBaseDate / 1000 + liveSeconds * VISUAL_ORBIT_TIME_SCALE;
    this.clusterGroups.forEach((group, index) => {
      const data = group.userData.data;
      const angle = data.phase + (visualOrbitTime / data.periodSeconds) * TAU;
      const base = new THREE.Vector3(
        Math.cos(angle) * data.radius,
        Math.sin(angle) * data.radius,
        0,
      );
      base.applyAxisAngle(new THREE.Vector3(1, 0, 0), data.inclination);
      base.applyAxisAngle(new THREE.Vector3(0, 1, 0), data.ascendingNode);
      group.position.copy(base);
      group.lookAt(0, 0, 0);
      // The overview marker represents a Cell, so keep its solar panels legible
      // from the user's camera instead of depicting a physical attitude.
      group.userData.visual.quaternion.copy(group.quaternion).invert().multiply(this.camera.quaternion)
        .multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(.95,.15,.25)));
      group.scale.setScalar(this.hoveredNode?.parent === group ? 1.05 : .95);
    });
  }

  updateFlowLines(time) {
    const stationPosition = new THREE.Vector3();
    this.flowLines.forEach((line) => {
      line.material.uniforms.uTime.value = time;
      if (line.userData.kind === "node-node") {
        updateArcGeometry(
          line,
          this.clusterGroups[line.userData.a].position,
          this.clusterGroups[line.userData.b].position,
          0.36,
          true,
        );
      } else {
        this.groundStations[line.userData.stationIndex].getWorldPosition(stationPosition);
        updateArcGeometry(
          line,
          this.clusterGroups[line.userData.clusterIndex].position,
          stationPosition,
          0.48,
          true,
        );
      }
      line.visible = this.stage === "home";
    });

    this.flowParticles.forEach((particle) => {
      const curve = particle.userData.line?.userData.curve;
      const progress = (time * 0.16 + particle.userData.offset) % 1;
      particle.visible = this.stage === "home" && Boolean(curve);
      if (curve) particle.position.copy(curve.getPoint(progress));
    });

    this.localFlowLines.forEach((line, index) => {
      line.material.uniforms.uTime.value = time + index * 0.08;
    });
  }

  getCameraTarget(stage) {
    if (stage === "formation" || stage === "hardware" || stage === "terminal" || stage === "inference") {
      const tangent = new THREE.Vector3(1, 0, 0)
        .applyQuaternion(this.configBaseQuaternion)
        .normalize();
      const leftSceneOffset = tangent.multiplyScalar(0.68);
      return {
        position: this.configRadial.clone().multiplyScalar(7.05).add(leftSceneOffset),
        lookAt: this.configAnchor.clone(),
      };
    }
    const targets = {
      home: {
        position: HOME_CAMERA_POSITION.clone(),
        lookAt: HOME_CAMERA_TARGET.clone(),
      },
      analysis: {
        position: new THREE.Vector3(0, 0.24, 8.8),
        lookAt: new THREE.Vector3(0, 0, 0),
      },
    };
    return targets[stage];
  }

  getConfigViewOffset() {
    // The canvas occupies the available preview area, beside/before the editor.
    return 0;
  }

  applyCameraViewOffset(offset) {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) return;
    if (Math.abs(offset) < 0.01) {
      this.camera.clearViewOffset();
      return;
    }
    this.camera.setViewOffset(width, height, offset, 0, width, height);
  }

  updateCamera() {
    if (!this.cameraTransition) {
      if (this.controls.enabled) this.controls.update();
      return;
    }

    const elapsed = performance.now() - this.transitionStart;
    const rawProgress = prefersReducedMotion ? 1 : Math.min(1, elapsed / this.transitionDuration);
    const progress = 0.5 - Math.cos(rawProgress * Math.PI) * 0.5;
    const directionStep = new THREE.Quaternion().slerpQuaternions(
      new THREE.Quaternion(),
      this.cameraTransition.directionRotation,
      progress,
    );
    const orbitDirection = this.cameraTransition.fromDirection
      .clone()
      .applyQuaternion(directionStep)
      .normalize();
    const orbitRadius = THREE.MathUtils.lerp(
      this.cameraTransition.fromRadius,
      this.cameraTransition.toRadius,
      progress,
    );
    this.camera.position.copy(orbitDirection.multiplyScalar(orbitRadius));
    this.controls.target.lerpVectors(
      this.cameraTransition.fromTarget,
      this.cameraTransition.toTarget,
      progress,
    );
    this.currentViewOffset = THREE.MathUtils.lerp(
      this.cameraTransition.fromViewOffset,
      this.cameraTransition.toViewOffset,
      progress,
    );
    this.applyCameraViewOffset(this.currentViewOffset);
    this.camera.lookAt(this.controls.target);

    if (rawProgress >= 1) {
      this.cameraTransition = null;
      this.currentViewOffset = this.stage === "formation"
        || this.stage === "hardware"
        || this.stage === "terminal"
        || this.stage === "inference"
        ? this.getConfigViewOffset()
        : 0;
      this.applyCameraViewOffset(this.currentViewOffset);
      this.homeReturnPending = false;
      this.controls.enabled = this.stage !== "analysis";
      this.controls.enableDamping = true;
      this.controls.enableZoom = this.stage !== "analysis";
      this.controls.enableRotate = this.stage !== "analysis";
      this.controls.rotateSpeed = this.stage === "home" ? 0.34 : 0.13;
      this.controls.zoomSpeed = this.stage === "home" ? 1 : 0.42;
      this.controls.minDistance = this.stage === "home" ? 6.7 : 2.15;
      this.controls.maxDistance = this.stage === "home" ? 9.4 : 7.85;
      if (this.stage === "home") {
        this.controls.minAzimuthAngle = -Infinity;
        this.controls.maxAzimuthAngle = Infinity;
        this.controls.minPolarAngle = 0.25;
        this.controls.maxPolarAngle = Math.PI - 0.25;
      } else {
        this.controls.minAzimuthAngle = -Infinity;
        this.controls.maxAzimuthAngle = Infinity;
        this.controls.minPolarAngle = 0.2;
        this.controls.maxPolarAngle = Math.PI - 0.2;
      }
    }
  }

  animate() {
    const delta = Math.min(this.clock.getDelta(), 0.04);
    const elapsed = performance.now();
    if (!prefersReducedMotion) {
      this.stars.rotation.y += delta * 0.0014;
    }
    const liveProgress = prefersReducedMotion
      ? 0
      : ((Date.now() - this.playbackBaseDate) / 1000)
        * TAU / VISUAL_EARTH_ROTATION_SECONDS;
    if (this.stage === "home") this.earthSpin.rotation.y = this.earthBaseRotation + liveProgress;
    this.updateClusterPositions();
    this.updateFlowLines(elapsed / 1000);
    this.updateCamera();
    if (!["home", "analysis"].includes(this.stage) && this.camera.position.length() < 2.08) {
      this.camera.position.setLength(2.08);
      this.camera.lookAt(this.controls.target);
    }

    if (this.stage === "formation" || this.stage === "hardware" || this.stage === "terminal" || this.stage === "inference") {
      this.satellites.forEach((satellite, index) => {
        // The spacecraft pose stays stable while comparing configuration edits.
        const basePosition = satellite.userData.basePosition;
        if (basePosition) {
          satellite.position.copy(basePosition);
        }
      });
    }

    const earthOpacity = this.stage === "analysis" ? 0.22 : 1;
    this.earth.material.transparent = earthOpacity < 1;
    this.earth.material.opacity = earthOpacity;

    this.scene.updateMatrixWorld(true);
    const rect = this.canvas.getBoundingClientRect();
    for (const satellite of this.clusterGroups) {
      const action = satellite.userData.action;
      const world = satellite.getWorldPosition(new THREE.Vector3());
      const point = world.clone().project(this.camera);
      const sightline = world.clone().sub(this.camera.position);
      const fraction = THREE.MathUtils.clamp(-this.camera.position.dot(sightline) / sightline.lengthSq(), 0, 1);
      const occluded = this.camera.position.clone().addScaledVector(sightline, fraction).length() < 2.03;
      action.hidden = this.stage !== "home" || occluded || Math.abs(point.x) > .98 || Math.abs(point.y) > .98 || point.z < -1 || point.z > 1;
      action.style.left = `${rect.left + (point.x+1)*rect.width/2}px`;
      action.style.top = `${rect.top + (1-point.y)*rect.height/2}px`;
    }
    for (const {satellite,label} of this.nodeLabels || []) {
      const point = satellite.getWorldPosition(new THREE.Vector3()).project(this.camera);
      const world = satellite.getWorldPosition(new THREE.Vector3());
      const sightline = world.clone().sub(this.camera.position);
      const fraction = THREE.MathUtils.clamp(-this.camera.position.dot(sightline) / sightline.lengthSq(), 0, 1);
      const occluded = this.camera.position.clone().addScaledVector(sightline, fraction).length() < 2.03;
      label.hidden = occluded || !this.configRoot.visible || Math.abs(point.x) > .98 || Math.abs(point.y) > .98 || point.z < -1 || point.z > 1;
      label.style.left = `${rect.left + (point.x+1)*rect.width/2 + 12}px`;
      label.style.top = `${rect.top + (1-point.y)*rect.height/2 + 18}px`;
    }
    this.renderer.render(this.scene, this.camera);
  }

  resize() {
    const width = this.canvas.clientWidth;
    const height = this.canvas.clientHeight;
    if (!width || !height) return;
    this.camera.aspect = width / height;
    this.renderer.setSize(width, height, false);
    if (!this.cameraTransition) {
      this.currentViewOffset = this.stage === "formation"
        || this.stage === "hardware"
        || this.stage === "terminal"
        || this.stage === "inference"
        ? this.getConfigViewOffset()
        : 0;
    }
    this.applyCameraViewOffset(this.currentViewOffset);
    this.camera.updateProjectionMatrix();
    if (!["home", "analysis"].includes(this.stage)) this.resetConfigView(true);
  }
}

export function syncSceneConfiguration(scene, config, preview = null) {
  const ids = config.hardware?.satellite_ids || [];
  sceneState.nodeCount = ids.length;
  sceneState.nodeIds = ids;
  sceneState.layout = config.constellation?.formation?.layout || "ring";
  sceneState.spacingKm = config.constellation?.formation?.spacing_km ?? 2;
  sceneState.altitudeKm = config.constellation?.formation?.altitude_km ?? 535;
  sceneState.geometrySource = config.topology?.source || "formation";
  if (sceneState.geometrySource === "orbit") {
    const geometryKey = JSON.stringify([ids, config.constellation]);
    const positions = orbitDisplay(config, preview);
    if (positions) scene.orbitGeometry = {key:geometryKey, positions};
    sceneState.orbitPositions = scene.orbitGeometry?.key === geometryKey ? scene.orbitGeometry.positions : null;
  } else sceneState.orbitPositions = null;
  const data = sceneData(config, preview);
  sceneState.explicitEdges = data.edges;
  scene.updateConfig();
}
