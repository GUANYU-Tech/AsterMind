function createFallbackEarthTexture(THREE) {
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
    context.ellipse(x, y, rx, ry, 0.2, 0, Math.PI * 2);
    context.fill();
  });
  return new THREE.CanvasTexture(canvas);
}

function createAtmosphereMaterial(THREE, opacity = 0.42, power = 2.45) {
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

export function createEarth(THREE, renderer, onReady = () => {}) {
  const earthTilt = new THREE.Group();
  earthTilt.rotation.z = -23.4 * Math.PI / 180;
  const earthSpin = new THREE.Group();
  earthTilt.add(earthSpin);


  const geometry = new THREE.SphereGeometry(2, 96, 64);
  const fallbackTexture = createFallbackEarthTexture(THREE);
  const earthMaterial = new THREE.MeshStandardMaterial({
    map: fallbackTexture,
    roughness: 0.68,
    metalness: 0,
    emissive: new THREE.Color(0x071726),
    emissiveIntensity: 0.28,
  });

  const earth = new THREE.Mesh(geometry, earthMaterial);
  const earthBaseRotation = 2.75;
  earthSpin.rotation.y = earthBaseRotation;
  earthSpin.add(earth);

  const textureLoader = new THREE.TextureLoader();
  textureLoader.load(
    "/cell/assets/earth-blue-marble.jpg",
    (texture) => {
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
      earthMaterial.map?.dispose();
      earthMaterial.map = texture;
      earthMaterial.needsUpdate = true;
      onReady();
    },
    undefined,
    onReady,
  );

  const atmosphere = new THREE.Mesh(
    new THREE.SphereGeometry(2.035, 96, 64),
    createAtmosphereMaterial(THREE, 0.62, 4.4),
  );
  earthTilt.add(atmosphere);

  const outerAtmosphere = new THREE.Mesh(
    new THREE.SphereGeometry(2.13, 96, 64),
    createAtmosphereMaterial(THREE, 0.11, 2.2),
  );
  earthTilt.add(outerAtmosphere);

  const latitudeMaterial = new THREE.MeshBasicMaterial({
    color: 0x76aeb6,
    transparent: true,
    opacity: 0.038,
    wireframe: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const latitudeShell = new THREE.Mesh(new THREE.SphereGeometry(2.025, 48, 24), latitudeMaterial);
  earthSpin.add(latitudeShell);
  return {earth, earthTilt, earthSpin, earthMaterial, earthBaseRotation};
}
