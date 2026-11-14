# Third-party assets

## Three.js

- Version: 0.180.0
- Source: <https://threejs.org/>
- License: MIT
- Vendored files:
  - `vendor/three.module.min.js`
  - `vendor/three.core.min.js`
  - `vendor/OrbitControls.js`

## Earth Blue Marble texture

- Source file: <https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg>
- Local file: `assets/earth-blue-marble.jpg`
- The texture is used only inside the AsterMind globe visualization and is
  vendored so that private deployments do not require public network access.
