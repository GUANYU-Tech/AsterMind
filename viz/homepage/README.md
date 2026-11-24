# AsterMind V1.0 project homepage

## OrbitServe integration

Start the platform from the repository root with `python -B -m viz` using
the environment described in [the frontend guide](../README.md). The project
homepage is served at `/` (Chinese) and `/en/` (English). Previous `/zh/`
links explicitly select Chinese. The homepage and simulation views share the selected
language across navigation, reloads and open tabs on the same origin. Both primary
buttons open the workspace selector at `/platform`. The standalone static
preview below serves only the homepage files.

Static bilingual research homepage inspired by the layout of
https://thu-nics.github.io/uni3sim-website/ and using GUANYU brand guidance.

## Preview and edit

```sh
python3 build.py
node --check dist/app.js
python3 -m http.server 4174 --bind 127.0.0.1 --directory dist
```

Chinese is at `/`; English is at `/en/`. `build.py` owns both languages and
emits complete HTML pages, including content that remains readable without JS.
`dist/style.css` and `dist/app.js` are shared. Language switching preserves the
current section. Request-stage tabs support arrow keys, Home and End.

## Brand assets and typography

- Logo: unchanged approved `guanyu-standard-black.png` from the user-supplied
  GUANYU PPT skill v1.0.2, used at the user's request. SHA-256:
  `0589aee96ee060e09cf329a5709a2f37794ce24b49fb0bc37ffb9add0805e02f`.
- Palette: Space Black `#000000`, Essential White `#FFFFFF`, Matrix Silver
  `#8C8C8C`; neutral tints provide borders and secondary backgrounds.
- Local font preference: Switzer for Latin text, MiSans for Chinese, with
  PingFang SC / Noto Sans CJK SC / system sans-serif fallbacks. Font binaries
  are not redistributed. Identical typography across visitor devices would
  require licensed webfont assets. The font behavior is intentional and
  documented, not a claim of embedded fonts.
- The PPT layout measurements are not used as fixed webpage dimensions.
  Responsive navigation, generous centered headings and diagram-led sections
  follow the reference website; company artwork is not reconstructed.

## Scope and content

The homepage introduces the project. The platform button opens the existing
workspace selector. The diagrams illustrate request stages. Repository examples are starting points, not performance claims.
Content is grounded in the version/AsterMind-V1.0 source inspected for the
initial homepage (a1baab7). No authors, paper URL, benchmark data, or partnership
claims have been invented.

## Verification

- English and Chinese pages: matching sections, valid internal anchors/assets.
- JavaScript syntax check.
- Browser checks at desktop and 390px mobile width: no horizontal overflow,
  logo loaded, language switch retains section, responsive navigation works.
- Four request stages, keyboard navigation, reset, localized command copying.
- No browser console errors observed during the tested flows.

The source includes only the approved logo asset needed for this page, not the
private PPT package. The existing Site remains owner-private.
