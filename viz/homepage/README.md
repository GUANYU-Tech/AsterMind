# AsterMind V1.0 project homepage

## OrbitServe integration

The project website is built independently for GitHub Pages. To run the simulator,
follow the [frontend guide](../README.md) and open [the local workspace selector](http://localhost:8766/).
Both simulation workspaces are available directly when the service starts.

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
current section. The four request stages cycle while the diagram is visible. Readers can pause, restart or select a stage. Keyboard navigation supports arrow keys, Home and End. Reduced-motion mode starts paused.

## Export the public homepage

From the repository root, build into a fresh directory outside the checkout.

```sh
python3 viz/homepage/build.py --pages-dir ../astermind-website
```

The export contains both languages, the brand images, the desktop and mobile
Chinese posters, and a copy of the [technical report](../../doc/AsterMind-V1-Tech-Report.pdf).
Update the PDF in `doc/` when a new report is available. The report opens inline
at `#pdf` and loads on first use.

The [Pages workflow](../../.github/workflows/pages.yml) builds both languages
on pull requests and deploys the static export when homepage or report changes
reach `main`. A maintainer selects **Settings → Pages → Build and deployment →
Source → GitHub Actions** once. After a successful deployment, the
[project homepage](https://guanyu-tech.github.io/AsterMind/) is available in
Chinese and [English](https://guanyu-tech.github.io/AsterMind/en/).

## Brand assets and typography

- Logo: unchanged approved `guanyu-standard-black.png` from the user-supplied
  GUANYU PPT skill v1.0.2, used at the user's request. SHA-256:
  `0589aee96ee060e09cf329a5709a2f37794ce24b49fb0bc37ffb9add0805e02f`.
- The adjacent AsterMind mark uses the supplied blue-purple transparent PNG.
  The AsterMind wordmark scales with the Chinese lettering in the GUANYU artwork.
  The English header uses the company name GUANYU in text.
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

The homepage introduces the project and provides download and installation
instructions. The diagrams illustrate request stages. Repository examples are starting points, not performance claims.
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
