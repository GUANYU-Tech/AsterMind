// The module is served beside index.html, both at / and behind a stripped prefix.
// Resolve against it, not the portal's origin or a task URL's query/hash.
export function appUrl(path, moduleUrl = import.meta.url) {
  return new URL(path.replace(/^\/+/, ""), new URL(".", moduleUrl)).href;
}

export function appStorageKey(key, moduleUrl = import.meta.url) {
  const prefix = new URL(".", moduleUrl).pathname;
  return prefix === "/" ? key : `${prefix}${key}`;
}
