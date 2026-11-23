import {phrases, patterns} from './language-en-catalog.js';

const han = /\p{Script=Han}/u;
const escape = source => source.replace(/[.*+?^$\{\}()|[\]\\]/g,'\\$&');
const rules = patterns.filter(([source]) => !/\}\{/.test(source)).map(([source,target]) => {
  const indexes = [];
  const parts = source.split(/(\{\d+\})/).map(part => {
    if (!/^\{\d+\}$/.test(part)) return escape(part);
    indexes.push(Number(part.slice(1,-1)));
    return '(.+?)';
  });
  return [new RegExp('^'+parts.join('')+'$','u'),indexes,target];
});

function lookup(source, depth) {
  if (!han.test(source)) return source;
  if (Object.hasOwn(phrases,source)) return phrases[source];
  if (depth > 4 || source.length > 1000) return source;
  for (const [rule,indexes,target] of rules) {
    const match = rule.exec(source);
    if (!match) continue;
    const values = new Map(indexes.map((index,i) => [index,match[i+1]]));
    return target.replace(/\{(\d+)\}/g,(_,index) => {
      const value = values.get(Number(index));
      return value && value !== source ? lookup(value,depth+1) : value ?? '';
    });
  }
  // Backend diagnostics may append detail from the execution environment.
  const parts = source.match(/^(.+?)[：:]\s*(\S.*)$/su);
  if (parts && Object.hasOwn(phrases,parts[1])) return phrases[parts[1]]+': '+lookup(parts[2],depth+1);
  return source;
}

export function translate(value, multiline = false) {
  if (multiline && value.includes('\n')) return value.split('\n').map(line => translate(line)).join('\n');
  const trimmed = value.trim();
  const leading = value.slice(0,value.length-value.trimStart().length);
  const trailing = value.slice(value.trimEnd().length);
  const normalized = trimmed.replace(/\s+/g,' ');
  const translated = lookup(normalized,0);
  return translated === normalized ? value : leading+translated+trailing;
}
