import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';

const source=readFileSync(new URL('../assets/platform_selector.js',import.meta.url),'utf8');
function mount(fetch){
 const handlers={},selector={value:'v1',disabled:false,addEventListener:(event,fn)=>handlers[event]=fn};
 const status={textContent:''},navigation=[];
 vm.runInNewContext(source,{document:{getElementById:id=>id==='simulationPlatform'?selector:id==='platformSwitchStatus'?status:null},
  fetch,URL,AbortSignal,Date,location:{assign:url=>navigation.push(url)},setTimeout:callback=>callback()});
 selector.value='lyz';
 return {selector,status,navigation,change:handlers.change};
}
test('wait for Windows forwarding before leaving the current working page',async()=>{
 let attempts=0;
 const app=mount(async(url,options)=>{
  if(url==='/simulation-platform')return {ok:true,json:async()=>({url:'http://localhost:8798/'})};
  assert.equal(String(url),'http://localhost:8798/health');
  assert.equal(options.mode,'no-cors');
  assert.equal(app.navigation.length,0);
  if(++attempts<3)throw new TypeError('Failed to fetch');
  return {type:'opaque'};
 });
 await app.change();
 assert.equal(attempts,3);
 assert.deepEqual(app.navigation,['http://localhost:8798/']);
});
test('launch errors restore the original selection and keep the page usable',async()=>{
 const app=mount(async()=>({ok:false,json:async()=>({error:'平台启动失败'})}));
 await app.change();
 assert.equal(app.selector.value,'v1');
 assert.equal(app.selector.disabled,false);
 assert.equal(app.status.textContent,'平台启动失败');
 assert.deepEqual(app.navigation,[]);
});
test('preload the original renderer and report timing only after a usable canvas exists',()=>{
 const dataset={},advance={disabled:false},preloaded=[],store=new Map([['lyz-native-launch-ms','1000']]);
 let canvas=null,changed,disconnected=false;
 const selector={value:'lyz',addEventListener(){}},status={},globe={};
 class Observer{constructor(callback){changed=callback}observe(){}disconnect(){disconnected=true}}
 const context={document:{getElementById:id=>({simulationPlatform:selector,platformSwitchStatus:status,globe,advance}[id]),
  querySelector:()=>canvas,createElement:()=>({}),head:{append:link=>preloaded.push(link)},body:{},documentElement:{dataset}},
  location:{pathname:'/live/qwen3-4b-test'},sessionStorage:{getItem:key=>store.get(key),removeItem:key=>store.delete(key)},
  Date:{now:()=>6000},MutationObserver:Observer,requestAnimationFrame:callback=>callback()};
 vm.runInNewContext(source,context);
 assert.equal(preloaded[0].rel,'modulepreload');assert.equal(preloaded[0].href,'/assets/globe.js');
 assert.ok(preloaded.some(link=>link.href==='/assets/three.core.js'));
 assert.equal(dataset.lyzFirstFrameMs,undefined);assert.equal(disconnected,false);
 canvas={width:800,height:600};changed();
 assert.equal(dataset.lyzFirstFrameMs,'5000');assert.equal(disconnected,true);assert.equal(store.size,0);
});
