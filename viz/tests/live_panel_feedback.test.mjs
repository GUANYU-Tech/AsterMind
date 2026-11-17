import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

test('background calculation never mounts a popup over the interactive globe',()=>{
 class Element {
  constructor(tag){this.tag=tag;this.children=[];this.className='';this.attributes={};this.style={setProperty(){}}}
  append(...children){this.children.push(...children)}
  setAttribute(name,value){this.attributes[name]=value}
  remove(){this.removed=true}
 }
 const document={getElementById:()=>true,createElement:tag=>new Element(tag)};
 const context={document,TRAFFIC_LEGEND:[['queued','排队中'],['transmitting','发送中']],TRAFFIC_PALETTE:{queued:1,transmitting:2}};
 vm.createContext(context);
 const source=readFileSync(new URL('../assets/live_panel_feedback.js',import.meta.url),'utf8')
  .replace(/^import .*;\r?\n/gm,'').replace('export function attachLivePanelFeedback','function attachLivePanelFeedback');
 vm.runInContext(source,context);
 const globe=new Element('div'),feedback=context.attachLivePanelFeedback(globe);
 for(const phase of ['preparing','computing','next',null,'computing'])feedback.update(phase&&{phase,title:'后台计算中',detail:'等待通信'});
 assert.equal(globe.children.length,1);
 assert.equal(globe.children[0].className,'live-traffic-legend');
 assert.equal(globe.children[0].attributes['aria-label'],'传输状态图例');
 assert.equal(globe.attributes['aria-busy'],undefined);
 feedback.dispose();assert.equal(globe.children[0].removed,true);
});
