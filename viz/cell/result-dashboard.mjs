const ns = 'http://www.w3.org/2000/svg';
const renders = new WeakMap();
const colors = {compute:'#8bcac7',communication:'#d6a27c',memory:'#b3a1d7',waiting:'#444b50',success:'#8bcac7',failed:'#dd8982'};
const kindNames = {GEMM:'矩阵计算',VECTOR:'向量计算',FLASH_ATTN:'Attention',KV_CACHE:'KV 访问',COMMUNICATION:'集合通信'};
const fmt = value => Number.isFinite(value) ? value.toLocaleString('zh-CN',{maximumFractionDigits:3}) : '—';
const finite = value => typeof value==='number' && Number.isFinite(value);
export function time(value) {
  if(!finite(value)) return '—';
  const [scale,unit]=value>=1e9?[1e9,'s']:value>=1e6?[1e6,'ms']:value>=1e3?[1e3,'μs']:[1,'ns'];
  return `${fmt(value/scale)} ${unit}`;
}
export function bytes(value) {
  if(!finite(value)) return '—';
  const exponent=Math.min(4,Math.max(0,Math.floor(Math.log2(Math.max(1,value))/10)));
  return `${fmt(value/1024**exponent)} ${['B','KiB','MiB','GiB','TiB'][exponent]}`;
}
const pct=value=>finite(value)?`${fmt(value*100)}%`:'—';
function el(tag,text,className) {
  const node=document.createElement(tag); if(text!==undefined) node.textContent=text;
  if(className) node.className=className; return node;
}
function svg(tag,attributes={},text) {
  const node=document.createElementNS(ns,tag);
  for(const [key,value] of Object.entries(attributes)) node.setAttribute(key,String(value));
  if(text!==undefined) node.textContent=text; return node;
}
function icon(kind) {
  const paths={time:'M12 8v4l3 2 M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18',
    request:'M7 3h8l4 4v14H5V3h2 M9 12l2 2 4-4',
    chip:'M7 7h10v10H7z M3 9h4 M3 15h4 M17 9h4 M17 15h4 M9 3v4 M15 3v4 M9 17v4 M15 17v4',
    network:'M5 5h4v4H5z M15 15h4v4h-4z M15 5h4v4h-4z M9 7h6 M7 9v8h8',
    memory:'M4 6h16v12H4z M8 10v4 M12 10v4 M16 10v4 M8 18v3 M16 18v3',
    arrow:'M5 12h14 M13 6l6 6-6 6',chart:'M4 4v16h16 M8 15l4-5 4 2 4-6',
    check:'M5 12l4 4L19 6'};
  const image=svg('svg',{viewBox:'0 0 24 24','aria-hidden':'true',fill:'none',stroke:'currentColor','stroke-width':1.5,'stroke-linecap':'round','stroke-linejoin':'round'});
  image.append(svg('path',{d:paths[kind]||paths.chart})); return image;
}
function action(text,run,className='report-button') { const node=el('button',text,className); node.type='button'; node.onclick=run; return node; }
function block(title,subtitle,extra) {
  const box=el('section',undefined,'report-block'), header=el('header'), name=el('div');
  name.append(el('h3',title)); if(subtitle) name.append(el('p',subtitle)); header.append(name); if(extra) header.append(extra); box.append(header); return box;
}
function empty(title,detail) { const box=el('div',undefined,'report-empty'); box.append(icon('chart'),el('strong',title),el('p',detail)); return box; }
function note(text) { return el('p',text,'report-note'); }
function definition(title,text) { const node=el('details',undefined,'report-definition'); node.append(el('summary',title),note(text)); return node; }
function stat(label,value,detail,kind='chart') {
  const card=el('div',undefined,'report-stat'), heading=el('div',undefined,'report-stat-label');
  heading.append(icon(kind),el('span',label)); card.append(heading,el('strong',value)); if(detail) card.append(el('small',detail)); return card;
}
function stats(items) { const group=el('div',undefined,'report-stats'); items.forEach(args=>group.append(stat(...args))); return group; }
function table(columns,rows,{click,selected}={}) {
  const frame=el('div',undefined,'report-table-scroll'), node=el('table',undefined,'report-table'), head=el('thead'), header=el('tr'), body=el('tbody');
  columns.forEach(title=>header.append(el('th',title))); head.append(header);
  rows.forEach((row,index)=>{
    const tr=el('tr'); row.forEach(value=>{ const td=el('td'); td.append(value instanceof Node?value:document.createTextNode(String(value??'—'))); tr.append(td); });
    if(click) { tr.onclick=()=>click(index); const button=tr.querySelector('button'); if(button) button.onclick=event=>{event.stopPropagation();click(index);}; tr.classList.toggle('is-selected',index===selected); }
    body.append(tr);
  });
  node.append(head,body); frame.append(node); return frame;
}
function label(main,sub) { const box=el('span',undefined,'report-cell-label'); box.append(el('strong',main)); if(sub) box.append(el('small',sub)); return box; }
function barList(items,format=fmt) {
  const list=el('div',undefined,'report-bars'), maximum=Math.max(0,...items.map(item=>item.value||0));
  for(const item of items) {
    const row=el('div',undefined,'report-bar-row'), name=el('span',item.label), track=el('i'), fill=el('i'), value=el('strong',format(item.value));
    fill.style.width=`${maximum?Math.max(.5,(item.value||0)/maximum*100):0}%`; fill.style.background=item.color||colors.compute; track.append(fill);
    row.append(name,track,value); if(item.detail) row.title=item.detail; list.append(row);
  }
  return list;
}
function chart(series,{label='',xFormat=time,yFormat=fmt,step=false}={}) {
  // A start and a completion at the same instant have no intervening duration.
  // Keep the post-event value, including reports exported by older adapters.
  if(step) series=series.map(item=>{const byTime=new Map(item.points.map(point=>[point[0],point]));return {...item,points:[...byTime.values()]};});
  const frame=el('div',undefined,'report-chart'), image=svg('svg',{viewBox:'0 0 800 228',role:'img','aria-label':label});
  const points=series.flatMap(item=>item.points).filter(point=>finite(point[0])&&finite(point[1]));
  if(!points.length) return empty('暂无曲线数据','此任务没有记录对应的采样点。');
  const xmin=points.reduce((value,point)=>Math.min(value,point[0]),Infinity), xmax=points.reduce((value,point)=>Math.max(value,point[0]),xmin+1), ymax=points.reduce((value,point)=>Math.max(value,point[1]),1e-12);
  const x=value=>66+(value-xmin)/(xmax-xmin)*710, y=value=>188-value/ymax*158;
  for(let tick=0;tick<=4;tick++) {
    const yy=188-tick/4*158;
    image.append(svg('line',{x1:66,x2:776,y1:yy,y2:yy,class:'report-grid-line'}),svg('text',{x:54,y:yy+4,'text-anchor':'end',class:'report-axis-label'},yFormat(ymax*tick/4)));
    const value=xmin+(xmax-xmin)*tick/4;
    image.append(svg('text',{x:x(value),y:214,'text-anchor':tick===0?'start':tick===4?'end':'middle',class:'report-axis-label'},xFormat(value)));
  }
  series.forEach((item,index)=>{
    const samples=item.points.filter(point=>finite(point[0])&&finite(point[1])); if(!samples.length) return;
    const color=item.color||colors.compute;
    let d=''; samples.forEach((point,i)=>{d+=i?(step?`H${x(point[0])}V${y(point[1])}`:`L${x(point[0])},${y(point[1])}`):`M${x(point[0])},${y(point[1])}`;});
    if(series.length===1) image.append(svg('path',{d:`${d}L${x(samples.at(-1)[0])},188L${x(samples[0][0])},188Z`,fill:color,opacity:.08}));
    image.append(svg('path',{d,fill:'none',stroke:color,'stroke-width':2,'stroke-linejoin':'round'}));
    const markerSamples=samples.length<=160?samples:samples.filter((_,i)=>i%Math.ceil(samples.length/160)===0);
    markerSamples.forEach(point=>{const marker=svg('circle',{cx:x(point[0]),cy:y(point[1]),r:samples.length<12?4:2.5,fill:color,tabindex:0,class:'report-chart-point','aria-label':`${item.label} ${xFormat(point[0])}：${yFormat(point[1])}`}); marker.append(svg('title',{},`${point[2]||xFormat(point[0])} ${item.label} ${yFormat(point[1])}`)); image.append(marker);});
  });
  frame.append(image); const legend=el('div',undefined,'report-legend');
  series.forEach(item=>{const key=el('span'), dot=el('i'); dot.style.background=item.color||colors.compute; key.append(dot,document.createTextNode(item.label)); legend.append(key);}); frame.append(legend); return frame;
}
function average(rows,key) { const values=rows.map(row=>row[key]).filter(finite); return values.length?values.reduce((a,b)=>a+b,0)/values.length:null; }
function addTabs(container) {
  const tabs=el('div',undefined,'report-tabs'), pages=el('div',undefined,'report-pages'), panels=[];
  tabs.setAttribute('role','tablist'); tabs.setAttribute('aria-label','仿真结果栏目'); container.append(tabs,pages);
  return (title,kind)=>{
    const index=panels.length, button=action('',()=>select(index),'report-tab'), panel=el('section',undefined,'cell-result-panel');
    button.append(icon(kind),el('span',title)); button.id=`cell-report-tab-${index}`; panel.id=`cell-report-panel-${index}`;
    button.setAttribute('role','tab'); button.setAttribute('aria-controls',panel.id); panel.setAttribute('role','tabpanel'); panel.setAttribute('aria-labelledby',button.id); panel.append(el('h2',title,'report-panel-title'));
    function select(next) { panels.forEach((item,i)=>{item.panel.hidden=i!==next; item.button.setAttribute('aria-selected',String(i===next)); item.button.tabIndex=i===next?0:-1;}); }
    button.onkeydown=event=>{const shifts={ArrowRight:1,ArrowLeft:-1,Home:-index,End:panels.length-1-index}; if(Object.hasOwn(shifts,event.key)) { event.preventDefault(); const next=(index+shifts[event.key]+panels.length)%panels.length; select(next); panels[next].button.focus(); }};
    panels.push({button,panel}); tabs.append(button); pages.append(panel); select(0); panel.open=()=>select(index); return panel;
  };
}
function overview(data,jumpRank,jumpOperators) {
  const page=el('div',undefined,'report-overview'), grid=el('div',undefined,'report-overview-grid');
  const critical=data.ranks.find(row=>row.critical);
  const breakdown=block('关键 Rank 时间分解','计算、通信与间隔按真实执行区间重建',critical?el('span',`RANK ${critical.rank}`,'report-tag'):null);
  if(critical) {
    const segments=[['计算与 KV 访问',critical.compute_ns,colors.compute],['未重叠通信',critical.exposed_communication_ns,colors.communication],['等待 / 空隙',critical.waiting_ns,colors.waiting]];
    const total=critical.wall_ns||1, stack=el('div',undefined,'report-time-stack');
    for(const [name,value,color] of segments) {const part=el('span'); part.style.width=`${value/total*100}%`; part.style.background=color; part.title=`${name}：${time(value)}`; stack.append(part);}
    breakdown.append(el('div',time(critical.wall_ns),'report-block-value'),stack);
    const legend=el('div',undefined,'report-time-legend');
    segments.forEach(([name,value,color])=>{const row=el('div'),dot=el('i'); dot.style.background=color; row.append(dot,el('span',name),el('strong',time(value)),el('small',pct(value/total))); legend.append(row);});
    breakdown.append(legend,definition('查看统计口径','按原始 call 所属 Rank 取区间并集。计算包含 KV 访问；重叠区间只计一次。等待 / 空隙为首个至末个 call 之间未被记录区间覆盖的部分，不推断其具体原因。'));
  } else breakdown.append(empty('没有 Rank 执行记录','此任务未生成可绘制的算子区间。'));
  const completion=block('请求完成情况','按任务最终状态统计'), ring=el('div',undefined,'report-completion-ring');
  const ratio=data.requests?data.completed/data.requests:0, image=svg('svg',{viewBox:'0 0 140 140',role:'img','aria-label':`完成 ${data.completed} / ${data.requests} 个请求`});
  image.append(svg('circle',{cx:70,cy:70,r:53,fill:'none',stroke:'#2d3032','stroke-width':9}),svg('circle',{cx:70,cy:70,r:53,fill:'none',stroke:colors.compute,'stroke-width':9,'stroke-dasharray':`${ratio*333.01} 333.01`,transform:'rotate(-90 70 70)','stroke-linecap':'round'}));
  image.append(svg('text',{x:70,y:72,'text-anchor':'middle',class:'report-ring-number'},pct(ratio)),svg('text',{x:70,y:94,'text-anchor':'middle',class:'report-axis-label'},'完成率'));
  const counts=el('div'); counts.append(label(fmt(data.completed),'已完成'),label(fmt(data.failed),'失败'),label(fmt(Math.max(0,data.requests-data.completed-data.failed)),'其他状态')); ring.append(image,counts); completion.append(ring);
  const satellites=new Map(); for(const row of data.ranks) satellites.set(row.satellite,Math.max(satellites.get(row.satellite)||0,row.end_ns));
  const distribution=block('算力星完成分布','各卫星已记录算子的最晚结束时刻',action('查看明细 →',jumpRank,'report-block-link'));
  distribution.append(barList([...satellites].map(([id,value])=>({label:id,value})),time));
  const operators=block('算子构成','按调用数量统计',action('查看算子 →',jumpOperators,'report-block-link')); operators.append(barList([...data.operator_types].sort((a,b)=>b.count-a.count).map(row=>({label:kindNames[row.kind]||row.kind,value:row.count,color:row.kind==='COMMUNICATION'?colors.communication:row.kind==='KV_CACHE'?colors.memory:colors.compute})),fmt));
  grid.append(breakdown,completion,distribution,operators); page.append(grid); return page;
}
function requestPage(data) {
  const page=el('div'), requests=data.request_rows;
  page.append(stats([['平均首 Token 延迟',time(average(requests,'ttft_ns')),'TTFT 有效请求均值','time'],['平均 Token 间隔',time(average(requests,'tpot_ns')),'TPOT 有效请求均值','time'],['平均端到端延迟',time(average(requests,'latency_ns')),'E2E 有效请求均值','chart'],['请求吞吐',`${fmt(data.throughput_rps)} req/s`,'完成请求数 ÷ 首次到达至最终完成','request']]));
  const box=block('请求延迟对比',`以请求序号排列 ${requests.length>200?'图中显示前 200 个请求':'悬停或聚焦节点查看数值'}`), samples=requests.slice(0,200);
  box.append(chart([{label:'端到端延迟',points:samples.map((r,i)=>[i+1,r.latency_ns,r.id]),color:colors.compute},{label:'首 Token 延迟',points:samples.map((r,i)=>[i+1,r.ttft_ns,r.id]),color:colors.memory}],{label:'请求端到端与首 Token 延迟曲线',xFormat:value=>Number.isInteger(value)?fmt(value):'',yFormat:time}));
  const rows=el('div'), footer=el('div',undefined,'report-pagination'); let offset=0;
  const previous=action('上一页',()=>{offset-=25;render();}), next=action('下一页',()=>{offset+=25;render();}), status=el('span');
  function render() {rows.replaceChildren(table(['请求','状态','TTFT','TPOT','端到端延迟','输出 Token'],requests.slice(offset,offset+25).map(r=>[r.id,r.status==='completed'?'已完成':r.status==='failed'?'失败':r.status,time(r.ttft_ns),time(r.tpot_ns),time(r.latency_ns),fmt(r.output_tokens)]))); previous.disabled=offset===0; next.disabled=offset+25>=requests.length; status.textContent=`${requests.length?offset+1:0}–${Math.min(offset+25,requests.length)} / ${requests.length} 个请求`;}
  render(); footer.append(status,previous,next); page.append(box,rows,footer); return page;
}
function selector(label,options,change) { const wrap=el('label',undefined,'report-select-label'), input=el('select'); wrap.append(el('span',label),input); input.setAttribute('aria-label',label); options.forEach(([value,title])=>{const option=el('option',title); option.value=value; input.append(option);}); input.onchange=()=>change(input.value); return {wrap,input}; }
function rankPage(data,load) {
  const page=el('div'), rows=el('div'), timeline=block('Rank 执行时间线','选择一行，查看该 Rank 的计算、KV 与通信区间'), chartArea=el('div'), tools=el('div',undefined,'report-pagination');
  let selected=data.ranks.find(row=>row.critical)?.id||data.ranks[0]?.id, version=0, offset=0;
  const title=el('span'), next=action('后续区间',()=>{offset+=500;draw();}), previous=action('前序区间',()=>{offset=Math.max(0,offset-500);draw();});
  tools.append(title,previous,next); timeline.append(chartArea,tools);
  function render() {
    rows.replaceChildren(table(['Rank / 算力星','执行跨度','计算 / KV','通信','重叠','调用数','权重驻留'],data.ranks.map(row=>{
      const button=action('',()=>{},'report-rank-button'); button.append(label(`RANK ${row.rank}`,`${row.satellite} 卡 ${row.local_index??'—'}`)); button.setAttribute('aria-pressed',String(selected===row.id));
      return [button,time(row.wall_ns),time(row.compute_ns),time(row.communication_ns),time(row.overlap_ns),fmt(row.calls),bytes(row.weight_bytes)];
    }),{click:index=>{selected=data.ranks[index].id;offset=0;render();draw();},selected:data.ranks.findIndex(row=>row.id===selected)}));
  }
  async function draw() {
    if(!selected) return; const current=++version; chartArea.textContent='正在读取时间线…';
    try { const result=await load(`section=timeline&rank=${encodeURIComponent(selected)}&offset=${offset}`); if(current!==version) return;
      const image=svg('svg',{viewBox:'0 0 900 208',role:'img','aria-label':`${selected} 执行时间线`}), max=Math.max(1,result.end_ns-result.start_ns), x=value=>100+(value-result.start_ns)/max*770;
      [['compute','计算',colors.compute],['memory','KV 访问',colors.memory],['communication','通信',colors.communication]].forEach(([key,name,color],index)=>{
        const y=34+index*48; image.append(svg('text',{x:12,y:y+19,class:'report-axis-label'},name),svg('line',{x1:100,x2:870,y1:y+13,y2:y+13,class:'report-grid-line'}));
        for(const [start,end] of result.lanes[key].items) {const bar=svg('rect',{x:x(start),y,width:Math.max(.8,x(end)-x(start)),height:25,rx:2,fill:color,tabindex:0,'aria-label':`${name} ${time(start)} 至 ${time(end)}`}); bar.append(svg('title',{},`${name} ${time(start)} → ${time(end)} 时长 ${time(end-start)}`)); image.append(bar);}
      });
      for(let i=0;i<=4;i++) image.append(svg('text',{x:100+i/4*770,y:196,'text-anchor':i===0?'start':i===4?'end':'middle',class:'report-axis-label'},time(result.start_ns+max*i/4)));
      chartArea.replaceChildren(image); title.textContent=`RANK ${selected.split('/').at(-1)} ${offset?`从第 ${offset+1} 段起`:'完整执行范围'}${result.has_more?' 本页每类最多 500 段':''}`; previous.disabled=offset===0; next.disabled=!result.has_more;
    } catch(error) {if(current===version) chartArea.replaceChildren(empty('时间线读取失败',error.message));}
  }
  page.append(rows,timeline,definition('时间线口径','同类相邻或重叠区间合并显示。通信按原始 call 的 Rank 归属展示；同一次集合通信不复制到未记录该 call 的 Rank。区间空隙不视为 CPU 实测。')); render(); draw(); return page;
}
function operatorPage(data,load) {
  const page=el('div'), top=block('算子类型分布','累计调用时长 并行或重叠区间会分别计入，不等同于任务总时间');
  top.append(barList(data.operator_types.map(row=>({label:kindNames[row.kind]||row.kind,value:row.duration_ns,color:row.kind==='COMMUNICATION'?colors.communication:row.kind==='KV_CACHE'?colors.memory:colors.compute})),time));
  const controls=el('div',undefined,'report-filters'), rows=el('div'), footer=el('div',undefined,'report-pagination'); let offset=0,kind='',rank='',search='',version=0,timer;
  const types=selector('算子类型',[['','全部类型'],...data.operator_types.map(row=>[row.kind,kindNames[row.kind]||row.kind])],value=>{kind=value;offset=0;render();});
  const ranks=selector('算子 Rank',[['','全部 Rank'],...data.ranks.map(row=>[row.id,`RANK ${row.rank} ${row.satellite}`])],value=>{rank=value;offset=0;render();});
  const input=el('input'); input.type='search';input.placeholder='搜索算子名称';input.setAttribute('aria-label','搜索算子名称'); input.oninput=()=>{clearTimeout(timer);timer=setTimeout(()=>{search=input.value;offset=0;render();},250);};
  controls.append(types.wrap,ranks.wrap,input);
  const previous=action('上一页',()=>{offset-=25;render();}),next=action('下一页',()=>{offset+=25;render();}),status=el('span'); footer.append(status,previous,next);
  async function render() {const current=++version;previous.disabled=next.disabled=true;
    try {const result=await load(`section=operators&offset=${offset}&kind=${encodeURIComponent(kind)}&rank=${encodeURIComponent(rank)}&search=${encodeURIComponent(search)}`);if(current!==version)return;
      rows.replaceChildren(result.items.length?table(['算子 / 调度步骤','类型','Rank','开始','结束','时长'],result.items.map(row=>[label(row.name,row.step),kindNames[row.kind]||row.kind,`R${row.rank}`,time(row.start_ns),time(row.end_ns),time(row.duration_ns)])):empty('没有匹配的算子','尝试切换类型、Rank 或清空搜索。'));
      status.textContent=`${result.total?offset+1:0}–${Math.min(offset+25,result.total)} / ${fmt(result.total)} 次调用`;previous.disabled=offset===0;next.disabled=!result.has_more;
    }catch(error){if(current===version)rows.replaceChildren(empty('算子读取失败',error.message));}
  }
  page.append(top,controls,rows,footer);render();return page;
}
function networkPage(data) {
  const page=el('div'), network=data.network;
  page.append(stats([['路径传输尝试',fmt(network.attempts),'Runtime Comm 准入记录','network'],['路径传输量',bytes(network.application_bytes),'服务记录中的应用字节','network'],['峰值并发流',fmt(network.peak_active),'运行时统计','chart'],['重试次数',fmt(network.retries),'运行时统计','network']]));
  const grid=el('div',undefined,'report-equal-grid'), concurrency=block('路径传输并发','按准入与完成时刻绘制'), bandwidth=block('端点间平均服务速率','应用字节 ÷ 服务区间累计时长 Gbps');
  if(network.attempts) {concurrency.append(chart([{label:'活跃传输',points:network.concurrency,color:colors.communication}],{label:'路径传输并发时间线',step:true,yFormat:value=>Number.isInteger(value)?fmt(value):''}));bandwidth.append(barList(network.paths.slice(0,6).map(row=>({label:`${row.source} → ${row.target}`,value:row.service_gbps,color:colors.communication})),value=>finite(value)?`${fmt(value)} Gbps`:'—'));}
  else {concurrency.append(empty('没有路径传输记录',network.intra_mode==='IntraSatelliteRoofline'?'本次通信在星内完成，采用 Roofline 模型。':'本次运行未记录路径传输。'));bandwidth.append(empty('没有可绘制的服务速率','星内 Roofline 通信不产生对应的路径服务记录。'));}
  grid.append(concurrency,bandwidth);page.append(grid);
  if(network.paths.length) page.append(table(['传输端点','尝试次数','应用传输量','平均服务速率'],network.paths.map(row=>[label(row.source,`→ ${row.target}`),fmt(row.attempts),bytes(row.bytes),finite(row.service_gbps)?`${fmt(row.service_gbps)} Gbps`:'—'])));
  page.append(definition('通信统计口径','应用传输量累计实际 service 记录，包含集合通信及其重试，不等于用户业务净荷；不按多跳路径重复累计。平均服务速率按服务区间时长加权，不是包含传播和等待的端到端速率。星内 Roofline 估算不包含在路径传输量中。'));return page;
}
function memoryPage(data) {
  const page=el('div'), memory=data.memory.filter(item=>item.available), rankMemory=data.ranks.filter(row=>finite(row.weight_bytes));
  const box=block('KV 缓存占用曲线','按调度步骤结束时的 HBM 分配 / 释放事件重建');
  box.append(memory.length?chart(memory.map((item,index)=>({label:item.cell,points:item.samples,color:index?colors.memory:colors.compute})),{label:'KV 缓存占用时间线',step:true,yFormat:bytes}):empty('没有 KV 事件记录','此任务没有可重建的 KV 缓存占用曲线。'));
  page.append(stats([['KV 事件峰值',memory.length?bytes(Math.max(...memory.map(item=>item.peak_kv_bytes))):'—','按单 Cell HBM 事件累计','memory'],['分配事件',fmt(memory.reduce((sum,item)=>sum+item.allocation_events,0)),'HBM allocate','memory'],['释放事件',fmt(memory.reduce((sum,item)=>sum+item.release_events,0)),'HBM release','memory'],['已记录 Rank',fmt(rankMemory.length),'提供权重驻留数据的 Rank','chip']]),box);
  const residency=block('Rank 内存配置','权重驻留与预留 KV 池 不是实测 HBM 峰值');
  residency.append(rankMemory.length?table(['Rank / 算力星','权重驻留','预留 KV 池'],rankMemory.map(row=>[label(`RANK ${row.rank}`,row.satellite),bytes(row.weight_bytes),bytes(row.reserved_kv_bytes)])):empty('没有内存配置记录','当前报告未提供 Rank 级驻留信息。'));
  page.append(residency,definition('与原版 Tensor 展示的差异','当前 V1 报告不包含完整临时 Tensor 生命周期和全量 HBM 峰值，不能恢复为实测 Tensor 热图。这里使用真实 KV 事件和权重驻留记录；KV 曲线只显示步骤末状态，事件峰值另行统计。显式配置的 KV 池容量不代表硬件自动探测容量。'));return page;
}
export async function renderDashboard(container,jobId,result,api,renderArtifacts) {
  const render=Symbol(); renders.set(container,render);
  container.classList.add('cell-results'); container.replaceChildren();
  const context=el('div',undefined,'report-context'), status=el('span',result.status==='completed'?'实验已完成':'实验结果','report-status'); status.dataset.reportStatus=result.status;status.prepend(icon('check'));context.append(status,el('span',jobId,'report-job-id'));container.append(context);
  const load=query=>api(`/api/simulations/${jobId}/analysis?${query}`); let data;
  try { data=await load('section=summary'); }
  catch(error) {if(renders.get(container)!==render||context.parentNode!==container)return;container.append(empty('图表数据暂未载入',error.message),action('重新载入图表',()=>renderDashboard(container,jobId,result,api,renderArtifacts))); await renderArtifacts(container,jobId,api);return;}
  if(renders.get(container)!==render||context.parentNode!==container)return;
  const model=el('span',data.model||'模型未记录','report-model');context.insertBefore(model,context.lastChild);
  const parallel=data.parallelism[0], info=el('span',`${data.satellite_count??'—'} 颗卫星 ${data.ranks.length} 个 Rank${parallel?` TP ${parallel.tensor_parallel_size} / PP ${parallel.pipeline_parallel_size}`:''}`);context.insertBefore(info,context.lastChild);
  const hero=el('div',undefined,'report-hero'), lead=el('div',undefined,'report-lead');
  lead.append(el('span','任务完成用时'),el('strong',time(data.completion_ns)),el('small',`首次请求到达 → 最终完成 仿真时间窗 ${time(data.simulation_ns)}`));
  hero.append(lead,stat('完成请求',`${data.completed} / ${data.requests}`,`${data.failed} 个请求失败`,'request'),stat('平均首 Token 延迟',time(average(data.request_rows,'ttft_ns')),'TTFT 有效请求均值','time'),stat('算子调用',fmt(data.kernel_calls),`${fmt(data.scheduler_steps)} 个调度步骤`,'chip'));container.append(hero);
  const addTab=addTabs(container), system=addTab('System 总览','chart');
  const requests=addTab('请求性能','request');requests.append(requestPage(data));
  const rank=addTab('Rank / 算力星','chip');rank.append(rankPage(data,load));
  const operators=addTab('算子','chip');operators.append(operatorPage(data,load));
  system.append(overview(data,()=>rank.open(),()=>operators.open()));
  const network=addTab('通信','network');network.append(networkPage(data));
  const memory=addTab('内存与 KV','memory');memory.append(memoryPage(data));
  const artifacts=addTab('产物与明细','request');
  const shortcut=document.querySelector('#artifact-shortcut');
  if(shortcut){
    shortcut.hidden=false;
    shortcut.onclick=()=>{artifacts.open();artifacts.scrollIntoView({block:'nearest'});};
  }
  if(result.warnings?.length) artifacts.append(definition('报告说明',result.warnings.join('\n')));
  const raw=definition('版本与原始参数',JSON.stringify({status:result.status,simulator:result.simulator,network_backend:result.network_backend,versions:result.versions},null,2));raw.classList.add('report-raw');artifacts.append(raw);
  await renderArtifacts(artifacts,jobId,api);
}
