// Optional, visibly simulated recording pointer. Page input stays with Chrome.
// Enabled only by ?pointer=1 in this tab; ?pointer=0 clears the tab's mode.
const mode=new URL(location.href).searchParams.get('pointer');
if(mode==='1')sessionStorage.setItem('orbitserve.demoPointer','1');
if(mode==='0')sessionStorage.removeItem('orbitserve.demoPointer');
if(sessionStorage.getItem('orbitserve.demoPointer')==='1'){
  const style=document.createElement('style');
  style.textContent=`.demo-pointer{position:fixed;left:0;top:0;width:30px;height:38px;z-index:2147483647;pointer-events:none;filter:drop-shadow(0 2px 3px #000a);transition:transform .32s cubic-bezier(.22,.7,.25,1)}.demo-pointer svg{width:30px;height:38px}.demo-click{position:fixed;width:44px;height:44px;margin:-22px;z-index:2147483646;border:3px solid #a3ffe0;border-radius:50%;pointer-events:none;animation:demo-click .85s .25s both}.demo-pointer-label{position:fixed;right:20px;bottom:16px;z-index:2147483645;border:1px solid #4b786b;border-radius:6px;padding:6px 10px;background:#0b1a19dd;color:#b4e8d7;font:11px 'Segoe UI','Microsoft YaHei',sans-serif;pointer-events:none}@keyframes demo-click{from{opacity:1;transform:scale(.45)}to{opacity:0;transform:scale(1.8)}}`;
  document.head.append(style);
  const cursor=document.createElement('div');cursor.className='demo-pointer';cursor.setAttribute('aria-hidden','true');
  cursor.innerHTML='<svg viewBox="0 0 30 38"><path d="M3 2 L3 29 L10 23 L16 35 L21 32 L15 21 L25 20 Z" fill="#f2fff8" stroke="#0d2920" stroke-width="2"/><path d="M5 5 L5 24 L10 19 L17 21 Z" fill="#7cd8c0"/></svg>';
  const label=document.createElement('span');label.className='demo-pointer-label';label.textContent='交互演示 网页模拟指针';
  document.body.append(cursor,label);cursor.style.transform='translate(260px,140px)';
  function move(event){cursor.style.transform=`translate(${event.clientX}px,${event.clientY}px)`;}
  document.addEventListener('pointermove',move,{passive:true,capture:true});
  document.addEventListener('pointerdown',event=>{move(event);const ring=document.createElement('span');ring.className='demo-click';ring.style.left=event.clientX+'px';ring.style.top=event.clientY+'px';ring.setAttribute('aria-hidden','true');document.body.append(ring);setTimeout(()=>ring.remove(),1200);},{passive:true,capture:true});
}
