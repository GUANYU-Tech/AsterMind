(() => {
 const selector=document.getElementById('simulationPlatform');
 if(!selector)return;
 const original=selector.value,status=document.getElementById('platformSwitchStatus');
 if(original==='lyz'&&document.getElementById('globe')){
  // Fetch the original renderer while the original solver is preparing.
  for(const asset of ['globe.js','three.module.js','three.core.js','OrbitControls.js','cell_visual.js','cell_model.js','runtime_overlay.js','runtime_faults.js','sun_visual.js','solar_state.js']){
   const preload=document.createElement('link');preload.rel='modulepreload';preload.href='/assets/'+asset;document.head.append(preload);
  }
  const timingKey='lyz-native-launch-ms';
  document.getElementById('startNative')?.addEventListener('click',()=>sessionStorage.setItem(timingKey,String(Date.now())),true);
  const started=Number(sessionStorage.getItem(timingKey));
  if(started>0&&location.pathname.startsWith('/live/')&&Date.now()-started<120000){
   let scheduled=false;
   const check=()=>{
    const canvas=document.querySelector('canvas[aria-label="真实 Runtime 三维地球、故障与传输路径"]'),advance=document.getElementById('advance');
    if(scheduled||!canvas||!canvas.width||!canvas.height||!advance||advance.disabled)return;
    scheduled=true;observer.disconnect();
    requestAnimationFrame(()=>requestAnimationFrame(()=>{
     document.documentElement.dataset.lyzFirstFrameMs=String(Date.now()-started);
     sessionStorage.removeItem(timingKey);
    }));
   };
   const observer=new MutationObserver(check);observer.observe(document.body,{childList:true,subtree:true,attributes:true});check();
  }
 }
 const reachable=async url=>{
  const deadline=Date.now()+15000;
  while(true){
   try{await fetch(new URL('health',url),{mode:'no-cors',cache:'no-store',signal:AbortSignal.timeout(1500)});return}
   catch(error){if(Date.now()>=deadline)throw Error('平台已启动，但浏览器暂时无法连接，请重试');await new Promise(resolve=>setTimeout(resolve,250))}
  }
 };
 selector.addEventListener('change',async()=>{
  selector.disabled=true;status.textContent='正在准备仿真平台…';
  try{
   const payload={platform:selector.value};
   const response=await fetch('/simulation-platform',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});
   const result=await response.json();
   if(!response.ok)throw Error(result.error||'平台切换失败');
   // WSL can report healthy before Windows localhost forwarding is ready.
   await reachable(result.url);
   location.assign(result.url);
  }catch(error){selector.value=original;status.textContent=error.message;selector.disabled=false}
 });
})();
