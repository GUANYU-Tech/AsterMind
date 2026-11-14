if(new URLSearchParams(location.search).get('recorded')==='1'){
  const {mountRecordedScene}=await import('/assets/recorded_scene.js');
  await mountRecordedScene(document.querySelector('main'),location.pathname.split('/')[2]);
}else{
  await import('/assets/live_runtime.js');
}
