const {mountRecordedScene}=await import('/assets/recorded_scene.js');
await mountRecordedScene(document.querySelector('main'),location.pathname.split('/')[2]);
