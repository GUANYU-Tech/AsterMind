const focus=new URLSearchParams(location.search).get('focus');
document.documentElement.classList.add('analysis-embedded');
if(['operators','kv'].includes(focus))document.documentElement.classList.add('analysis-'+focus);
window.addEventListener('message',event=>{if(event.origin!==location.origin||event.data?.type!=='analysis-visibility'||event.data.visible)return;const play=document.getElementById('play');if(play&&/暂停/.test(play.textContent))play.click()});
