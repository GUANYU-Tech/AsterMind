import {createTimeClock} from '/assets/time_clock.js';
const $=id=>document.getElementById(id),adapter=window.runtimeContinuousAdapter;
const player=createTimeClock({render:time=>adapter.at(time),onStop:()=>$('continuousPlay').textContent='连续播放',onError:e=>$('status').textContent='连续播放暂停：'+e.message});
window.runtimeTimePlayer=player;
$('continuousPlay').onclick=()=>{
 if(player.running){player.stop();return}adapter.stopEvents();
 const end=adapter.end(),start=adapter.current();player.start(start>=end?0:start,end,Number($('timeRate').value));$('continuousPlay').textContent='暂停连续播放';
};
$('timeRate').onchange=()=>player.stop();
$('seekPlaybackTime').onclick=async()=>{
 player.stop();adapter.stopEvents();const time=Number($('playbackTime').value);
 if(!Number.isFinite(time)||time<0||time>adapter.end()){$('status').textContent='定位时刻超出仿真时域';return}
 try{await adapter.at(time)}catch(e){$('status').textContent=e.message}
};
