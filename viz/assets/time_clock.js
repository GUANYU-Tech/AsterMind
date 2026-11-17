// One pending frame at a time; I/O pauses time instead of skipping events.
export function createTimeClock({render,onError,onStop,request=requestAnimationFrame,cancel=cancelAnimationFrame}){
 let running=false,handle=null,last=null,time=0,end=0,rate=1,generation=0;
 async function tick(wall){
  if(!running)return;const own=generation,delta=last==null?0:Math.max(0,Math.min(.1,(wall-last)/1000));last=wall;
  const next=Math.min(end,time+delta*rate);
  try{await render(next);if(own!==generation)return;time=next;if(time>=end){stop();return}handle=request(tick)}
  catch(error){if(own===generation){stop();onError?.(error)}}
 }
 function stop(){running=false;generation++;if(handle!=null)cancel(handle);handle=null;last=null;onStop?.()}
 return {start(start,finish,speed=1){stop();time=start;end=finish;rate=speed;running=true;last=null;handle=request(tick)},stop,
  get running(){return running},get time(){return time}};
}
