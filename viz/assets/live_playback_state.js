// Viewer intent survives an empty streaming buffer. Only an exhausted terminal
// record ends playback; a paused solver is resumed automatically at low water.
export const terminalState=state=>['completed','cancelled','failed'].includes(state);
// The first preparation snapshot has no version. It must not prevent the first
// versioned startup/ready snapshot from replacing it, while late polls must not
// roll a versioned state back after an advance acknowledgement.
export function acceptsLiveState(current,update){
 if(!update||update.unchanged)return false;
 return current?.version==null||(update.version!=null&&update.version>=current.version);
}
export async function requestLiveState(url,{fetcher=globalThis.fetch,timeoutMs=10000}={}){
 const response=await fetcher(url,{cache:'no-store',signal:AbortSignal.timeout(timeoutMs)});
 const value=await response.json();
 if(!response.ok)throw Error(value.error||'读取在线仿真状态失败');
 return value;
}
export function shouldAdvanceLive(state,{stream,pending,remaining,selectedComplete=false}){
 return state?.state==='paused'&&state.time_s<state.end_s&&!selectedComplete&&(stream?remaining<1.2:!pending);
}
export function liveAdvancePlan(state,{stream=false,activeTransfers=false,singleStep=false}={}){
 if(singleStep)return {mode:'event'};
 const finished=Number.isInteger(state.request_count)&&state.request_count>=0
  &&(state.completed??0)+(state.failed??0)>=state.request_count;
 // Execute the remaining native queue when business has ended. This retains
 // scheduled faults/migrations and finalization, without 300 empty HTTP batches.
 if(finished)return {mode:'time',until_s:state.end_s};
 // Event stepping processes physical boundaries internally, and pauses at the
 // next real business/fault event. An empty scene need not play empty seconds.
 return stream&&activeTransfers?{mode:'time',until_s:Math.min(state.end_s,state.time_s+2)}:{mode:'event'};
}
export function livePlaybackState(state,{playing,pending,stream,through,observeOnly=false}){
 const terminal=terminalState(state?.state);
 const drained=terminal&&!pending&&(!stream||through>=state.event_count);
 const waiting=playing&&!pending&&!drained;
 // Background look-ahead alternates paused/advancing without changing the
 // viewer's ability to pause and step. Do not expose that transport detail as
 // an enabled/disabled pulse on the Next control.
 const ready=pending||(!observeOnly&&['paused','advancing'].includes(state?.state)&&state.time_s<state.end_s);
 const label=playing?'暂停观看':drained?(state.state==='failed'?'执行失败':state.state==='cancelled'?'实验已结束':'通信已播放完'):'播放传输';
 return {drained,waiting,ready,label,canToggle:playing||ready||state?.state==='advancing'};
}
export function liveWaitingFeedback(state,{waiting,pending=false,error,observeOnly=false,playing=true}){
 if(error||state?.state==='failed')return null;
 if(!waiting&&state?.state!=='preparing'&&!(state?.state==='finalizing'&&!pending))return null;
 if(state?.state==='preparing'||!state)return {phase:'preparing',title:'正在准备场景',detail:playing?'准备完成后自动播放':'准备完成后可开始播放'};
 if(state.state==='finalizing')return {phase:'finalizing',title:'正在保存实验结果',detail:'保存完成后可查看分析'};
 if(terminalState(state.state))return {phase:'recording',title:'正在整理通信记录',detail:'记录就绪后自动续播'};
 if(observeOnly&&state.state==='paused')return {phase:'paused',title:'等待仿真继续',detail:'新数据就绪后自动续播'};
 if(state.state==='paused')return {phase:'next',title:'准备下一段通信',detail:'新数据就绪后自动续播'};
 return {phase:'computing',title:'后台计算中',detail:'新数据就绪后自动续播'};
}
