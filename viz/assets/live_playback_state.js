// Viewer intent survives an empty streaming buffer. Only an exhausted terminal
// record ends playback; a paused solver is resumed automatically at low water.
export const terminalState=state=>['completed','cancelled','failed'].includes(state);
export function shouldAdvanceLive(state,{stream,pending,remaining,selectedComplete=false}){
 return state?.state==='paused'&&state.time_s<state.end_s&&!selectedComplete&&(stream?remaining<1.2:!pending);
}
export function livePlaybackState(state,{playing,pending,stream,through,observeOnly=false}){
 const terminal=terminalState(state?.state);
 const drained=terminal&&!pending&&(!stream||through>=state.event_count);
 const waiting=playing&&!pending&&!drained;
 const ready=pending||(!observeOnly&&state?.state==='paused'&&state.time_s<state.end_s);
 const label=playing?'暂停观看':ready?'播放传输':drained?(state.state==='failed'?'执行失败':state.state==='cancelled'?'实验已结束':'通信已播放完'):'等待执行';
 return {drained,waiting,ready,label,canToggle:playing||ready||state?.state==='advancing'};
}
export function liveWaitingFeedback(state,{waiting,pending=false,error,observeOnly=false}){
 if(error||state?.state==='failed')return null;
 if(!waiting&&state?.state!=='preparing'&&!(state?.state==='finalizing'&&!pending))return null;
 if(state?.state==='preparing'||!state)return {phase:'preparing',title:'正在准备场景',detail:'准备完成后自动播放'};
 if(state.state==='finalizing')return {phase:'finalizing',title:'正在保存实验结果',detail:'保存完成后可查看分析'};
 if(terminalState(state.state))return {phase:'recording',title:'正在整理通信记录',detail:'记录就绪后自动续播'};
 if(observeOnly&&state.state==='paused')return {phase:'paused',title:'等待仿真继续',detail:'新数据就绪后自动续播'};
 if(state.state==='paused')return {phase:'next',title:'准备下一段通信',detail:'新数据就绪后自动续播'};
 return {phase:'computing',title:'后台计算中',detail:'新数据就绪后自动续播'};
}
