import '/assets/motion_state.js';
import {createFrameCache} from '/assets/runtime_frames.js';

export function createMotionFrames(root,descriptors){
 const cache=createFrameCache(async row=>{const r=await fetch(root+row.artifact);if(!r.ok)throw Error('运动数据读取失败');return r.json()},6);
 return {async at(time){
  const i=MotionState.heldIndex(descriptors,time);if(i<0)throw Error('运动时间超出场景');
  const [left,right]=await Promise.all([cache.get(descriptors[i]),cache.get(descriptors[Math.min(i+1,descriptors.length-1)])]);
  return {left,right};
 },size:cache.size};
}
