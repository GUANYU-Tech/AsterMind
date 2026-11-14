(function(root){
'use strict';
function sunAt(rows,time){
 let low=0,high=rows.length;while(low<high){const middle=(low+high)>>1;if(rows[middle].time_s<=time)low=middle+1;else high=middle}
 return low?rows[low-1]:null;
}
// Coordinates below are Three.js scene units (1 unit = 1000 km), where Y
function sunInterpolated(rows,time){
 const a=sunAt(rows,time);if(!a)return null;const i=rows.indexOf(a),b=rows[i+1];if(!b||b.time_s<=a.time_s)return a;
 const u=Math.max(0,Math.min(1,(time-a.time_s)/(b.time_s-a.time_s)));
 return {time_s:time,position_itrs_km:a.position_itrs_km.map((v,k)=>v+(b.position_itrs_km[k]-v)*u)};
}
// corresponds to ITRS Z. Clip a light ray at its first opaque Earth surface.
function earthEntry(origin,direction){
 const axes=[6.378137,6.356752314245,6.378137],o=origin.map((v,i)=>v/axes[i]),d=direction.map((v,i)=>v/axes[i]);
 const a=d.reduce((s,v)=>s+v*v,0),b=2*d.reduce((s,v,i)=>s+v*o[i],0),c=o.reduce((s,v)=>s+v*v,0)-1,disc=b*b-4*a*c;
 if(!a||disc<0)return null;const distances=[(-b-Math.sqrt(disc))/(2*a),(-b+Math.sqrt(disc))/(2*a)].filter(t=>t>=0);return distances.length?Math.min(...distances):null;
}
const api={sunAt,sunInterpolated,earthEntry};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.SolarVisual=api;
})(typeof window!=='undefined'?window:globalThis);
