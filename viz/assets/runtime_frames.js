
export function createFrameCache(load,limit=6){
 const values=new Map(),pending=new Map(),queue=[];let active=0;
 function pump(){
  while(active<2&&queue.length){
   const {descriptor,resolve,reject}=queue.shift(),id=descriptor.artifact;active++;
   Promise.resolve().then(()=>load(descriptor)).then(value=>{
    values.set(id,value);while(values.size>limit)values.delete(values.keys().next().value);
    resolve(value);
   },reject).finally(()=>{active--;pending.delete(id);pump()});
  }
 }
 return {get(descriptor){
  const id=descriptor.artifact;
  if(values.has(id)){const value=values.get(id);values.delete(id);values.set(id,value);return Promise.resolve(value)}
  if(pending.has(id))return pending.get(id);
  let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b});
  pending.set(id,promise);queue.push({descriptor,resolve,reject});pump();return promise;
 },size:()=>values.size};
}
