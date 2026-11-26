function listInput(host, caption) {
  const label=document.createElement('label'), input=document.createElement('input');
  label.textContent=caption;input.type='text';input.setAttribute('aria-label',caption);
  label.append(input);host.append(label);
  return input;
}

export function renderIngressFields(host, formation, changed) {
  const free=formation.egress_policy==='free';
  const members=listInput(host,'对外接入成员编号（多个成员逗号分隔）');
  const limits=listInput(host,'接入成员对外 ISL 终端数（顺序与前相同，逗号分隔）');
  const base=formation.cluster.satellites_per_cluster;
  let maximum=base, minimum=base;
  for(const row of Object.values(formation.cell_overrides||{})){
    maximum=Math.max(maximum,row.satellites_per_cluster??base);
    minimum=Math.min(minimum,row.satellites_per_cluster??base);
  }
  const all=free?Array.from({length:maximum},(_,index)=>index):[];
  members.disabled=free;members.value=(free?all:formation.ingress_members??[0]).join(',');
  limits.value=(formation.ingress_isl_terminals??[]).join(',');
  limits.placeholder='留空沿用每成员共享 ISL 终端数 '+(formation.isl_terminals??4);
  const parse=input=>input.value.split(/[,，]/).map(part=>/^\d+$/.test(part.trim())?Number(part):NaN);
  function update(commit) {
    const ids=free?all:parse(members);
    const validIds=free || (ids.every(id=>Number.isSafeInteger(id)&&id<minimum) &&
      new Set(ids).size===ids.length && (formation.egress_policy!=='single'||ids.length===1));
    members.setCustomValidity(validIds?'':'接入成员编号须唯一且在编队范围内，单星出口只允许一个成员。');
    const counts=limits.value.trim()?parse(limits):null;
    const validCounts=!counts || (counts.length===ids.length && counts.every(Number.isSafeInteger));
    limits.setCustomValidity(validCounts?'':'终端数须与接入成员一一对应，且均为非负整数。');
    if(!commit)return;
    if(validIds&&!free)formation.ingress_members=ids;
    if(validCounts && validIds){
      if(counts)formation.ingress_isl_terminals=counts;
      else delete formation.ingress_isl_terminals;
    }
    changed();
  }
  members.oninput=limits.oninput=()=>update(true);
  update(false);
}
