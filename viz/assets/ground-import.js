const required=['ingress_id','lat_deg','lon_deg'];
export const groundDefaults={altitude_km:0,min_elevation_deg:10,terminal_count:1,ground_kind:'gateway'};
const columns=[...required,...Object.keys(groundDefaults)];
const decimal=/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i;

// Station files use one record per line. Quoted fields may contain commas and doubled quotes.
function fields(line){
  const pattern=/(?:[ \t]*"((?:[^"]|"")*)"[ \t]*|([^",]*))(,|$)/y;
  const values=[];
  let match;
  do{
    match=pattern.exec(line);
    if(!match)throw Error('CSV 引号或分隔符无效');
    values.push((match[1]?.replaceAll('""','"')??match[2]).trim());
  }while(match[3]===',');
  return values;
}

function station(values){
  const row={...groundDefaults,...values};
  if(!row.ingress_id)throw Error('ingress_id 不能为空');
  for(const key of ['lat_deg','lon_deg','altitude_km','min_elevation_deg','terminal_count']){
    if(!decimal.test(row[key])||!Number.isFinite(Number(row[key])))throw Error(key+' 需要有限数值');
    row[key]=Number(row[key]);
  }
  if(Math.abs(row.lat_deg)>90)throw Error('lat_deg 范围为 -90 到 90');
  if(Math.abs(row.lon_deg)>180)throw Error('lon_deg 范围为 -180 到 180');
  if(row.min_elevation_deg<0||row.min_elevation_deg>90)throw Error('min_elevation_deg 范围为 0 到 90');
  if(!Number.isSafeInteger(row.terminal_count)||row.terminal_count<0)throw Error('terminal_count 需要非负整数');
  if(!['gateway','user'].includes(row.ground_kind))throw Error('ground_kind 请填写 gateway 或 user');
  return row;
}

export function parseGroundCsv(text,existing){
  const ids=new Set(existing.map(row=>row.ingress_id));
  const rows=[];
  let headers;
  const lines=text.replace(/^\uFEFF/,'').split(/\r\n|\n|\r/);
  for(const [index,line] of lines.entries()){
    if(!line.trim())continue;
    try{
      const cells=fields(line);
      if(!headers){
        if(new Set(cells).size!==cells.length)throw Error('CSV 表头含重复列');
        const missing=required.filter(key=>!cells.includes(key));
        if(missing.length)throw Error('缺少必填列 '+missing.join(', '));
        const unknown=cells.filter(key=>!columns.includes(key));
        if(unknown.length)throw Error('无法识别的列 '+unknown.join(', '));
        headers=cells;
        continue;
      }
      if(cells.length!==headers.length)throw Error('列数与表头不一致');
      const values=Object.fromEntries(headers.map((key,i)=>[key,cells[i]]).filter(([,value])=>value!==''));
      const row=station(values);
      if(ids.has(row.ingress_id))throw Error('地面站 ID 重复 '+row.ingress_id);
      ids.add(row.ingress_id);
      rows.push(row);
    }catch(error){throw Error('第 '+(index+1)+' 行：'+error.message)}
  }
  if(!rows.length)throw Error('CSV 中没有地面站数据');
  return rows;
}

export function initGroundImport(append){
  const file=document.getElementById('groundFile'),button=document.getElementById('importGrounds');
  const status=document.getElementById('groundImportStatus');
  const sample=[columns.join(','),'demo-gateway,31.23,121.47,0.01,10,2,gateway','demo-user,30.57,104.07,0.50,10,1,user'];
  document.getElementById('groundTemplate').href='data:text/csv;charset=utf-8,'+encodeURIComponent('\uFEFF'+sample.join('\r\n')+'\r\n');
  file.onchange=()=>{button.disabled=!file.files.length;status.textContent=''};
  button.onclick=async()=>{
    const selected=file.files[0];
    if(!selected||button.disabled)return;
    file.disabled=button.disabled=true;
    try{
      const bytes=await selected.arrayBuffer();
      let content;
      try{content=new TextDecoder('utf-8',{fatal:true}).decode(bytes)}
      catch{throw Error('CSV 文件须使用 UTF-8 编码')}
      const count=append(content);
      file.value='';file.dispatchEvent(new Event('change'));
      status.textContent='已添加 '+count+' 个地面站。';
    }catch(error){status.textContent=error.message}
    finally{file.disabled=false;button.disabled=!file.files.length}
  };
}
