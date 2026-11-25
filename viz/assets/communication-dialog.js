import {field} from './studio_common.js';
import {sourceFields} from './network_options.js';
import {inspectTleShells,showTleResult} from './tle_importer.js';

export function initCommunicationDialog(readConfig,template,render){
  const $=id=>document.getElementById(id);
  const dialog=$('addCommunication'),form=dialog.querySelector('form');
  const parameters=$('shellParameters'),tle=$('tleOptions'),file=$('tleFile');
  const methods=$('shellMethods'),submit=$('confirmCommunication'),status=$('tleStatus');
  let draft,session=0;

  function updateMethod(){
    const method=form.elements.method.value;
    parameters.hidden=parameters.disabled=method!=='parameters';
    tle.hidden=tle.disabled=method!=='tle';
    submit.disabled=!method||(method==='tle'&&!file.files.length);
    status.textContent='';
  }
  function renderParameters(){
    parameters.replaceChildren();
    const fields=document.createElement('div');fields.className='fields';parameters.append(fields);
    const id=field(fields,draft,'shell_id','壳层 ID','text');id.required=true;
    sourceFields(fields,parameters,draft,false,renderParameters,undefined,true);
    const validateId=()=>{
      const value=id.value.trim();
      const duplicate=readConfig().constellations.communication_shells.some(shell=>shell.shell_id===value);
      id.setCustomValidity(value&&!duplicate?'':document.documentElement.lang==='en'?'Enter a unique shell ID.':'请填写唯一的壳层 ID');
    };
    id.addEventListener('input',validateId);validateId();
  }
  $('addShell').onclick=()=>{
    session++;
    form.reset();file.dispatchEvent(new Event('change'));
    methods.disabled=false;
    draft=structuredClone(template);
    const used=new Set(readConfig().constellations.communication_shells.map(shell=>shell.shell_id));
    let n=1;while(used.has('shell-'+n))n++;
    draft.shell_id='shell-'+n;
    renderParameters();updateMethod();dialog.showModal();
  };
  methods.onchange=updateMethod;
  file.addEventListener('change',updateMethod);
  $('cancelCommunication').onclick=()=>dialog.close();

  form.onsubmit=async event=>{
    event.preventDefault();
    if(submit.disabled)return;
    const config=readConfig(),shells=config.constellations.communication_shells;
    const first=shells.length;
    let result;
    if(form.elements.method.value==='tle'){
      const current=session,selected=file.files[0];
      methods.disabled=tle.disabled=submit.disabled=true;
      status.textContent='正在校验 TLE 并分析共同历元的轨道面和槽位…';
      try{
        result=await inspectTleShells(selected,config,template);
        if(current!==session||!dialog.open)return;
        if(config!==readConfig())throw Error('场景已切换，请重新导入。');
      }catch(error){
        if(current===session&&dialog.open){
          methods.disabled=false;updateMethod();status.textContent='导入失败：'+error.message;
        }
        return;
      }
      shells.push(...result.additions);
    }else{
      draft.shell_id=draft.shell_id.trim();
      shells.push(draft);
    }
    dialog.close();render();
    const card=$('shells').children[first];
    if(result)showTleResult(card,result);
    const heading=card.querySelector('h3');heading.tabIndex=-1;
    heading.focus({preventScroll:true});card.scrollIntoView({block:'start'});
  };
  $('addShell').disabled=false;
}
