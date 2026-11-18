"""V4 model-SP hash-mask workload: physical metadata/output, valid expert math."""
from .moe_ep_math import marlin_ep_work
PROGRAM='v4_modelsp_masked_marlin_ep_v1'
def logical_work(shape,attrs):
 p=attrs['physical_tokens'];t=shape['tokens']
 if attrs.get('mathematical_program')!=PROGRAM or type(p) is not int or not t<=p:
  raise ValueError('V4 model-SP physical tokens must cover the positive valid token extent')
 w=marlin_ep_work(shape,attrs,model='v4');extra=8*(p-t)*6
 stages=[]
 for stage in w['stages']:
  s={**stage,'demand':dict(stage['demand'])}
  if s['stage'] in ('route_metadata','masked_clamped_silu','masked_sum'):s['read_bytes']+=extra
  if s['stage']=='masked_sum':s['write_bytes']=2*p*shape['hidden']
  s['logical_bytes']=s['read_bytes']+s['write_bytes'];stages.append(s)
 return {**w,'physical_tokens':p,'valid_tokens':t,'output_bytes':2*p*shape['hidden'],'stages':tuple(stages),
  'limitations':w['limitations']+('hash_minus_one_rows_have_zero_expert_work_but_full_output_and_metadata_extent',)}
