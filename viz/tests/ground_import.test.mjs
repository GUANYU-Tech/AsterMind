import assert from 'node:assert/strict';
import test from 'node:test';
import {parseGroundCsv} from '../assets/ground-import.js';

test('ground CSV accepts UTF-8 BOM, reordered columns, quoted IDs and manual-entry defaults',()=>{
  const csv='\uFEFFlon_deg,ingress_id,lat_deg,terminal_count\r\n-180,"站, ""A""",90,\r\n180,b,-90,0\r\n';
  const [a,b]=parseGroundCsv(csv,[]);
  assert.deepEqual(a,{ingress_id:'站, "A"',lat_deg:90,lon_deg:-180,altitude_km:0,min_elevation_deg:10,terminal_count:1,ground_kind:'gateway'});
  assert.equal(b.terminal_count,0);
});

test('invalid ground rows reject the batch and identify the physical file line',()=>{
  const existing=[{ingress_id:'known'}];
  const header='ingress_id,lat_deg,lon_deg,terminal_count,ground_kind\n';
  for(const invalid of ['known,0,0,1,user','a,0,0,1,user','b,,0,1,user','b,0,181,1,user',
    'b,Infinity,0,1,user','b,0,0,-1,user','b,0,0,1.5,user','b,0,0,1,unknown','b,"bad,0,1,user','b,0,0']){
    assert.throws(()=>parseGroundCsv(header+'a,0,0,1,user\n\n'+invalid,existing),/第 4 行/);
  }
  assert.deepEqual(existing,[{ingress_id:'known'}]);
  for(const columns of ['lat_deg,lon_deg','ingress_id,lat_deg,lon_deg,lat_deg','ingress_id,lat_deg,lon_deg,typo']){
    assert.throws(()=>parseGroundCsv(columns+'\na,0,0',[]),/第 1 行/);
  }
});
