#!/usr/bin/env python3
from pathlib import Path
import subprocess, json, csv, time, statistics as st, threading, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT=Path('/root/autodl-tmp/fast30_final')
CODE=ROOT/'code_sensitivity_v6_20261008'
FLAT=ROOT/'code_flat_profilefix_20260929'
FROZEN=ROOT/'results/fast30_semanticperturb_flat_v6_20260930'
DATA=ROOT/'code_revised_20260910/dataset'
OUT=ROOT/'results/alpha001_full30_20261009'
LOG=OUT/'logs'
OUT.mkdir(parents=True,exist_ok=True); LOG.mkdir(parents=True,exist_ok=True)

RERUN=['BankFull','BrainMethod','CustomerChurn','DriedBean','HeartDisease','MentalStress','OnlineShoppers','SmokeAlarm','SoftwareQuality','StudentsAcademic','ThyroidDiff']
REUSE=['DiabetesHealth_sample10','Income','MagicGamma','PrecisionFog']
SEEDS=[f'S{i:02d}' for i in range(1,31)]
CASES=[(d,s) for d in RERUN for s in SEEDS]
lock=threading.Lock()
status={
 'started':time.strftime('%Y-%m-%dT%H:%M:%S'),
 'alpha':0.01,
 'source_alpha005_dir':str(FROZEN),
 'output_dir':str(OUT),
 'rerun_datasets':RERUN,
 'reuse_from_alpha005':REUSE,
 'total_cases':len(CASES),
 'p2_completed':0,'p2_failed':0,'e2e_completed':0,'e2e_failed':0,
 'p2_running':[],'e2e_current':None,'messages':[],'phase':'running'
}

def save():
 with lock:
  (OUT/'status.json').write_text(json.dumps(status,indent=2))

def note(s):
 with lock:
  print(time.strftime('%F %T'),s,flush=True)
  status['messages'].append(s); status['messages']=status['messages'][-200:]
  (OUT/'status.json').write_text(json.dumps(status,indent=2))

def run_to_log(cmd,log):
 Path(log).parent.mkdir(parents=True,exist_ok=True)
 with open(log,'a') as f:
  f.write('\n### '+time.strftime('%F %T')+' ###\n'+' '.join(map(str,cmd))+'\n'); f.flush()
  r=subprocess.run(list(map(str,cmd)),stdout=f,stderr=subprocess.STDOUT)
 return r.returncode

def p2_case(ds,sid):
 od=OUT/ds/sid; od.mkdir(parents=True,exist_ok=True)
 p2=od/'alpha001_p2.json'
 if p2.exists():
  with lock: status['p2_completed']+=1
  return ds,sid,True,'cached'
 with lock:
  status['p2_running'].append(f'{ds}/{sid}'); (OUT/'status.json').write_text(json.dumps(status,indent=2))
 try:
  rc=run_to_log(['python',CODE/'p2_sensitivity_case.py','--code-dir',FLAT,'--case-dir',FROZEN/ds/sid,
                 '--alpha','0.01','--max-level','adaptive','--workers','4','--out',p2], LOG/f'p2/{ds}_{sid}.log')
  ok=(rc==0 and p2.exists())
 except Exception:
  ok=False
  with open(LOG/f'p2/{ds}_{sid}.log','a') as f: f.write('\n'+traceback.format_exc())
 with lock:
  key=f'{ds}/{sid}'
  if key in status['p2_running']: status['p2_running'].remove(key)
  if ok: status['p2_completed']+=1
  else: status['p2_failed']+=1
  (OUT/'status.json').write_text(json.dumps(status,indent=2))
 return ds,sid,ok,'new'

def e2e_case(ds,sid):
 od=OUT/ds/sid; p2=od/'alpha001_p2.json'; rec=od/'end2end.json'
 if rec.exists():
  with lock: status['e2e_completed']+=1
  note(f'CACHED E2E {ds}/{sid} ({status["e2e_completed"]}/{len(CASES)})')
  return True
 with lock:
  status['e2e_current']=f'{ds}/{sid}'; (OUT/'status.json').write_text(json.dumps(status,indent=2))
 try:
  rc=run_to_log(['python',CODE/'end2end_alpha_case.py','--code-dir',FLAT,'--case-dir',FROZEN/ds/sid,
                 '--dataset-file',DATA/f'{ds}.csv','--p2-json',p2,'--out',rec], LOG/f'e2e/{ds}_{sid}.log')
  ok=(rc==0 and rec.exists())
 except Exception:
  ok=False
  with open(LOG/f'e2e/{ds}_{sid}.log','a') as f: f.write('\n'+traceback.format_exc())
 with lock:
  if ok: status['e2e_completed']+=1
  else: status['e2e_failed']+=1
  status['e2e_current']=None
  n=status['e2e_completed']; nf=status['e2e_failed']
  (OUT/'status.json').write_text(json.dumps(status,indent=2))
 note(('COMPLETE' if ok else 'FAIL')+f' E2E {ds}/{sid} (ok={n}, fail={nf}, total={len(CASES)})')
 return ok

def build_outputs():
 rows=[]
 for ds,sid in CASES:
  p=OUT/ds/sid/'end2end.json'
  if not p.exists(): continue
  o=json.load(open(p))
  rows.append({
   'dataset':ds,'seed_id':sid,'result_source':'rerun_alpha001',
   'alpha001_p2_dim':o['p2']['dimension'],'alpha001_final_dim':o['final']['dimension'],
   'alpha005_final_dim':o['frozen_alpha005']['dimension'],
   'alpha001_accuracy':o['metrics_alpha001']['accuracy']*100,'alpha005_accuracy':o['frozen_alpha005']['accuracy']*100,
   'dacc_001_minus_005_pp':o['paired_alpha001_minus_alpha005_pp']['accuracy'],
   'alpha001_f1':o['metrics_alpha001']['f1_macro']*100,'alpha005_f1':o['frozen_alpha005']['f1_macro']*100,
   'df1_001_minus_005_pp':o['paired_alpha001_minus_alpha005_pp']['f1_macro'],
   'p2_sec':o['timing']['p2_sec'],'p3_sec':o['timing']['p3_sec'],'evaluator_sec':o['timing']['evaluator_sec']})
 if rows:
  with open(OUT/'paired_per_seed_rerun11.csv','w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 summary={}
 for ds in RERUN:
  q=[r for r in rows if r['dataset']==ds]
  if not q: continue
  def ms(k):
   v=[x[k] for x in q]; return {'n':len(v),'mean':st.mean(v),'sd':st.stdev(v) if len(v)>1 else 0,'values':v}
  summary[ds]={
   'n':len(q),'result_source':'rerun_alpha001',
   'alpha001_final_dim':ms('alpha001_final_dim'),'alpha005_final_dim':ms('alpha005_final_dim'),
   'alpha001_accuracy':ms('alpha001_accuracy'),'alpha005_accuracy':ms('alpha005_accuracy'),
   'dacc_001_minus_005_pp':ms('dacc_001_minus_005_pp'),
   'alpha001_f1':ms('alpha001_f1'),'alpha005_f1':ms('alpha005_f1'),
   'df1_001_minus_005_pp':ms('df1_001_minus_005_pp')}
 (OUT/'summary_rerun11.json').write_text(json.dumps(summary,indent=2))
 final_manifest={
  'default_alpha':0.01,
  'alpha005_source_preserved':str(FROZEN),
  'alpha001_rerun_dir':str(OUT),
  'rerun_alpha001_30seeds':RERUN,
  'reused_from_alpha005':REUSE,
  'reuse_rule':'S01-S03 final selected features, final dimension, accuracy, macro-F1 all identical between alpha=0.01 and alpha=0.05',
  'reuse_audit':str(ROOT/'results/sensitivity_v6_20261009/alpha001_vs005_rerun_reuse_manifest.json')
 }
 (OUT/'final_result_source_manifest.json').write_text(json.dumps(final_manifest,indent=2))

save(); note(f'START alpha=.01 full30 rerun: {len(RERUN)} datasets x 30 seeds = {len(CASES)} cases; P2 concurrency=3, E2E GPU serial')
# P2 runs concurrently; E2E consumes completed P2 futures serially.
with ThreadPoolExecutor(max_workers=3) as ex:
 futures={ex.submit(p2_case,ds,sid):(ds,sid) for ds,sid in CASES}
 for fut in as_completed(futures):
  ds,sid=futures[fut]
  try: _,_,ok,_=fut.result()
  except Exception:
   ok=False; note(f'EXCEPTION P2 {ds}/{sid}')
  if not ok:
   note(f'FAIL P2 {ds}/{sid}; skip E2E')
   continue
  e2e_case(ds,sid)
build_outputs()
with lock:
 status['finished']=time.strftime('%Y-%m-%dT%H:%M:%S'); status['phase']='completed'; status['e2e_current']=None
 (OUT/'status.json').write_text(json.dumps(status,indent=2))
note('ALPHA .01 FULL30 RERUN11 COMPLETE')
