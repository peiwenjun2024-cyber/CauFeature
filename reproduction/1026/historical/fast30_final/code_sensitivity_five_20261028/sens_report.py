from sens_common import *
import csv,statistics,collections,fcntl
def csvsave(path,rows):
 if not rows:return
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 lock=(OUT/'report.lock').open('a')
 try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:return
 errors=[];unique=[];grouped=[];counts=collections.Counter();p1parity=0;p3parity=0;changed=collections.Counter()
 for ds in DS:
  for i in range(1,31):
   sid=f'S{i:02d}';m=manifest(ds,sid);base=load(artifact(ds,sid,'selection','default'));b_eval=load(artifact(ds,sid,'evaluation','default'))
   for field,h in [('p1','p1_sha256'),('record','record_sha256')]:
    if sha(m[field])!=m[h]:errors.append('source changed '+m[field])
   for key in CONFIGS:
    f=artifact(ds,sid,'selection',key);ev=artifact(ds,sid,'evaluation',key)
    if not f.exists() or not ev.exists():continue
    x,e=load(f),load(ev)
    if e['features']!=x['features'] or e['split_hash']!=m['split_hash']:errors.append('evaluation mismatch '+str(ev));continue
    a,b=set(x['features']),set(base['features']);row=dict(dataset=ds,seed_id=sid,key=key,dimension=len(a),final_jaccard=len(a&b)/len(a|b),delta_dimension=len(a)-len(b),accuracy=e['accuracy'],f1_macro=e['f1_macro'],delta_f1_pp=100*(e['f1_macro']-b_eval['f1_macro']),delta_acc_pp=100*(e['accuracy']-b_eval['accuracy']),recovered=len(x.get('recovered_features',x.get('p3',{}).get('recovered_features',[]))),eligible=len(x.get('eligible_residual',x.get('p3',{}).get('eligible_residual_features',[]))),candidate_pairs=x.get('candidate_pairs',x.get('p3',{}).get('candidate_pairs')),evaluated_pairs=x.get('evaluated_pairs',x.get('p3',{}).get('evaluated_pairs')),selection_path=str(f),evaluation_path=str(ev),reuse=e.get('reuse',''))
    unique.append(row);counts[key]+=1
   for g,(stage,values,default) in GROUPS.items():
    for v in values:
     key=cfgkey(g,v);row=next((x for x in unique[-len(CONFIGS):] if x['dataset']==ds and x['seed_id']==sid and x['key']==key),None)
     if row:grouped.append(dict(phase=stage,group=g,value=v,is_default=v==default,**row))
   f=case(ds,sid)/'p1batch_done.json'
   if f.exists():p1parity+=bool(load(f).get('max_and_default_reliability_parity'))
   f=case(ds,sid)/'p3_fixed_default_check.json'
   if f.exists():p3parity+=bool(load(f).get('default_parity'))
 summary=[];by=collections.defaultdict(list)
 for row in grouped:by[row['dataset'],row['group'],str(row['value'])].append(row)
 for (ds,g,v),rows in sorted(by.items()):
  z=dict(dataset=ds,group=g,value=v,n=len(rows),complete=len(rows)==30,changed_subset_cases=sum(x['final_jaccard']<1 for x in rows),active_p3_cases=sum(x['eligible']>0 for x in rows),mean_abs_delta_f1_pp=statistics.mean(abs(x['delta_f1_pp']) for x in rows))
  for k in ['dimension','final_jaccard','delta_dimension','f1_macro','delta_f1_pp','delta_acc_pp','recovered']:
   z[k+'_mean']=statistics.mean(x[k] for x in rows);z[k+'_sd']=statistics.stdev(x[k] for x in rows) if len(rows)>1 else None
  summary.append(z)
 csvsave(OUT/'reports/unique_config_seed_results.csv',unique);csvsave(OUT/'reports/group_parameter_seed_results.csv',grouped);csvsave(OUT/'reports/dataset_parameter_summary.csv',summary)
 expected=150*len(CONFIGS);group_expected=150*sum(len(v[1]) for v in GROUPS.values());complete=len(unique)==expected and len(grouped)==group_expected and p1parity==150 and p3parity==150 and not errors
 v=dict(protocol=PROTOCOL,complete=complete,passed=complete and not errors,errors=errors,unique_evaluations=len(unique),expected_unique_evaluations=expected,parameter_seed_records=len(grouped),expected_parameter_seed_records=group_expected,p1_default_parity_cases=p1parity,p3_fixed_support_default_parity_cases=p3parity,completed_by_config=dict(counts),legacy_weights='Not in current frozen executable P2; no fictitious weight scan',bootstrap='P1 reference resampling count, excluding the original reference; not old DDG bootstrap',fixed_support_p3='Pair-specific deterministic support; distinct from online global-RNG early exit; default parity required',cancelled_ablations='Historical SoftwareQuality global control and unguided S14/S28 remain stopped; not part of this campaign')
 failure_path=OUT/'user_failed_path_length4.json'
 if failure_path.exists():
  policy=load(failure_path)
  sched=load(OUT/'status.json')
  v.update(failed_experiments=[dict(dataset='SoftwareQuality',seed_id=sid,config='path_length__4',status='failed',reason='user_requested_stop',evaluation_status='blocked') for sid in policy['seeds']],available_expected_unique_evaluations=expected-2,available_expected_parameter_seed_records=group_expected-2,available_results_complete=len(unique)==expected-2 and len(grouped)==group_expected-2 and p1parity==150 and p3parity==150 and not errors,all_runnable_jobs_finished=not sched.get('running') and sched.get('pending')==0)
 save(OUT/'verification.json',v);lines=['# Five-dataset sensitivity '+('COMPLETE' if complete else 'PARTIAL'),'',json.dumps(v,indent=2),'','All grids fixed before execution. No test-score-based parameter selection.','', '| Dataset | Parameter | Value | n | Mean Jaccard | Mean delta F1 (pp) | Mean abs delta F1 (pp) |','|---|---|---|---:|---:|---:|---:|']
 for x in summary:lines.append(f"|{x['dataset']}|{x['group']}|{x['value']}|{x['n']}|{x['final_jaccard_mean']:.4f}|{x['delta_f1_pp_mean']:.4f}|{x['mean_abs_delta_f1_pp']:.4f}|")
 (OUT/'reports/report.md').write_text('\n'.join(lines));print(json.dumps(v),flush=True)
if __name__=='__main__':main()
