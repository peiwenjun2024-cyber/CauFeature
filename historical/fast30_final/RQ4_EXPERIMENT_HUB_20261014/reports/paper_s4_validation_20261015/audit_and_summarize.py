import os,sys
from pathlib import Path
import json,csv,statistics as st,hashlib,importlib.util
source=Path(os.environ.get('FAST30_SOURCE_ROOT','/root/autodl-tmp/fast30_final')).resolve()
H=source/'RQ4_EXPERIMENT_HUB_20261014';B=H/'deep3_extension_20261015'
entry=Path(__file__).resolve()
O=entry.parent;generator=H/'configs/e1_deep3_extension_20261015.py'
if os.environ.get('RQ4_DEEP3_AUDIT_OUT'):
 sys.path.insert(0,str(entry.parents[5]/'scripts'))
 from isolated_worker import required_output
 O=required_output('RQ4_DEEP3_AUDIT_OUT',forbidden=(H,source/'results'))
 generator=entry.parents[2]/'configs/e1_deep3_extension_20261015.py'
 O.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('generator',generator);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
rows=[]
def add(seed,kind,label,T,P):
 tp=len(T&P);fp=len(P-T);fn=len(T-P)
 rows.append(dict(seed=seed,kind=kind,stage=label,tp=tp,fp=fp,fn=fn,selected=len(P),precision=tp/len(P) if P else 0,recall=tp/len(T),full_coverage=int(T<=P)))
for seed in range(1,31):
 stem=f's4_two_chains_deep3_n5000_d100_s{seed:02d}';p=B/'runs'/stem/f'S{seed:02d}'
 t=json.loads((B/'datasets'/(stem+'.truth.json')).read_text());a=json.loads((p/'ddg_skeleton_audit.json').read_text());r=json.loads((p/'record.json').read_text());p1=json.loads((p/'p1_cache.json').read_text())
 assert hashlib.sha256((B/'datasets'/(stem+'.csv')).read_bytes()).hexdigest()==t['dataset_sha256']
 assert hashlib.sha256((B/'datasets'/(stem+'.split.csv')).read_bytes()).hexdigest()==t['split_sha256']==r['split_hash']
 assert a['ci_alpha']==.05 and t['n']==5000 and t['d']==100 and not t['true_interaction_pairs']
 truth=set(t['direct_target_parents']+t['indirect_ancestors']); assert len(truth)==6
 paths=g.paths_to_target(t['true_skeleton_edges']);raw={tuple(q) for q in a['all_target_paths']};screened={tuple(q) for q in a['screened_target_paths']}
 assert raw==g.paths_to_target(a['final_skeleton_edges'])
 assert screened<=raw
 # Reconstruct the implemented contribution-guided acceptance rule.
 cutoff=a['cutoff'];scores=p1['reliable_scores']
 assert screened=={q for q in raw if len(q)==2 or scores[q[0]]>=cutoff}
 node_sets={'P1-only':set(r['selection']['p1_high_contrib_features']),'Unscreened paths':{v for q in raw for v in q if v!='target'},'Screened paths (P2)':set(r['selection']['p2_features']),'Final subset':set(r['selection']['final_features'])}
 assert node_sets['Screened paths (P2)']=={v for q in screened for v in q if v!='target'}
 for label,P in node_sets.items():
  add(seed,'nodes',label,truth,P)
  add(seed,'parents',label,set(t['direct_target_parents']),P&set(t['direct_target_parents']))
  add(seed,'ancestors',label,set(t['indirect_ancestors']),P&set(t['indirect_ancestors']))
 add(seed,'edges','Skeleton',{tuple(sorted(e)) for e in t['true_skeleton_edges']},{tuple(sorted(e)) for e in a['final_skeleton_edges']})
 for depth in (1,2,3):
  T={q for q in paths if len(q)-1==depth}
  for label,P in [('Unscreened',raw),('Screened',screened)]:add(seed,f'paths_{depth}',label,T,{q for q in P if len(q)-1==depth})
with (O/'verified_per_seed.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for kind,label in dict.fromkeys((r['kind'],r['stage']) for r in rows):
 group=[r for r in rows if (r['kind'],r['stage'])==(kind,label)]
 tp,fp,fn=(sum(r[k] for r in group) for k in ['tp','fp','fn'])
 summary.append(dict(kind=kind,stage=label,n=len(group),tp=tp,fp=fp,fn=fn,precision_micro=tp/(tp+fp) if tp+fp else 0,recall_micro=tp/(tp+fn),f1_micro=2*tp/(2*tp+fp+fn),selected_mean=st.mean(r['selected'] for r in group),selected_sd=st.stdev(r['selected'] for r in group),noise_mean=fp/len(group) if kind=='nodes' else None,full_coverage=sum(r['full_coverage'] for r in group)))
(O/'verified_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
