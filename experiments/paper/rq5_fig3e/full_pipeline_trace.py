import os
import sys
from pathlib import Path
PACKAGE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PACKAGE/'scripts'))
from isolated_worker import required_output
SOURCE=Path(os.environ.get('FAST30_SOURCE_ROOT','/root/autodl-tmp/fast30_final')).resolve()
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr,kendalltau
ROOT=SOURCE/'results/sensitivity_five_20261028/cases'
OUT=required_output('RQ5_OUTPUT_ROOT',forbidden=(SOURCE/'results',Path('/autodl-fs/data/multibranch_ablation_20261113')))
DS=['BankFull','OnlineShoppers','StudentsAcademic','Optdigits','SoftwareQuality']
STRATS=['adaptive','quantile_only','linspace_only']
def readp(p): return json.load(open(p))
def ranks(v): return pd.Series(-np.asarray(v,float)).rank(method='average').to_numpy()
def jac(a,b):
 a,b=set(a),set(b)
 return len(a&b)/len(a|b) if a|b else 1.
def find_p3(sel):
 d=readp(sel)
 # selection json stores P3 result under p3
 return d.get('p3',{})
rows=[]
for ds in DS:
 for i in range(1,31):
  sid=f'S{i:02d}';case=ROOT/ds/sid;manifest=readp(case/'manifest.json')
  p1paths={'adaptive':Path(manifest['p1']),
           'quantile_only':case/'p1/continuous_policy__quantile_only.json',
           'linspace_only':case/'p1/continuous_policy__linspace_only.json'}
  p2paths={'adaptive':case/'p2/default.json',
           'quantile_only':case/'p2/continuous_policy__quantile_only.json',
           'linspace_only':case/'p2/continuous_policy__linspace_only.json'}
  p3paths={'adaptive':case/'selection/default.json',
           'quantile_only':case/'selection/continuous_policy__quantile_only.json',
           'linspace_only':case/'selection/continuous_policy__linspace_only.json'}
  p1={s:readp(p) for s,p in p1paths.items()}
  p2={s:readp(p) for s,p in p2paths.items()}
  p3={s:find_p3(p3paths[s]) for s in STRATS}
  base=p1['adaptive']; names=base['feature_names']
  base_c=np.array([base['contribution_scores'][n] for n in names],float)
  base_r=np.array([base['reliable_scores'][n] for n in names],float)
  base_cr,base_rr=ranks(base_c),ranks(base_r)
  base_hi=set(base['high_contrib_features'])
  base_p2=set(p2['adaptive']['p2_features'])
  base_core=set(p3['adaptive'].get('core_features',p3['adaptive'].get('final_features',[])))
  base_rec=set(p3['adaptive'].get('recovered_features',[]))
  for st in STRATS:
   d=p1[st]; c=np.array([d['contribution_scores'][n] for n in names],float)
   r=np.array([d['reliable_scores'][n] for n in names],float)
   cr,rr=ranks(c),ranks(r)
   hi=set(d['high_contrib_features']); p2set=set(p2[st]['p2_features'])
   z=p3[st]; core=set(z.get('core_features',z.get('final_features',[])))
   rec=set(z.get('recovered_features',[])); final=core|rec
   shallow=z2=p2[st].get('shallow',{}); deep=p2[st].get('deep',{})
   shallow_tests=sum(int(v.get('ci_tests',0)) for v in shallow.values())
   deep_tests=sum(int(v.get('ci_tests',0)) for v in deep.values())
   rows.append(dict(dataset=ds,seed=sid,strategy=st,
    n_features=len(names),
    contribution_values_changed=int(not np.allclose(base_c,c,rtol=1e-10,atol=1e-12)),
    contribution_spearman=float(spearmanr(base_c,c).statistic),
    contribution_kendall=float(kendalltau(base_c,c).statistic),
    contribution_mean_abs_rank_shift=float(np.mean(np.abs(base_cr-cr))),
    contribution_max_abs_rank_shift=float(np.max(np.abs(base_cr-cr))),
    reliability_values_changed=int(not np.allclose(base_r,r,rtol=1e-10,atol=1e-12)),
    reliability_spearman=float(spearmanr(base_r,r).statistic),
    reliability_kendall=float(kendalltau(base_r,r).statistic),
    reliability_mean_abs_rank_shift=float(np.mean(np.abs(base_rr-rr))),
    high_set_jaccard=jac(base_hi,hi),
    high_set_changed=int(base_hi!=hi),
    p2_core_jaccard=jac(base_p2,p2set),
    p2_core_changed=int(base_p2!=p2set),
    p2_dim=len(p2set),
    p2_added_vs_full=len(p2set-base_p2),p2_dropped_vs_full=len(base_p2-p2set),
    p3_core_jaccard=jac(base_core,core),p3_core_changed=int(base_core!=core),
    p3_recovered_count=len(rec),p3_recovered_jaccard=jac(base_rec,rec),
    p3_recovered_changed=int(base_rec!=rec),
    p3_recovered_added=len(rec-base_rec),p3_recovered_dropped=len(base_rec-rec),
    p3_candidate_pairs=int(z.get('candidate_pairs',0)),
    p3_evaluated_pairs=int(z.get('evaluated_pairs',0)),
    p3_checks_saved=int(z.get('candidate_pairs',0))-int(z.get('evaluated_pairs',0)),
    p3_final_jaccard=jac(base_core|base_rec,final),
    p3_final_dim=len(final),
    p2_shallow_ci_tests=shallow_tests,p2_deep_ci_tests=deep_tests,
    p2_total_ci_tests=shallow_tests+deep_tests,
    p2_task_paths=int(p2[st].get('task_paths',0)),
    p2_all_target_paths=int(p2[st].get('all_target_paths',0)),
    p2_frontier_nodes=int(p2[st].get('frontier_nodes',0)),
    p2_candidate_edges=int(p2[st].get('candidate_edges',0)),
    p2_stopped=bool(p2[st].get('stopped',False)),
    p2_stop_at=p2[st].get('stop_at')))
df=pd.DataFrame(rows)
df.to_csv(OUT/'full_pipeline_trace_150runs.csv',index=False)
# Summary vs adaptive for each dataset and strategy.
metrics=['contribution_values_changed','contribution_spearman','contribution_mean_abs_rank_shift',
'reliability_values_changed','reliability_spearman','reliability_mean_abs_rank_shift',
'high_set_jaccard','p2_core_jaccard','p3_core_jaccard','p3_recovered_count','p3_recovered_jaccard',
'p3_candidate_pairs','p3_evaluated_pairs','p2_total_ci_tests','p2_task_paths','p2_candidate_edges']
summary=[]
for ds in DS:
 for st in ['quantile_only','linspace_only']:
  g=df[(df.dataset==ds)&(df.strategy==st)]
  for m in metrics:
   a=g[m].to_numpy(float)
   summary.append(dict(dataset=ds,strategy=st,metric=m,n=len(a),mean=np.nanmean(a),sd=np.nanstd(a,ddof=1)))
s=pd.DataFrame(summary);s.to_csv(OUT/'full_pipeline_trace_summary.csv',index=False)
print('=== FULL PIPELINE COMPARISON VS ADAPTIVE ===')
for ds in DS:
 print('\n###',ds)
 for st in ['quantile_only','linspace_only']:
  g=df[(df.dataset==ds)&(df.strategy==st)]
  print(st,'contrib changed',g.contribution_values_changed.sum(),
   'rho',round(g.contribution_spearman.mean(),4),'rank shift',round(g.contribution_mean_abs_rank_shift.mean(),3),
   'reliability changed',g.reliability_values_changed.sum(),'rel rho',round(g.reliability_spearman.mean(),4),
   'high changed',g.high_set_changed.sum(),'high J',round(g.high_set_jaccard.mean(),4),
   'P2 changed',g.p2_core_changed.sum(),'P2 J',round(g.p2_core_jaccard.mean(),4),
   'P3 core changed',g.p3_core_changed.sum(),'recovered changed',g.p3_recovered_changed.sum(),
   'P3 final changed',int((g.p3_final_jaccard<1).sum()),
   'recovered mean',round(g.p3_recovered_count.mean(),3),
   'candidate/evaluated',int(g.p3_candidate_pairs.sum()),int(g.p3_evaluated_pairs.sum()),
   'CI tests mean',round(g.p2_total_ci_tests.mean(),1))
print('\nSaved rows',len(df),'file',OUT/'full_pipeline_trace_150runs.csv')
