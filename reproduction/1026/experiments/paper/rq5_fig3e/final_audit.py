import os
import sys
from pathlib import Path
PACKAGE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PACKAGE/'scripts'))
from isolated_worker import required_output
SOURCE=Path(os.environ.get('FAST30_SOURCE_ROOT','/root/autodl-tmp/fast30_final')).resolve()
import json,glob,hashlib,os
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict

FROZEN=SOURCE/'results/fast30_semanticperturb_flat_v6_20260930'
CASES=SOURCE/'results/sensitivity_five_20261028/cases'
META=PACKAGE/'CauFeature/feature_semantic_metadata.json'
REPORT=SOURCE/'results/sensitivity_five_20261028/reports'
OUT=required_output('RQ5_OUTPUT_ROOT',forbidden=(SOURCE/'results',Path('/autodl-fs/data/multibranch_ablation_20261113')))
DATASETS=['BankFull','OnlineShoppers','StudentsAcademic','Optdigits','SoftwareQuality']
STRATS={'adaptive':'adaptive','quantile_only':'quantile_only','linspace_only':'linspace_only'}
meta=json.load(open(META))['records']
meta_by={(r['dataset'].removesuffix('.csv'),r['feature']):r for r in meta}

def encoded_model_space(frame, features):
    x=frame[features].copy()
    for c in x.select_dtypes(exclude=['number']).columns:
        x[c]=LabelEncoder().fit_transform(x[c].astype(str))
    if x.isnull().any().any(): x=x.fillna(x.mean())
    return x

def percentile_gaps(raw, candidates):
    x=np.sort(np.asarray(raw,float)); v=np.sort(np.unique(np.asarray(candidates,float)))
    if len(x)==0 or len(v)==0:return np.nan,np.nan,0
    l=np.searchsorted(x,v,'left'); r=np.searchsorted(x,v,'right')
    ranks=np.clip((l+r)/(2*len(x)),0,1)
    gaps=np.diff(np.r_[0.,np.sort(ranks),1.])
    return float(gaps.max()),float(np.quantile(gaps,.95)),len(v)

rows=[]
for ds in DATASETS:
 for si in range(1,31):
    sid=f'S{si:02d}'; case=CASES/ds/sid
    manifest=json.load(open(case/'manifest.json'))
    p1=json.load(open(manifest['p1']))
    raw=pd.read_csv(p1['selector_input'])
    feats=p1['feature_names']
    modelx=encoded_model_space(raw,feats)
    pmap={}
    for st in STRATS:
        path=manifest['p1'] if st=='adaptive' else str(case/f'p1/continuous_policy__{st}.json')
        d=json.load(open(path)); pmap[st]=d
    for f in feats:
        pol=p1['perturbation_policy'][f]
        mr=meta_by.get((ds,f),{})
        sem=mr.get('semantic_type','unknown')
        pclass=mr.get('perturbation_semantic_class','unknown')
        vals0=np.asarray(pmap['adaptive']['perturbation_values'][f],float)
        if pol in ('observed_model_support','observed_model_support_capped'):
            support=np.unique(pd.to_numeric(modelx[f],errors='coerce').dropna().to_numpy(float))
            for st,d in pmap.items():
                vals=np.unique(np.asarray(d['perturbation_values'][f],float))
                in_support=np.isin(vals,support)
                rows.append(dict(dataset=ds,seed=sid,feature=f,feature_type='discrete_observed_support',
                    semantic_type=sem,policy=pol,strategy=st,n_observed_support=len(support),
                    n_perturbation_points=len(vals),all_points_observed=bool(in_support.all()) if len(vals) else False,
                    point_support_precision=float(in_support.mean()) if len(vals) else np.nan,
                    support_coverage=float(len(np.intersect1d(vals,support))/len(support)) if len(support) else np.nan,
                    within_observed_range=bool((vals.min()>=support.min()-1e-9 and vals.max()<=support.max()+1e-9)) if len(vals) and len(support) else False,
                    max_ecdf_gap=np.nan,p95_ecdf_gap=np.nan,decile_coverage=np.nan))
        elif pol=='continuous_profile_quantile_or_linspace':
            xr=pd.to_numeric(raw[f],errors='coerce').dropna().to_numpy(float)
            if len(xr)==0: continue
            for st,d in pmap.items():
                vals=np.unique(np.asarray(d['perturbation_values'][f],float))
                mx,p95,n=percentile_gaps(xr,vals)
                # Bin each perturbation candidate by its empirical percentile into deciles.
                xsort=np.sort(xr)
                ranks=(np.searchsorted(xsort,vals,'left')+np.searchsorted(xsort,vals,'right'))/(2*len(xsort))
                dec=len(np.unique(np.minimum((ranks*10).astype(int),9)))/10 if len(ranks) else 0
                lo,hi=float(xr.min()),float(xr.max())
                rows.append(dict(dataset=ds,seed=sid,feature=f,feature_type='continuous_bounded',
                    semantic_type=sem,policy=pol,strategy=st,n_observed_support=len(np.unique(xr)),
                    n_perturbation_points=n,all_points_observed=bool(np.isin(vals,xr).all()) if len(vals) else False,
                    point_support_precision=float(np.isin(vals,xr).mean()) if len(vals) else np.nan,
                    support_coverage=np.nan,
                    within_observed_range=bool((vals.min()>=lo-1e-9 and vals.max()<=hi+1e-9)) if len(vals) else False,
                    max_ecdf_gap=mx,p95_ecdf_gap=p95,decile_coverage=dec))
        else:
            raise RuntimeError(f'unknown policy {pol}')
audit=pd.DataFrame(rows)
audit.to_csv(OUT/'feature_support_validity_coverage.csv',index=False)

print('=== FEATURE SUPPORT AUDIT ===')
for (ds,typ),g in audit.groupby(['dataset','feature_type'],sort=False):
 print('\n',ds,typ,'feature-seed records',len(g))
 for st,h in g.groupby('strategy',sort=False):
  print(st,'all-valid %',round(100*h.all_points_observed.mean(),2),
        'range-valid %',round(100*h.within_observed_range.mean(),2),
        'mean points',round(h.n_perturbation_points.mean(),2),
        'support precision',round(h.point_support_precision.mean(),4) if h.point_support_precision.notna().any() else 'NA',
        'support coverage',round(h.support_coverage.mean(),4) if h.support_coverage.notna().any() else 'NA',
        'max ECDF gap',round(h.max_ecdf_gap.mean(),4) if h.max_ecdf_gap.notna().any() else 'NA',
        'decile coverage',round(h.decile_coverage.mean(),4) if h.decile_coverage.notna().any() else 'NA')

# Paired outcome inference using all 30 matched seeds in each dataset.
src=pd.read_csv(REPORT/'group_parameter_seed_results.csv')
src=src[src.group.eq('continuous_policy')].copy()
base=src[src.is_default.astype(bool)][['dataset','seed_id','dimension','accuracy','f1_macro']]
base=base.rename(columns={'dimension':'dim_full','accuracy':'acc_full','f1_macro':'f1_full'})
outcomes=[]
rng=np.random.default_rng(20261113)
for st in ['quantile_only','linspace_only']:
 x=src[src.value.eq(st)].merge(base,on=['dataset','seed_id'],validate='one_to_one')
 for ds,g in x.groupby('dataset',sort=False):
  metrics={
   'delta_dim':g.dimension.to_numpy(float)-g.dim_full.to_numpy(float),
   'delta_accuracy_pp':g.delta_acc_pp.to_numpy(float),
   'delta_macro_f1_pp':g.delta_f1_pp.to_numpy(float),
   'final_jaccard':g.final_jaccard.to_numpy(float)}
  for metric,a in metrics.items():
   a=np.asarray(a,float); n=len(a)
   # Paired-seed percentile bootstrap CI for the mean; deterministic 20k replicates.
   idx=rng.integers(0,n,size=(20000,n))
   means=a[idx].mean(axis=1)
   lo,hi=np.quantile(means,[.025,.975])
   outcomes.append(dict(strategy=st,dataset=ds,metric=metric,n=n,mean=float(a.mean()),
       sd=float(a.std(ddof=1)),ci95_low=float(lo),ci95_high=float(hi),
       wins=int((a>1e-12).sum()),ties=int((np.abs(a)<=1e-12).sum()),
       losses=int((a< -1e-12).sum())))
  outcomes.append(dict(strategy=st,dataset=ds,metric='changed_subset_runs',n=len(g),
      mean=int((g.final_jaccard.to_numpy(float)<1-1e-12).sum()),sd=np.nan,ci95_low=np.nan,ci95_high=np.nan,
      wins=np.nan,ties=np.nan,losses=np.nan))
ci=pd.DataFrame(outcomes)
ci.to_csv(OUT/'paired_outcome_bootstrap_ci.csv',index=False)

print('\n=== PAIRED OUTCOME SUMMARY VS ADAPTIVE ===')
for st in ['quantile_only','linspace_only']:
 print('\n',st)
 for ds in DATASETS:
  g=ci[(ci.strategy==st)&(ci.dataset==ds)]
  def get(m): return g[g.metric==m].iloc[0]
  f=get('delta_macro_f1_pp'); a=get('delta_accuracy_pp'); dim=get('delta_dim'); jac=get('final_jaccard'); ch=get('changed_subset_runs')
  print(ds,'changed',int(ch['mean']),'J',f"{jac['mean']:.4f} [{jac['ci95_low']:.4f},{jac['ci95_high']:.4f}]",
    'dDim',f"{dim['mean']:.4f} [{dim['ci95_low']:.4f},{dim['ci95_high']:.4f}]",
    'dAcc pp',f"{a['mean']:.4f} [{a['ci95_low']:.4f},{a['ci95_high']:.4f}]",
    'dF1 pp',f"{f['mean']:.4f} [{f['ci95_low']:.4f},{f['ci95_high']:.4f}]")

# Threshold robustness records: exact invariance across each alternative, with all per-dataset counts.
summary=pd.read_csv(REPORT/'dataset_parameter_summary.csv')
thr=summary[summary.group.isin(['skew_cutoff','outlier_cutoff'])].copy()
thr.to_csv(OUT/'branch_threshold_robustness.csv',index=False)
print('\n=== THRESHOLD ROBUSTNESS ===')
for grp,g in thr.groupby('group',sort=False):
 print(grp,'records',len(g),'changed',int(g.changed_subset_cases.sum()),
       'all Jaccard 1',bool(np.allclose(g.final_jaccard_mean,1)),
       'all dDim 0',bool(np.allclose(g.delta_dimension_mean,0)),
       'all dAcc 0',bool(np.allclose(g.delta_acc_pp_mean,0)),
       'all dF1 0',bool(np.allclose(g.delta_f1_pp_mean,0)))

print('\nOUTPUTS')
for f in ['feature_support_validity_coverage.csv','paired_outcome_bootstrap_ci.csv','branch_threshold_robustness.csv']:
 p=OUT/f;print(p,p.stat().st_size)
