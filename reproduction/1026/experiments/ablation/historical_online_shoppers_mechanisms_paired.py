"""OnlineShoppers S01 isolated mechanistic controls, alpha=.01.
No modifications to frozen Full, stage ablation or v2 batch.
"""
import argparse, hashlib, json, os, sys, time, resource
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/fast30_final')
BASE=ROOT/'code_flat_profilefix_20260929'
sys.path.insert(0,str(BASE))
from stage1_runtime_mech import run_stage1_mechanism
from stage2_runtime import encode_frame
from stage3_runtime_mech import run_stage3_mechanism
from fixed_split_evaluator import evaluate_fixed_split
from high_contribution import select_high_contribution_features
import p2_parallel_pc as pc

VERSIONS=('full_control','wo_reliability','wo_target_local_pc','wo_recovery',
 'wo_contribution_guided_paths','wo_adaptive_deep_stop','wo_semantic_perturbation')
OUT=ROOT/'results/mechanism_ablation_onlineshoppers_paired_20261013'
DS='OnlineShoppers'; SEED='S01'

def paths(adj,ti,scores,cutoff,unguided=False):
    allp=pc.find_target_skeleton_paths(adj,ti,pc.MAX_PATH_LENGTH)
    task=[p for p in allp if unguided or len(p)==2 or scores[p[0]]>=cutoff]
    return allp,task

def frontier_from_paths(adj,ti,task,margins):
    import math
    from itertools import combinations
    nodes={ti}
    for p in task:nodes.update(map(int,p))
    edges={(i,j) for i,j in combinations(sorted(nodes),2) if adj[i,j]}
    path_edges=pc._path_edge_union(task) & edges
    others=sorted(edges-path_edges)
    ranked=sorted(others,key=lambda e:(-float(margins.get(e,-1.0)),e))
    uncertain=set(ranked[:math.ceil(pc.U_FRAC*len(others))])
    return {'frontier_nodes':nodes,'candidate_edges':path_edges|uncertain,
       'frontier_edges':edges,'path_edges':path_edges,'uncertainty_edges':uncertain}

def recompute_p2(p1,variant):
    df=encode_frame(p1['selector_input']);raw=list(p1['feature_names'])
    nonconstant=[x for x in raw if df[x].nunique(dropna=False)>1]
    names=nonconstant+[p1['target_name']];ti=len(nonconstant);d=len(names)
    corr=np.corrcoef(df[names].to_numpy(float),rowvar=False)
    assert np.isfinite(corr).all()
    scores=np.array([p1['reliable_scores'][x] for x in nonconstant]+[0.0],float)
    cutoff=float(np.percentile(scores[:ti],75))
    adj=np.ones((d,d),dtype=np.int8)-np.eye(d,dtype=np.int8)
    sep=[[[] for _ in range(d)] for _ in range(d)]
    evidence={};shallow={};deep={};margins={}
    unguided=variant=='wo_contribution_guided_paths'
    t0=time.perf_counter()
    for level in range(4):
        s=pc.parallel_level(corr,len(df),adj,sep,evidence,level,.01,1,
                            mode='global',record_margin=(level==3))
        shallow[str(level)]={k:s[k] for k in ('ci_tests','deletions','any_testable')}
        if level==3:margins=s['margins']
        if not s['any_testable']:break
    _,shallow_task=paths(adj,ti,scores,cutoff,unguided)
    frontier=frontier_from_paths(adj,ti,shallow_task,margins)
    stop=False;stop_level=None
    for level in range(4,d-1):
        if variant=='wo_target_local_pc':
            s=pc.parallel_level(corr,len(df),adj,sep,evidence,level,.01,1,mode='global')
            mode='global'
        else:
            s=pc.parallel_level(corr,len(df),adj,sep,evidence,level,.01,1,
                  edge_candidates=frontier['candidate_edges'],mode='restricted',
                  allowed_nodes=frontier['frontier_nodes'],rescue_k=min(level,8))
            mode='restricted'
        deep[str(level)]={'ci_tests':s['ci_tests'],'deletions':s['deletions'],
          'local_tests':s['local_tests'],'rescue_tests':s['rescue_tests'],
          'any_testable':s['any_testable'],'mode':mode}
        if not s['any_testable']:break
        if variant!='wo_adaptive_deep_stop':
            stop,stop_level=pc.should_adaptive_stop({int(k):v for k,v in deep.items()})
            if stop:break
    allpaths,selectedpaths=paths(adj,ti,scores,cutoff,unguided)
    nodes={int(k) for p in selectedpaths for k in p if k!=ti}
    selected=[x for x in raw if x in {names[k] for k in nodes}]
    if not selected:raise RuntimeError('empty P2')
    return {'p2_features':selected,'p2_dimension':len(selected),
      'pc_mode':'global' if variant=='wo_target_local_pc' else 'target_local',
      'path_policy':'all_short_target_paths' if unguided else 'reliability_threshold_paths',
      'adaptive_stop_enabled':variant!='wo_adaptive_deep_stop',
      'adaptive_stopped':stop,'adaptive_stop_level':stop_level,
      'shallow':shallow,'deep':deep,'shallow_task_paths':len(shallow_task),
      'task_path_count':len(selectedpaths),'target_path_count':len(allpaths),
      'frontier_nodes':len(frontier['frontier_nodes']),
      'frontier_edges':len(frontier['frontier_edges']),
      'deep_candidate_edges':len(frontier['candidate_edges']),
      'p2_seconds':time.perf_counter()-t0,'split_hash':p1['split_hash']}

def generic_p1():
    """Fixed numerical grid independent of semantic type; true P1 contribution and R recompute."""
    import tensorflow as tf
    import shared_globals
    from model_train_v6 import load_and_preprocess_data
    from feature_perturbation import CausalFeaturePerturbation
    from p1_reliability import compute_reliability_batched
    p1=run_stage1_mechanism(DS,SEED,True)
    seed=int(p1['seed'])
    tf.keras.utils.set_random_seed(seed)
    try:tf.config.experimental.enable_op_determinism()
    except Exception:pass
    shared_globals.init();shared_globals.random_seed=seed
    load_and_preprocess_data(p1['selector_input'],None)
    model=tf.keras.models.load_model(p1['selector_model'],compile=False)
    perturb=CausalFeaturePerturbation(model,shared_globals.data,dataset_name=DS+'.csv')
    names=list(shared_globals.feature_names)
    # Generic, non-semantic 21-quantile rule for every encoded variable.
    # Values are taken from observed development rows, avoiding impossible fractional categories.
    grids={}
    for n in names:
        x=np.asarray(shared_globals.feature_data[n],dtype=float)
        x=x[np.isfinite(x)]
        if len(x):
            quantiles=np.quantile(x,np.linspace(0,1,21))
            grids[n]=sorted(set(map(float,quantiles)))
        else: grids[n]=[]
    perturb.perturb_values=grids
    _, con=perturb.calculate_contributions([grids[n] for n in names])
    records,scores=compute_reliability_batched(perturb,con,n_bootstrap=4,sample_fraction=.8,seed=seed)
    high,threshold,reason=select_high_contribution_features(names,con)
    p1.update(contribution_scores=dict(zip(names,map(float,con))),reliable_scores=scores,
       reliability_records=records,high_contrib_features=high,high_contrib_threshold=threshold,
       high_contrib_reason=reason,perturbation_values=grids,
       mechanism_score_source='generic_21quantile_recomputed_con_and_R',
       perturbation_ablation='generic_observed_21_quantile_support')
    return p1

def main(variant,selection_only):
    import pandas as pd
    full=json.loads((ROOT/'results/alpha001_full30_20261009'/DS/SEED/'record.json').read_text())
    t_all=time.perf_counter();t0=time.perf_counter()
    p1=generic_p1() if variant=='wo_semantic_perturbation' else run_stage1_mechanism(DS,SEED,variant!='wo_reliability')
    p1_sec=time.perf_counter()-t0
    if variant=='wo_reliability':
        con=p1['contribution_scores'];denom=max(max(con.values()),1e-10)
        p1['reliable_scores']={n:float(c)/denom for n,c in con.items()}
        p1['mechanism_score_source']='normalized_raw_contributions_R_equals_one'
    assert p1['split_hash']==full['split_hash']
    p2=recompute_p2(p1,variant)
    if variant=='full_control':
        assert p2['p2_features']==full['selection']['p2_features'], ('Full P2 mismatch',p2['p2_features'])
    t0=time.perf_counter()
    final,p3=run_stage3_mechanism(p2,p1,DS,SEED,variant!='wo_recovery')
    p3_sec=time.perf_counter()-t0
    if variant=='full_control':
        assert final==full['selection']['final_features'],('Full P3 mismatch',final)
    if variant=='wo_recovery':
        assert final==p2['p2_features'] and p3.get('mode')=='disabled'
    if variant=='wo_reliability':
        m=max(p1['contribution_scores'].values())
        assert all(abs(p1['reliable_scores'][k]-v/max(m,1e-10))<1e-10
                   for k,v in p1['contribution_scores'].items())
    if variant=='wo_target_local_pc':
        assert all(row['mode']=='global' for row in p2['deep'].values())
    if variant=='wo_contribution_guided_paths':assert p2['path_policy']=='all_short_target_paths'
    if variant=='wo_adaptive_deep_stop':assert not p2['adaptive_stop_enabled']
    if variant=='wo_semantic_perturbation':assert p1['perturbation_ablation']=='generic_observed_21_quantile_support'
    matches=[json.loads(l) for l in (ROOT/'manifest/split_manifest.jsonl').read_text().splitlines()
        if l.strip() and (lambda x:x.get('dataset')==DS+'.csv' and x.get('split_seed_id')==SEED)(json.loads(l))]
    assert len(matches)==1
    man=matches[0]
    assert man['split_hash']==full['split_hash']
    sf=Path(man['split_file']);assert hashlib.sha256(sf.read_bytes()).hexdigest()==man['split_hash']
    ix=pd.read_csv(sf);tr=ix.loc[ix.partition=='train','row_index'].to_numpy(int)
    va=ix.loc[ix.partition=='validation','row_index'].to_numpy(int)
    te=ix.loc[ix.partition=='test','row_index'].to_numpy(int)
    data=ROOT/'code_revised_20260910/dataset/OnlineShoppers.csv'
    assert hashlib.sha256(data.read_bytes()).hexdigest()==man['sha256']
    t0=time.perf_counter()
    metrics={} if selection_only else evaluate_fixed_split(data,tr,va,te,int(man['split_seed']),final)
    eval_sec=time.perf_counter()-t0
    result={'variant':variant,'dataset':DS,'seed':SEED,'split_hash':man['split_hash'],
      'split_seed':man['split_seed'],'selection_only':selection_only,
      'p1_source':p1.get('mechanism_score_source'),'p1_high':p1['high_contrib_features'],
      'p1_contributions':p1['contribution_scores'],'p1_scores':p1['reliable_scores'],
      'p1_support_sizes':{n:len(p1['perturbation_values'][n]) for n in p1['feature_names']},
      'p2':p2,'p3':p3,'final_features':final,'metrics':metrics,
      'split_sizes':{'train':len(tr),'validation':len(va),'test':len(te)},
      'timing_seconds':{'p1_loading_or_recompute':p1_sec,'p2':p2['p2_seconds'],'p3':p3_sec,
      'evaluator':eval_sec,'total':time.perf_counter()-t_all},
      'peak_process_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0,
      'alpha':0.01}
    out=OUT/SEED/variant/('selection_debug.json' if selection_only else 'mechanism_record.json')
    out.parent.mkdir(parents=True,exist_ok=True)
    if out.exists():raise FileExistsError(out)
    tmp=out.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2,ensure_ascii=False));tmp.replace(out)
    print('VALIDATED_AND_SAVED',out,'dim',len(final), 'p2',p2['p2_dimension'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--variant',choices=VERSIONS,required=True)
    ap.add_argument('--selection-only',action='store_true')
    ap.add_argument('--seed',choices=[f'S{i:02d}' for i in range(1,31)],default='S01')
    x=ap.parse_args()
    SEED=x.seed
    main(x.variant,x.selection_only)
