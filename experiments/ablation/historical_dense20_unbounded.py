"""Frozen-P1/P2 P3 counterfactuals. Safe by default: --execute required.
Reports membership, generated candidate pairs, evaluated pairs and final retention separately.
"""
import argparse,hashlib,json,sys,time,itertools
from pathlib import Path
ROOT=Path('/root/autodl-tmp/fast30_final')
CODE=ROOT/'code_synthetic_v6_20261001'
OLD=ROOT/'results/synthetic_v6_20261001'
TRUTH=json.loads((ROOT/'synthetic_v6_20261001/synthetic_truth.json').read_text())
H=ROOT/'RQ4_EXPERIMENT_HUB_20261014'
OUT=H/'p3_dense20_fixed_20261021/policy_results_unbounded'
VARIANTS=('no_recovery','original','all_residual','all_pairs')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loadcase(name):
    from pathlib import Path
    import json
    candidates=(H/'p3_dense20_fixed_20261021',)
    for base in candidates:
        tpath=base/'datasets'/name.replace('.csv','.truth.json')
        if not tpath.exists():continue
        t=json.loads(tpath.read_text())
        seed=int(t['seed']); folder=base/'runs'/Path(name).stem/f"S{seed:02d}"
        p1=folder/'p1_cache.json';p2=folder/'p2_cache.json'
        if not (p1.exists() and p2.exists()):raise FileNotFoundError(folder)
        raw_pairs=t.get('interaction_pairs',t.get('true_interaction_pairs',[]))
        t['interaction_features']=list(raw_pairs[0]) if raw_pairs else []
        t['true_interaction_pairs']=raw_pairs
        t['generation_seed']=seed
        return t,folder,json.loads(p1.read_text()),json.loads(p2.read_text()),sha(p1),sha(p2)
    raise KeyError(name)
def candidate_space(names,core,high,con,variant):
    if variant not in VARIANTS: raise ValueError(variant)
    core=set(core); high=set(high); residual=set(names)-core
    eligible=(residual & high) if variant=='original' else residual
    pairs={(c,r) for c in core for r in eligible if c!=r} if variant!='no_recovery' else set()
    if variant=='all_pairs':
        pairs.update(tuple(sorted(pair)) for pair in itertools.combinations(sorted(residual),2))
    return sorted(pairs,key=lambda x:(-float(con[x[0]])*float(con[x[1]]),x[0],x[1])),sorted(eligible)

def make_pairs(names,core,high,con,variant,limit):
    if limit<1:raise ValueError('budget must be positive')
    ordered,eligible=candidate_space(names,core,high,con,variant)
    return ordered[:limit],len(ordered),eligible

def pair_diagnostics(truth_pair,scheduled,ordered):
    target=frozenset(truth_pair)
    exists=any(frozenset(pair)==target for pair in ordered)
    selected=any(frozenset(pair)==target for pair in scheduled)
    rank=next((i+1 for i,pair in enumerate(ordered) if frozenset(pair)==target),None)
    return {'pair_in_unbudgeted_space':exists,'pair_in_topk':selected,'pair_rank':rank}
def run_case(name,variant,budget,execute,forward_budget=None):
    truth,folder,p1,p2,h1,h2=loadcase(name)
    names=p1['feature_names'];core=p2['p2_features'];high=p1['high_contrib_features']
    con=p1['contribution_scores'];pairs,total,eligible=make_pairs(names,core,high,con,variant,budget)
    expected={tuple(sorted(p)) for p in truth['true_interaction_pairs']}
    ordered,_=candidate_space(names,core,high,con,variant)
    pair_audit={','.join(p):pair_diagnostics(p,pairs,ordered) for p in expected}
    output={'dataset':name,'variant':variant,'budget_pairs':budget,'forward_budget':forward_budget,'sampling_policy':'remaining_budget_truncation_changes_joint_support_when_cap_binding' if forward_budget is not None else 'original','source_p1_sha256':h1,'source_p2_sha256':h2,
            'original_full_p3':str(folder/'p3_cache.json'),
            'core':core,'high':high,'eligible_residual':eligible,'total_candidate_pairs':total,
            'scheduled_candidate_pairs':len(pairs),'true_pairs':[list(v) for v in sorted(expected)],
            'true_pair_in_search_space':{','.join(p):any(frozenset(x)==frozenset(p) for x in ordered) for p in expected},
            'true_pair_coverage_audit':pair_audit,
            'true_members_eligible':{f:f in eligible for pair in expected for f in pair},
            'actual_evaluated_pairs':0,'actual_joint_value_combinations':None}
    if not execute:return output
    if variant=='no_recovery':
        output.update(recovered=[],final_features=[f for f in names if f in set(core)],actual_joint_value_combinations=0,total_forward_rows=0,total_model_calls=0);return output
    import tensorflow as tf
    import numpy as np
    import shared_globals
    from model_train_v6 import load_and_preprocess_data
    from feature_perturbation import CausalFeaturePerturbation
    tf.keras.utils.set_random_seed(int(p1['seed']))
    np.random.seed(int(p1['seed']))
    try:tf.config.experimental.enable_op_determinism()
    except Exception:pass
    shared_globals.init();shared_globals.random_seed=int(p1['seed'])
    modelpath=Path(p1['selector_model'])
    assert sha(modelpath)==p1['selector_model_sha256'],'selector hash mismatch'
    load_and_preprocess_data(p1['selector_input'],None)
    assert list(shared_globals.feature_names)==list(names),'feature order mismatch'
    model=tf.keras.models.load_model(str(modelpath),compile=False);shared_globals.model=model
    class CountingModel:
        def __init__(self,delegate):self.delegate=delegate;self.forward_rows=0;self.calls=0
        def __call__(self,x,*args,**kwargs):
            self.forward_rows+=int(x.shape[0]);self.calls+=1
            return self.delegate(x,*args,**kwargs)
    counter=CountingModel(model)
    context=CausalFeaturePerturbation(model,shared_globals.data,dataset_name=p1['dataset'])
    assert sha(context.semantic_metadata_path)==p1['semantic_metadata_sha256']
    assert all(abs(float(context.baseline_np[0,i])-float(p1['baseline_values'][f]))<1e-6 for i,f in enumerate(names))
    context.perturb_values={f:list(p1['perturbation_values'][f]) for f in names}
    context.con_list=[float(con[f]) for f in names]
    # Count actual candidate forward rows, excluding baseline/setup predictions.
    context.model=counter
    recovered=set();evalpairs=[];maxcon=max(max(map(float,con.values())),1e-10)
    forward_cap=forward_budget
    stopped_for_budget=False
    for f,g in pairs:
        # Original would skip pairs where the residual was already recovered.
        residuals=[x for x in (f,g) if x not in core]
        if variant!='all_pairs' and residuals and residuals[0] in recovered:continue
        cs=float(con[f])+float(con[g]);t=1-.4*((float(con[f])+float(con[g]))/(2*maxcon))
        threshold=1e-6 if cs<1e-9 else t*cs
        start=time.perf_counter()
        before=counter.forward_rows
        if forward_cap is not None and counter.forward_rows>=forward_cap:
            stopped_for_budget=True;break
        if forward_cap is not None:
            remaining=forward_cap-counter.forward_rows
            # Enforce the cap by making the evaluator's chunk fit remaining rows.
            # max_combinations governs the sampled-support proposal; never exceed remaining.
            significant,observed=context.joint_effect_exceeds([f,g],required_threshold=threshold,max_combinations=min(10000,remaining),chunk_size=min(4096,remaining))
        else:
            significant,observed=context.joint_effect_exceeds([f,g],required_threshold=threshold,max_combinations=10000)
        evaluated_rows=counter.forward_rows-before
        duration=time.perf_counter()-start
        evalpairs.append({'pair':[f,g],'threshold':threshold,'observed_max':float(observed),
                          'accepted':bool(significant),'time_sec':duration,'forward_rows':evaluated_rows})
        if significant:recovered.update(residuals)
    output.update(stopped_for_forward_budget=stopped_for_budget,forward_budget=forward_cap,total_forward_rows=counter.forward_rows,total_model_calls=counter.calls,recovered=sorted(recovered),final_features=[f for f in names if f in set(core)|recovered],
                  actual_evaluated_pairs=len(evalpairs),evaluated_pair_details=evalpairs,
                  actual_joint_value_combinations=counter.forward_rows)
    output['true_pair_actually_tested']={','.join(p):any(tuple(sorted(z['pair']))==p for z in evalpairs) for p in expected}
    output['true_pair_accepted']={','.join(p):any(tuple(sorted(z['pair']))==p and z['accepted'] for z in evalpairs) for p in expected}
    return output

def main():
    import argparse
    parser=argparse.ArgumentParser(description="Frozen P1/P2 paired candidate-policy ablation")
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--variant',choices=VARIANTS,required=True)
    parser.add_argument('--budget',type=int,default=200)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--verify-original',action='store_true')
    args=parser.parse_args()
    t,folder,p1,p2,h1,h2=loadcase(args.dataset)
    if args.budget<1:raise ValueError('positive candidate-pair cap required')
    sys.path.insert(0,str(folder.parents[2]/'runtime'))
    # Never modify frozen runtime or source P1/P2 caches.
    dst=OUT/Path(args.dataset).stem/(args.variant+'_B100000.json')
    if dst.exists():raise FileExistsError(dst)
    if not args.execute:
        con=p1['contribution_scores']
        pairs,all_n,elig=make_pairs(p1['feature_names'],p2['p2_features'],p1['high_contrib_features'],con,args.variant,args.budget)
        true=t['true_interaction_pairs']
        print(json.dumps({'case':args.dataset,'variant':args.variant,'candidate_pool':all_n,
                          'scheduled':len(pairs),'ground_truth_pairs':true,
                          'true_pair_top200':{','.join(v):any(set(v)==set(q) for q in pairs) for v in true},
                          'DRY_RUN':True}))
        return
    result=run_case(args.dataset,args.variant,args.budget,True)
    result.update(source_run_dir=str(folder),scenario=t.get('scenario',t.get('kind')),
                  true_predictive_features=t.get('all_true_predictive_features',
                  sorted(set(t.get('main_effect_anchor',[])+t.get('interaction_features',[])))),
                  published_algorithm=(args.variant=='original'))
    if args.verify_original:
        if args.variant!='original':raise ValueError('verify original only')
        expected=set(json.loads((folder/'p3_cache.json').read_text())['recovered_features'])
        got=set(result['recovered'])
        result['original_parity']={'passed':expected==got,'expected':sorted(expected),'got':sorted(got)}
        if expected!=got:raise RuntimeError('ORIGINAL_PARITY_MISMATCH: refusing publishable comparison')
    dst.parent.mkdir(parents=True,exist_ok=True)
    temp=dst.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2)+'\n')
    temp.replace(dst)
    print('SAVED',dst,'RECOVERED',len(result['recovered']),'TESTED',result['actual_evaluated_pairs'],flush=True)

if __name__=='__main__': main()
