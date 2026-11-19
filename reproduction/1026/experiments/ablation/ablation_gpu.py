from ablation_common import *
import sys,itertools,ast
import numpy as np
sys.path.insert(0,str(BASE))
def context(ds,sid,p=None):
    tf=tf_setup();p=p or p1(ds,sid)
    import shared_globals as sg
    from model_train_v6 import load_and_preprocess_data
    from feature_perturbation import CausalFeaturePerturbation
    tf.keras.utils.set_random_seed(int(p['seed']));sg.init();sg.random_seed=int(p['seed'])
    assert sha(p['selector_model'])==p['selector_model_sha256']
    load_and_preprocess_data(p['selector_input'],None)
    assert list(sg.feature_names)==p['feature_names']
    model=tf.keras.models.load_model(p['selector_model'],compile=False);sg.model=model
    c=CausalFeaturePerturbation(model,sg.data,dataset_name=ds+'.csv',semantic_metadata_path=p['semantic_metadata_file'])
    assert sha(c.semantic_metadata_path)==p['semantic_metadata_sha256']
    assert all(abs(float(c.baseline_np[0,j])-p['baseline_values'][f])<1e-6 for j,f in enumerate(p['feature_names']))
    c.perturb_values={f:list(p['perturbation_values'][f]) for f in p['feature_names']};c.con_list=[p['contribution_scores'][f] for f in p['feature_names']]
    sg.con_list=c.con_list;sg.high_contrib_features=list(p['high_contrib_features']);sg.p1_reliable_score=dict(p['reliable_scores']);sg.perturbation_instance=c
    return tf,sg,c,p
def generic(ds,sid):
    dest=case(ds,sid)/'generic_p1.json'
    if dest.exists():return
    tf,sg,c,p=context(ds,sid)
    from p1_reliability import compute_reliability_batched
    from high_contribution import select_high_contribution_features
    grids={}
    for f in p['feature_names']:
        x=np.asarray(sg.feature_data[f],float);x=x[np.isfinite(x)];grids[f]=sorted(set(map(float,np.quantile(x,np.linspace(0,1,21))))) if len(x) else []
    c.perturb_values=grids
    _,con=c.calculate_contributions([grids[f] for f in p['feature_names']])
    records,scores=compute_reliability_batched(c,con,n_bootstrap=4,sample_fraction=.8,seed=int(p['seed']))
    high,threshold,reason=select_high_contribution_features(p['feature_names'],con)
    p.update(contribution_scores=dict(zip(p['feature_names'],map(float,con))),reliable_scores=scores,reliability_records=records,high_contrib_features=high,high_contrib_threshold=threshold,high_contrib_reason=reason,perturbation_values=grids,perturbation_ablation='generic_observed_21_quantile_support')
    save(dest,p)
def recover(ds,sid,kind):
    # Mechanism extension uses exactly the old P3 implementation against recomputed P2.
    dest=selection_path(ds,sid,kind)
    if dest.exists():return
    variant=kind.removeprefix('mechanism_');p=p1(ds,sid)
    if variant=='wo_semantic_perturbation':p=load(case(ds,sid)/'generic_p1.json')
    if variant=='wo_reliability':
        den=max(max(p['contribution_scores'].values()),1e-10);p['reliable_scores']={f:float(v)/den for f,v in p['contribution_scores'].items()}
    q=load(case(ds,sid)/'p2'/(variant+'.json'))
    eligible=(set(p['feature_names'])-set(q['p2_features'])) & set(p['high_contrib_features'])
    if not eligible:
        features=list(q['p2_features']);result=base_result(ds,sid,kind,features)
        result.update(p2=q,p3=dict(core_features=sorted(features),eligible_residual_features=[],candidate_pairs=0,evaluated_pairs=0,recovered_features=[],final_features=features),pair_loop_seconds=0.,no_model_load_required=True)
        if variant=='full_control':assert features==main_record(ds,sid)['selection']['final_features'];result['default_p3_parity']=True
        save(dest,result);return
    tf,sg,c,p=context(ds,sid,p)
    from p3_recovery import ResidualInteractionRecovery
    sg.filtered_features=q['p2_features'];start=time.perf_counter();z=ResidualInteractionRecovery(candidate_top_k=200).run()
    features=list(sg.filtered_features)
    result=base_result(ds,sid,kind,features);result.update(p2=q,p3=z,pair_loop_seconds=time.perf_counter()-start)
    if variant=='full_control':assert features==main_record(ds,sid)['selection']['final_features'];result['default_p3_parity']=True
    save(dest,result)
def policy(ds,sid,variant):
    kind='p3_'+variant;dest=selection_path(ds,sid,kind)
    if dest.exists():return
    p=p1(ds,sid)
    core=main_record(ds,sid)['selection']['p2_features'];cs=set(core);res=set(p['feature_names'])-cs;con=p['contribution_scores'];high=set(p['high_contrib_features'])
    eligible=res & high if variant=='original' else res
    pairs={(f,g) for f in cs for g in eligible if f!=g}
    if variant=='all_pairs':pairs.update(itertools.combinations(sorted(res),2))
    ordered=sorted(pairs,key=lambda z:(-con[z[0]]*con[z[1]],z[0],z[1]))
    if not ordered:
        result=base_result(ds,sid,kind,list(core));result.update(core=core,eligible_residual=sorted(eligible),recovered_features=[],candidate_pairs=0,evaluated_pairs=0,pair_details=[],forward_rows=0,model_calls=0,pair_loop_seconds=0.,coefficient=.4,candidate_pair_cap=None,per_pair_support_cap=10000,rr_continues_after_recovery=variant=='all_pairs',sampling_policy='historical seed reset and sequential RNG; no forward truncation',known_truth_metrics=None,no_model_load_required=True)
        if variant=='original':assert core==main_record(ds,sid)['selection']['final_features'];result['default_p3_parity']=True
        save(dest,result);return
    tf,sg,c,p=context(ds,sid,p)
    maxcon=max(max(con.values()),1e-10);recovered=set();details=[];start=time.perf_counter();rows=0;calls=0
    class CountModel:
        def __init__(self,model):self.model=model
        def __call__(self,x,*args,**kwargs):
            nonlocal rows,calls
            rows+=int(x.shape[0]);calls+=1;return self.model(x,*args,**kwargs)
    c.model=CountModel(c.model)
    for i,(f,g) in enumerate(ordered):
        residuals=[v for v in (f,g) if v not in cs]
        if variant!='all_pairs' and residuals and residuals[0] in recovered:continue
        s=con[f]+con[g];threshold=1e-6 if s<1e-9 else s*(1-.4*s/(2*maxcon));before=rows;clock=time.perf_counter()
        accepted,observed=c.joint_effect_exceeds([f,g],required_threshold=threshold,max_combinations=10000)
        details.append(dict(pair=[f,g],threshold=threshold,observed_max=float(observed),accepted=bool(accepted),forward_rows=rows-before,seconds=time.perf_counter()-clock))
        if accepted:recovered.update(residuals)
        if len(details)%100==0:save(case(ds,sid)/'progress'/(kind+'.json'),dict(evaluated=len(details),candidates=len(ordered),recovered=len(recovered),wall_seconds=time.perf_counter()-start))
    features=[f for f in p['feature_names'] if f in cs or f in recovered];result=base_result(ds,sid,kind,features)
    result.update(core=core,eligible_residual=sorted(eligible),recovered_features=sorted(recovered),candidate_pairs=len(ordered),evaluated_pairs=len(details),pair_details=details,forward_rows=rows,model_calls=calls,pair_loop_seconds=time.perf_counter()-start,coefficient=.4,candidate_pair_cap=None,per_pair_support_cap=10000,rr_continues_after_recovery=variant=='all_pairs',sampling_policy='historical seed reset and sequential RNG; no forward truncation',known_truth_metrics=None)
    if variant=='original':assert features==main_record(ds,sid)['selection']['final_features'],('original P3 parity failed',ds,sid);result['default_p3_parity']=True
    save(dest,result)
def evaluate(ds,sid,kind):
    dest=evaluation_path(ds,sid,kind)
    if dest.exists():return
    features=load(selection_path(ds,sid,kind))['features'];assert features
    archived=ROOT/'results/mechanism_ablation_scaled_eval_20261014'/ds/sid/(kind.removeprefix('mechanism_')+'.json')
    if kind.startswith('mechanism_') and archived.exists():
        previous=load(archived)
        if previous['features']==features and previous['split_hash']==manifest(ds,sid)['split_hash']:
            previous.update(kind=kind,protocol='ablation-five-frozen-20261028',reuse='verified_historical_scaled_mechanism',source_evaluation=str(archived),source_evaluation_sha256=sha(archived));save(dest,previous);return
    tf=tf_setup()
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score,f1_score
    from fixed_split_evaluator import _prepare_fixed_splits,_one_hot
    from model_train_v6 import smote_enn_resample,build_model,train_model
    import pandas as pd
    import shared_globals
    # Execute the exact historical evaluator function. Only dataset/P1 locations differ for Optdigits.
    source=ROOT/'code_sensitivity_v6_20261008/run_scaled_evaluator_full_vs_selected.py'
    tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='scaled_eval')
    m=manifest(ds,sid);ns=dict(globals());ns.update(locals(),OLD=Path(m['p1']).parents[2],DATA=Path(m['data']).parent)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
    result=ns['scaled_eval'](ds,sid,features,'selected')
    assert result['features']==features and result['split_hash']==m['split_hash']
    result.update(kind=kind,protocol='ablation-five-frozen-20261028',evaluation_source=str(source),evaluation_source_sha256=sha(source),reuse='new_standardized_evaluation')
    save(dest,result)
if __name__=='__main__':
    mode,ds,sid,*rest=sys.argv[1:]
    scratch=case(ds,sid)/'scratch';scratch.mkdir(parents=True,exist_ok=True);os.chdir(scratch)
    if mode=='generic':generic(ds,sid)
    elif mode=='recover':recover(ds,sid,rest[0])
    elif mode=='policy':policy(ds,sid,rest[0])
    elif mode=='evaluate':evaluate(ds,sid,rest[0])
    else:raise ValueError(mode)
    print('DONE',mode,ds,sid,*rest,flush=True)
    sys.stdout.flush();sys.stderr.flush();os._exit(0)
