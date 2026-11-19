from ablation_common import *
import sys,ast,pickle,math,resource
import numpy as np
sys.path.insert(0,str(BASE))
import p2_parallel_pc as pc
from stage2_runtime import encode_frame
from ablation_batched_pc import install as install_batched_kernel
install_batched_kernel()
# Use the exact path/frontier admission functions from the published paired ablation.
tree=ast.parse((CODE/'historical_online_shoppers_mechanisms_paired.py').read_text())
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('paths','frontier_from_paths')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'historical_paths','exec'))
def cpuclock():
    children=resource.getrusage(resource.RUSAGE_CHILDREN)
    return time.process_time()+children.ru_utime+children.ru_stime
def shallow(ds,sid):
    dest=case(ds,sid)/'shallow.pkl'
    if dest.exists():return
    os.environ['CAU_BATCH_AUDIT_DIR']=str(case(ds,sid)/'shallow_kernel_audit')
    p=p1(ds,sid);frame=encode_frame(p['selector_input']);raw=list(p['feature_names'])
    names=[f for f in raw if frame[f].nunique(dropna=False)>1]+[p['target_name']]
    ti=len(names)-1;corr=np.corrcoef(frame[names].to_numpy(float),rowvar=False);assert np.isfinite(corr).all()
    d=len(names);adj=np.ones((d,d),dtype=np.int8)-np.eye(d,dtype=np.int8);sep=[[[] for _ in range(d)] for _ in range(d)]
    evidence={};summ={};margins={};wall=time.perf_counter();cpu=cpuclock()
    workers=2 if ds=='SoftwareQuality' else 1
    for level in range(4):
        st=pc.parallel_level(corr,len(frame),adj,sep,evidence,level,.01,workers,mode='global',record_margin=(level==3))
        summ[str(level)]={k:st[k] for k in ('ci_tests','deletions','any_testable')}
        if level==3:margins=st['margins']
        save(case(ds,sid)/'shallow_progress.json',dict(level=level,workers=workers,summary=summ,wall_seconds=time.perf_counter()-wall,cpu_seconds=cpuclock()-cpu))
        if not st['any_testable']:break
    x=dict(names=names,ti=ti,corr=corr,n=len(frame),adj=adj,sep=sep,evidence=evidence,summary=summ,margins=margins,wall_seconds=time.perf_counter()-wall,cpu_seconds=cpuclock()-cpu,workers=workers,p1_sha256=manifest(ds,sid)['p1_sha256'],kernel='verified_batched_shallow_global_levels_2_3_for_high_dimension; original scalar deep')
    temp=dest.with_name(dest.name+'.tmp');temp.write_bytes(pickle.dumps(x,protocol=5));temp.replace(dest)
def deep(ds,sid,variant,audit=False):
    dest=case(ds,sid)/('p2_vector_equivalence' if audit else 'p2')/(variant+'.json')
    if dest.exists():return
    os.environ['CAU_BATCH_AUDIT_DIR']=str(case(ds,sid)/'deep_kernel_audit'/(variant+('_check' if audit else '')))
    if ds=='SoftwareQuality' and variant=='wo_target_local_pc':
        assert load(OUT/'deep_kernel_verification.json')['passed']
        from ablation_batched_pc_deep import install as install_deep_kernel
        install_deep_kernel()
    m=manifest(ds,sid);p=p1(ds,sid)
    if variant=='wo_semantic_perturbation':p=load(case(ds,sid)/'generic_p1.json')
    if variant=='wo_reliability':
        den=max(max(p['contribution_scores'].values()),1e-10)
        p['reliable_scores']={f:float(v)/den for f,v in p['contribution_scores'].items()}
    x=pickle.loads((case(ds,sid)/'shallow.pkl').read_bytes());assert x['p1_sha256']==m['p1_sha256']
    names=x['names'];ti=x['ti'];d=len(names);corr=x['corr'];adj=x['adj'];sep=x['sep'];evidence=x['evidence'];raw=p['feature_names']
    scores=np.array([p['reliable_scores'][f] for f in names[:ti]]+[0.],float);cutoff=float(np.percentile(scores[:ti],75))
    unguided=variant=='wo_contribution_guided_paths';start=time.perf_counter();cpu=cpuclock()
    _,stask=paths(adj,ti,scores,cutoff,unguided);frontier=frontier_from_paths(adj,ti,stask,x['margins']);deepstats={};stop=False;stoplevel=None
    for level in range(4,d-1):
        if variant=='wo_target_local_pc':st=pc.parallel_level(corr,x['n'],adj,sep,evidence,level,.01,1,mode='global');mode='global'
        else:
            st=pc.parallel_level(corr,x['n'],adj,sep,evidence,level,.01,1,edge_candidates=frontier['candidate_edges'],mode='restricted',allowed_nodes=frontier['frontier_nodes'],rescue_k=min(level,8));mode='restricted'
        deepstats[str(level)]={k:st[k] for k in ('ci_tests','deletions','local_tests','rescue_tests','any_testable')};deepstats[str(level)]['mode']=mode
        save(case(ds,sid)/'deep_progress'/((variant+('_check' if audit else ''))+'.json'),dict(level=level,deep=deepstats,wall_seconds=time.perf_counter()-start,cpu_seconds=cpuclock()-cpu))
        if not st['any_testable']:break
        stop,stoplevel=pc.should_adaptive_stop({int(k):v for k,v in deepstats.items()})
        if stop:break
    allpaths,selectedpaths=paths(adj,ti,scores,cutoff,unguided);nodeids={int(k) for path in selectedpaths for k in path if k!=ti};chosen={names[k] for k in nodeids};features=[f for f in raw if f in chosen]
    assert features,'empty P2 is a mechanism failure, never silently replace'
    result=base_result(ds,sid,variant,features)
    result.update(p2_features=features,p2_dimension=len(features),shallow=x['summary'],deep=deepstats,target_path_count=len(allpaths),task_path_count=len(selectedpaths),frontier_nodes=len(frontier['frontier_nodes']),frontier_edges=len(frontier['frontier_edges']),deep_candidate_edges=len(frontier['candidate_edges']),adaptive_stopped=stop,adaptive_stop_level=stoplevel,pc_mode='global' if variant=='wo_target_local_pc' else 'target_local',path_policy='all_short_target_paths' if unguided else 'reliability_threshold_paths',alpha=.01,reused_common_shallow=True,shallow_cpu_seconds=x['cpu_seconds'],shallow_wall_seconds=x['wall_seconds'],deep_cpu_seconds=cpuclock()-cpu,deep_wall_seconds=time.perf_counter()-start)
    result['p2_cpu_seconds']=result['shallow_cpu_seconds']+result['deep_cpu_seconds']
    if ds=='SoftwareQuality' and variant=='wo_target_local_pc':
        import ablation_batched_pc_deep as kernel
        result['deep_kernel']='verified_batched_global';result['deep_kernel_counters']=dict(kernel.COUNTS)
    else:result['deep_kernel']='historical_scalar'
    result['p2_wall_seconds_accounted']=result['shallow_wall_seconds']+result['deep_wall_seconds']
    result['timing_note']='Shallow search computed once per seed and reused across variants; wall times from concurrent queues are not controlled standalone speedups.'
    if variant=='full_control':
        record=main_record(ds,sid);assert features==record['selection']['p2_features'],('default P2 membership mismatch',ds,sid)
        if 'target_path_count' in record['selection']:assert len(allpaths)==record['selection']['target_path_count']
        original=record['pc']
        for level,row in x['summary'].items():
            for key in ('ci_tests','deletions','any_testable'):
                assert row[key]==original['shallow'][level][key],('default shallow count mismatch',ds,sid,level,key)
        for level,row in deepstats.items():
            for key in ('ci_tests','deletions','local_tests','rescue_tests'):
                assert row[key]==original['deep'][level][key],('default deep count mismatch',ds,sid,level,key)
        result['default_p2_parity']=True
    save(dest,result)
if __name__=='__main__':
    mode,ds,sid,*rest=sys.argv[1:]
    if mode=='shallow':shallow(ds,sid)
    elif mode=='deep':deep(ds,sid,rest[0])
    elif mode=='deepcheck':deep(ds,sid,rest[0],audit=True)
    else:raise ValueError(mode)
