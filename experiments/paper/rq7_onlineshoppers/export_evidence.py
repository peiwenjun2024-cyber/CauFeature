import argparse
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,hashlib,pickle,sys,ast,time
import numpy as np
PACKAGE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PACKAGE/'scripts'))
from isolated_worker import required_output
ROOT=Path(os.environ.get('FAST30_SOURCE_ROOT','/root/autodl-tmp/fast30_final')).resolve();R=ROOT/'results'
parser=argparse.ArgumentParser(description='Rebuild OnlineShoppers S01 explanation evidence.')
parser.add_argument('--ablation-root',type=Path,help='A fresh RQ4 output root with shallow, full_control and original P3 records')
args=parser.parse_args()
fresh_root=args.ablation_root.expanduser().resolve() if args.ablation_root is not None else None
CODE=PACKAGE/'experiments'/'ablation'
BASE=PACKAGE/'06_ablation'/'frozen_runtime'
sys.path.insert(0,str(BASE))
import p2_parallel_pc as pc
s=(CODE/'historical_online_shoppers_mechanisms_paired.py').read_text()
tree=ast.parse(s);nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('paths','frontier_from_paths')]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'historical_paths','exec'))
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
src={
'p1':R/'fast30_semanticperturb_flat_v6_20260930/OnlineShoppers/S01/p1_cache.json',
'main_record':R/'alpha001_full30_20261009/OnlineShoppers/S01/record.json',
'main_p2':R/'alpha001_full30_20261009/OnlineShoppers/S01/alpha001_p2.json',
'main_end2end':R/'alpha001_full30_20261009/OnlineShoppers/S01/end2end.json',
'manifest':R/'ablation_five_20261028/cases/OnlineShoppers/S01/manifest.json',
'shallow':R/'ablation_five_20261028/cases/OnlineShoppers/S01/shallow.pkl',
'full_control':R/'ablation_five_20261028/cases/OnlineShoppers/S01/p2/full_control.json',
'p3_original':R/'ablation_five_20261028/cases/OnlineShoppers/S01/selection/p3_original.json',
'p2_code':BASE/'p2_parallel_pc.py',
'path_frontier_code':CODE/'historical_online_shoppers_mechanisms_paired.py',
'ablation_cpu_code':CODE/'ablation_cpu.py'}
if fresh_root is not None:
    case=fresh_root/'cases/OnlineShoppers/S01'
    src.update(manifest=case/'manifest.json',shallow=case/'shallow.pkl',
               full_control=case/'p2/full_control.json',p3_original=case/'selection/p3_original.json')
    m=load(src['manifest'])
    record=Path(m['record']).resolve()
    src.update(p1=Path(m['p1']).resolve(),main_record=record,main_p2=record.with_name('p2_cache.json'))
    del src['main_end2end']
    OUT=required_output('RQ7_OUTPUT_ROOT',forbidden=(R,fresh_root,record.parent))
else:
    m=load(src['manifest'])
    OUT=required_output('RQ7_OUTPUT_ROOT',forbidden=(R,))
p=load(src['p1']);rec=load(src['main_record']);mainp2=load(src['main_p2']);ctl=load(src['full_control']);p3=load(src['p3_original'])
assert sha(src['p1'])==m['p1_sha256']
assert sha(src['main_record'])==m['record_sha256']
if fresh_root is None:
    assert sha(src['p2_code'])==rec['provenance']['frozen_code_provenance']['p2_parallel_pc.py']
    assert sha(src['main_p2'])==rec['provenance']['alpha001_p2_sha256']
    assert sha(src['main_end2end'])==rec['provenance']['alpha001_end2end_sha256']
else:
    assert (m['dataset'],m['seed_id'])==('OnlineShoppers','S01')
    assert rec['status']=='complete' and rec['split_seed_id']=='S01'
    assert sha(src['p2_code'])==rec['provenance']['p2_parallel_pc.py']
    assert sha(src['p1'])==rec['input_cache_sha256']['p1_cache.json']==mainp2['input_p1_cache_sha256']
    assert sha(src['main_p2'])==rec['input_cache_sha256']['p2_cache.json']
    assert mainp2['pc_alpha']==.01
    assert sha(src['manifest'])==ctl['source_manifest_sha256']==p3['source_manifest_sha256']
    assert sha(Path(m['data']))==m['data_sha256']==p['dataset_sha256']
    assert sha(Path(p['selector_model']))==p['selector_model_sha256']
    split_hash=sha(Path(p['split_file']))
    assert all(row['split_hash']==split_hash for row in (m,p,rec,mainp2,ctl,p3))
x=pickle.loads(src['shallow'].read_bytes());assert x['p1_sha256']==sha(src['p1'])
names=x['names'];ti=x['ti'];adj=x['adj'];sep=x['sep'];evidence=x['evidence'];corr=x['corr'];d=len(names)
scores=np.array([p['reliable_scores'][f] for f in names[:ti]]+[0.],float)
cutoff=float(np.percentile(scores[:ti],75))
shallow_all,shallow_task=paths(adj,ti,scores,cutoff,False)
frontier=frontier_from_paths(adj,ti,shallow_task,x['margins'])
deep={};stop=False;stoplevel=None;t=time.perf_counter()
for level in range(4,d-1):
 st=pc.parallel_level(corr,x['n'],adj,sep,evidence,level,.01,1,edge_candidates=frontier['candidate_edges'],mode='restricted',allowed_nodes=frontier['frontier_nodes'],rescue_k=min(level,8))
 deep[str(level)]={k:st[k] for k in ('ci_tests','deletions','local_tests','rescue_tests','any_testable')}
 if not st['any_testable']:break
 stop,stoplevel=pc.should_adaptive_stop({int(k):v for k,v in deep.items()})
 if stop:break
allpaths,task=paths(adj,ti,scores,cutoff,False)
chosen={names[k] for path in task for k in path if k!=ti}
core=[f for f in p['feature_names'] if f in chosen]
assert core==mainp2['p2_features']==rec['selection']['p2_features']==ctl['p2_features']
assert len(allpaths)==mainp2['target_path_count']==ctl['target_path_count']
assert len(task)==mainp2['task_path_count']==ctl['task_path_count']
for lev,row in x['summary'].items():
 for key in ('ci_tests','deletions','any_testable'):assert row[key]==rec['pc']['shallow'][lev][key],(lev,key)
for lev,row in deep.items():
 for key in ('ci_tests','deletions','local_tests','rescue_tests'):assert row[key]==rec['pc']['deep'][lev][key],(lev,key)
assert p3['default_p3_parity'] and p3['core']==core
assert p3['features']==rec['selection']['final_features']
assert set(p3['features'])==set(core)|set(p3['recovered_features'])
assert p3['candidate_pairs']==44 and p3['evaluated_pairs']==4, 'Paper case requires 44 P3 candidates and 4 evaluated pairs'
assert set(p['high_contrib_features'])=={f for f,v in p['contribution_scores'].items() if v>=p['high_contrib_threshold']}
def named_paths(pp):return [[names[k] for k in q] for q in pp]
edges=[[names[i],names[j]] for i in range(d) for j in range(i+1,d) if adj[i,j]]
retained_edges=sorted({tuple(sorted((names[i],names[j]))) for q in task for i,j in zip(q[:-1],q[1:])})
path_counts={str(length):sum(len(q)-1==length for q in task) for length in (1,2,3)}
fh=p['high_contrib_features'];raw=p['feature_names'];final=p3['features']
sources={k:{'path':str(v),'sha256':sha(v)} for k,v in src.items()}
sources['dataset']={'path':m['data'],'sha256':sha(Path(m['data']))}
sources['selector_model']={'path':p['selector_model'],'sha256':sha(Path(p['selector_model']))}
sources['selector_input']={'path':p['selector_input'],'sha256':sha(Path(p['selector_input']))}
sources['split_file']={'path':p['split_file'],'sha256':sha(Path(p['split_file']))}
out={
'dataset':'OnlineShoppers','seed_id':'S01','alpha':.01,'split_hash':p['split_hash'],
'note':('P2 final graph/path diagnostics replayed from archived identical shallow cache using frozen scalar deep code; P1 and P3 pair witnesses read from archived records. No fitting or raw-record modification.'
        if fresh_root is None else
        'P2 diagnostics replayed from the regenerated shallow cache using frozen scalar deep code; P1 and P3 witnesses are from the new run.'),
'sources':sources,
'p1':{'feature_names':raw,'target_name':p['target_name'],'contribution_scores':p['contribution_scores'],'reliable_scores':p['reliable_scores'],'high_contrib_threshold':p['high_contrib_threshold'],'high_contrib_reason':p['high_contrib_reason'],'high_contrib_features':fh,'baseline_values':p['baseline_values'],'reliability_records':p['reliability_records'],'perturbation_values':p['perturbation_values']},
'p2':{'core_features':core,'path_start_threshold':cutoff,'path_start_features':[f for f in names[:ti] if p['reliable_scores'][f]>=cutoff],'target_name':names[ti],'target_neighbors':[names[i] for i in range(ti) if adj[i,ti]],'core_outside_FH':[f for f in core if f not in fh],'FH_outside_core':[f for f in fh if f not in core],'all_target_path_count':len(allpaths),'retained_path_count':len(task),'retained_path_length_counts':path_counts,'final_edges':edges,'retained_edges':[list(e) for e in retained_edges],'all_target_paths':named_paths(allpaths),'retained_paths':named_paths(task),'shallow_retained_paths':named_paths(shallow_task),'frontier_features':[names[i] for i in sorted(frontier['frontier_nodes'])],'deep':deep,'shallow':x['summary'],'total_ci_tests':sum(z['ci_tests'] for z in x['summary'].values())+sum(z['ci_tests'] for z in deep.values()),'adaptive_stop_level':stoplevel,'replay_seconds':time.perf_counter()-t,'replay_parity':{'main_core_order':True,'main_path_counts':True,'all_shallow_deep_counts':True,'full_control_core':True}},
'p3':p3,
'final':{'features':final,'dimension':len(final),'excluded_features':[f for f in raw if f not in final],'recovered_features':p3['recovered_features'],'reduction_percent':100*(len(raw)-len(final))/len(raw)},
'main_metrics':rec['metrics']}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'evidence.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False))
print('COMPLETE',str(OUT/'evidence.json'),sha(OUT/'evidence.json'))
print(json.dumps({'cutoff':cutoff,'core':core,'outside_FH':out['p2']['core_outside_FH'],'path_counts':path_counts,'target_neighbors':out['p2']['target_neighbors'],'edges':len(edges),'retained_edges':len(retained_edges),'allpaths':len(allpaths),'retained':len(task),'pairs':p3['pair_details'],'mainmetrics':rec['metrics']},ensure_ascii=False))
