from sens_common import *
import numpy as np,ast,resource,itertools,math
sys.path.insert(0,str(BASE))
def cpuclock():
 c=resource.getrusage(resource.RUSAGE_CHILDREN);return time.process_time()+c.ru_utime+c.ru_stime
def setup_p1_context(ds,sid,p=None):
 import ablation_gpu as ag
 return ag.context(ds,sid,p)
def score_reference(c,baseline,names,chunk=4096,bootstrap=False):
 baseline=np.asarray(baseline,np.float32).reshape(1,-1)
 if bootstrap:
  pred=c.model(baseline.reshape(1,-1,1),training=False);pred=np.asarray(pred)[0]
 else:pred=c.baseline_pred_dist
 scores=[[] for n in names];jobs=[]
 def flush():
  batch=np.repeat(baseline,len(jobs),axis=0)
  for row,(j,v) in enumerate(jobs):batch[row,j]=v
  predictions=np.asarray(c.model(batch.reshape(-1,len(names),1),training=False))
  from p1_reliability import _js_divergence as reference_js
  for (j,v),dist in zip(jobs,predictions):scores[j].append(float(reference_js(dist,pred) if bootstrap else c._js_divergence(dist,pred)))
 for j,n in enumerate(names):
  for v in c.perturb_values[n]:
   jobs.append((j,float(v)))
   if len(jobs)>=chunk:flush();jobs=[]
 if jobs:flush()
 # Original maxima start at zero, so all-negative roundoff must retain zero.
 return {kind:[(max(0.,float(fn(x))) if kind=='max' else float(fn(x))) if x else 0. for x in scores] for kind,fn in [('max',np.max),('mean',np.mean),('median',np.median)]}
def reliability(matrix,names):
 matrix=np.asarray(matrix,float);med=np.median(matrix,axis=0);mad=np.median(np.abs(matrix-med),axis=0);disp=1.4826*mad/(np.abs(med)+1e-10);rel=1/(1+disp);den=max(float(np.max(matrix[0])),1e-10)
 records={n:dict(original_contribution=float(matrix[0,j]),median_contribution=float(med[j]),mad_contribution=float(mad[j]),robust_relative_dispersion=float(disp[j]),reliability=float(rel[j]),baseline_scores=matrix[:,j].tolist()) for j,n in enumerate(names)}
 return records,{n:float(matrix[0,j]/den*rel[j]) for j,n in enumerate(names)}
def p1batch(ds,sid):
 tf,sg,c,p=setup_p1_context(ds,sid);names=p['feature_names'];seed=int(p['seed']);start=time.perf_counter()
 from high_contribution import select_high_contribution_features
 from p1_reliability import compute_reliability_batched
 # Load only the historical pure support constructor, avoiding any old campaign writes.
 hist=ROOT/'code_reviewer_sensitivity_20261022/p1_scan.py';tree=ast.parse(hist.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='construct_values')
 import pandas as pd
 ns=dict(np=np,pd=pd);exec(compile(ast.Module(body=[node],type_ignores=[]),str(hist),'exec'),ns)
 raw=getattr(sg,'raw_feature_data',None)
 if raw is None:raw=sg.feature_data
 def output(key,z):
  z.update(protocol=PROTOCOL,kind=key,config=CONFIGS[key],source_p1_sha256=manifest(ds,sid)['p1_sha256']);save(artifact(ds,sid,'p1',key),z)
  print('P1_SAVED',ds,sid,key,flush=True)
 # Recompute all three aggregations on exactly the same frozen supports/references.
 c.perturb_values=copy.deepcopy(p['perturbation_values']);base_scores=score_reference(c,c.baseline_np,names)
 assert np.allclose(base_scores['max'],[p['contribution_scores'][n] for n in names],rtol=1e-6,atol=1e-8),'max contribution source mismatch'
 matrices={k:[v] for k,v in base_scores.items()};rng=np.random.default_rng(seed);count=max(2,int(round(len(sg.feature_data)*.8)))
 for b in range(8):
  idx=rng.integers(0,len(sg.feature_data),size=count);baseline=c.reference_baseline_vector(idx);bad=~np.isfinite(baseline);baseline[bad]=c.baseline_np[0][bad]
  with tf.device('/CPU:0'):values=score_reference(c,baseline,names,bootstrap=True)
  for kind in matrices:matrices[kind].append(values[kind])
  save(case(ds,sid)/'progress/p1batch.json',dict(stage='reference_aggregation',references_done=b+1,wall_seconds=time.perf_counter()-start))
 for n in names:
  assert np.allclose(np.asarray(matrices['max'])[:5,names.index(n)],p['reliability_records'][n]['baseline_scores'],rtol=1e-5,atol=1e-7),'max bootstrap source mismatch'
 # Numeric equality validated before any nondefault artifacts are committed.
 default_records,default_scores=reliability([list(p['contribution_scores'].values())]+matrices['max'][1:5],names) if list(p['contribution_scores'])==names else reliability([[p['contribution_scores'][n] for n in names]]+matrices['max'][1:5],names)
 assert np.allclose([default_scores[n] for n in names],[p['reliable_scores'][n] for n in names],rtol=1e-5,atol=1e-6),'default reliable score mismatch'
 for key in P1KEYS:
  if artifact(ds,sid,'p1',key).exists():continue
  cfg=CONFIGS[key];g,v=cfg['group'],cfg['value'];z=copy.deepcopy(p);supports=copy.deepcopy(p['perturbation_values'])
  if g=='high_rule':
   threshold=float(np.percentile([p['contribution_scores'][n] for n in names],int(v[1:])));z.update(high_contrib_threshold=threshold,high_contrib_features=[n for n in names if p['contribution_scores'][n]>=threshold],threshold_rule=v)
  elif g in ('aggregation','bootstrap'):
   kind=v if g=='aggregation' else 'max';b=int(v) if g=='bootstrap' else 4;matrix=copy.deepcopy(matrices[kind][:b+1])
   if kind=='max':matrix[0]=[p['contribution_scores'][n] for n in names]
   records,scores=reliability(matrix,names);con=matrix[0];high,threshold,reason=select_high_contribution_features(names,con)
   z.update(contribution_scores=dict(zip(names,con)),reliability_records=records,reliable_scores=scores,high_contrib_features=high,high_contrib_threshold=threshold,high_contrib_reason=reason,aggregation=kind,bootstrap_iterations=b)
  else:
   supports,branches=ns['construct_values'](p,c,sg.feature_data,raw,g,v)
   unchanged=all(supports[n]==p['perturbation_values'][n] for n in names)
   z.update(effective_support_branch=branches,support_changed_features=[n for n in names if supports[n]!=p['perturbation_values'][n]])
   if not unchanged:
    c.perturb_values=supports;tf.keras.utils.set_random_seed(seed)
    _,con=c.calculate_contributions([supports[n] for n in names]);c.con_list=con;sg.con_list=con
    with tf.device('/CPU:0'):records,scores=compute_reliability_batched(c,con,n_bootstrap=4,sample_fraction=.8,seed=seed,chunk_size=4096)
    high,threshold,reason=select_high_contribution_features(names,con)
    z.update(contribution_scores=dict(zip(names,map(float,con))),reliability_records=records,reliable_scores=scores,high_contrib_features=high,high_contrib_threshold=threshold,high_contrib_reason=reason,perturbation_values=supports)
  output(key,z)
 save(case(ds,sid)/'p1batch_done.json',dict(protocol=PROTOCOL,configs=P1KEYS,source_p1_sha256=manifest(ds,sid)['p1_sha256'],max_and_default_reliability_parity=True,wall_seconds=time.perf_counter()-start))
def shallow(ds,sid,alpha):
 import p2_parallel_pc as pc
 from stage2_runtime import encode_frame
 from ablation_batched_pc import install
 install();os.environ['CAU_BATCH_AUDIT_DIR']=str(case(ds,sid)/'shallow_kernel_audit'/str(alpha));dest=case(ds,sid)/'shallow'/('alpha_'+str(alpha)+'.pkl')
 p=p1(ds,sid);frame=encode_frame(p['selector_input']);names=[n for n in p['feature_names'] if frame[n].nunique(dropna=False)>1]+[p['target_name']];d=len(names);adj=np.ones((d,d),np.int8)-np.eye(d,dtype=np.int8);sep=[[[] for _ in range(d)] for _ in range(d)];evidence={};summ={};margins={};corr=np.corrcoef(frame[names].to_numpy(float),rowvar=False);assert np.isfinite(corr).all();start=time.perf_counter();cpu=cpuclock();workers=2 if ds=='SoftwareQuality' else 1
 for level in range(4):
  st=pc.parallel_level(corr,len(frame),adj,sep,evidence,level,alpha,workers,mode='global',record_margin=level==3);summ[str(level)]={k:st[k] for k in ['ci_tests','deletions','any_testable']}
  if level==3:margins=st['margins']
  save(case(ds,sid)/'progress'/('shallow_'+str(alpha)+'.json'),dict(level=level,summary=summ,wall_seconds=time.perf_counter()-start))
  if not st['any_testable']:break
 x=dict(names=names,ti=d-1,corr=corr,n=len(frame),adj=adj,sep=sep,evidence=evidence,summary=summ,margins=margins,wall_seconds=time.perf_counter()-start,cpu_seconds=cpuclock()-cpu,workers=workers,p1_sha256=manifest(ds,sid)['p1_sha256'],alpha=alpha)
 dest.parent.mkdir(parents=True,exist_ok=True);tmp=dest.with_suffix('.tmp');tmp.write_bytes(pickle.dumps(x,protocol=5));tmp.replace(dest)
def p2(ds,sid,key,force=False):
 import p2_parallel_pc as pc
 from sens_restricted_kernel import install
 install();cfg=CONFIGS[key];g,v=cfg['group'],cfg['value'];p=getp1(ds,sid,key);orig=p1(ds,sid);alpha=float(v) if g=='alpha' else .01;length=int(v) if g=='path_length' else 3;pct=float(v) if g=='path_percentile' else 75.;fraction=float(v) if g=='frontier' else .25
 dest=artifact(ds,sid,'p2','default_probe' if force else key);same_scores=p['reliable_scores']==orig['reliable_scores']
 if not force and same_scores and (alpha,length,pct,fraction)==(.01,3,75.,.25):
  z=load(artifact(ds,sid,'p2','default'));z.update(kind=key,config=cfg,p2_reused_exact_default_scores=True,source_p2_sha256=sha(artifact(ds,sid,'p2','default')));save(dest,z);return
 pc.MAX_PATH_LENGTH=length;pc.U_FRAC=fraction
 x=pickle.loads((case(ds,sid)/'shallow'/('alpha_'+str(alpha)+'.pkl')).read_bytes());assert x['p1_sha256']==manifest(ds,sid)['p1_sha256'];names=x['names'];ti=x['ti'];scores=np.asarray([p['reliable_scores'][n] for n in names[:ti]]+[0.]);cut=float(np.percentile(scores[:ti],pct));adj=x['adj'];sep=x['sep'];evidence=x['evidence'];start=time.perf_counter();cpu=cpuclock()
 os.environ['SENS_KERNEL_PROGRESS']=str(case(ds,sid)/'progress'/('kernel_'+key+('_probe' if force else '')+'.json'))
 frontier=pc.build_frontier(adj,ti,scores,cut,x['margins']);deep={};stopped=False;stop_at=None
 for level in range(4,len(names)-1):
  st=pc.parallel_level(x['corr'],x['n'],adj,sep,evidence,level,alpha,1,edge_candidates=frontier['candidate_edges'],mode='restricted',allowed_nodes=frontier['frontier_nodes'],rescue_k=min(level,8));deep[str(level)]={k:st[k] for k in ['ci_tests','deletions','local_tests','rescue_tests','any_testable']}
  save(case(ds,sid)/'progress'/('p2_'+key+'.json'),dict(level=level,deep=deep,frontier_nodes=len(frontier['frontier_nodes']),candidate_edges=len(frontier['candidate_edges']),wall_seconds=time.perf_counter()-start))
  if not st['any_testable']:break
  stopped,stop_at=pc.should_adaptive_stop({int(k):v for k,v in deep.items()})
  if stopped:break
 allpaths,task=pc.task_paths_from_adj(adj,ti,scores,cut);nodes={i for path in task for i in path if i!=ti};features=[n for n in p['feature_names'] if n in {names[i] for i in nodes}];assert features,'empty P2 core'
 z=stamp(ds,sid,key,features);z.update(p2_features=features,alpha=alpha,path_length=length,path_percentile=pct,frontier_fraction=fraction,frontier_nodes=len(frontier['frontier_nodes']),candidate_edges=len(frontier['candidate_edges']),task_paths=len(task),all_target_paths=len(allpaths),shallow=x['summary'],deep=deep,adaptive_stopped=stopped,adaptive_stop_level=stop_at,shallow_wall_seconds=x['wall_seconds'],deep_wall_seconds=time.perf_counter()-start,shallow_cpu_seconds=x['cpu_seconds'],deep_cpu_seconds=cpuclock()-cpu,source_p1_sha256=sha(artifact(ds,sid,'p1',key)) if cfg['stage']=='P1' else manifest(ds,sid)['p1_sha256'],kernel='verified_batched_restricted_ordered_first_exit')
 if force:
  old=load(artifact(ds,sid,'p2','default'));assert features==old['p2_features'],('default P2 features',ds,sid)
  for level,stats in deep.items():
   for field in ['ci_tests','deletions','local_tests','rescue_tests']:assert stats[field]==old['deep'][level][field],('default P2 counts',ds,sid,level,field)
  assert set(deep)==set(old['deep']),('default stop level',ds,sid);z['default_p2_parity']=True
 save(dest,z)
def recover(ds,sid,key):
 p=getp1(ds,sid,key);q=load(artifact(ds,sid,'p2',key));core=q['p2_features'];eligible=(set(p['feature_names'])-set(core))&set(p['high_contrib_features']);start=time.perf_counter()
 if eligible:
  tf,sg,c,p=setup_p1_context(ds,sid,p);from p3_recovery import ResidualInteractionRecovery
  sg.filtered_features=core;z=ResidualInteractionRecovery(candidate_top_k=200).run();features=list(sg.filtered_features)
 else:features=list(core);z=dict(core_features=core,eligible_residual_features=[],recovered_features=[],candidate_pairs=0,evaluated_pairs=0)
 result=stamp(ds,sid,key,features);result.update(p3=z,p2_source=str(artifact(ds,sid,'p2',key)),p1_source=str(artifact(ds,sid,'p1',key)) if CONFIGS[key]['stage']=='P1' else manifest(ds,sid)['p1'],pair_loop_seconds=time.perf_counter()-start,sampling_protocol='original global RNG / threshold-aware first exit');save(artifact(ds,sid,'selection',key),result)
def p3sweep(ds,sid):
 # New common fixed pair-response experiment; do not write historical caches.
 p=p1(ds,sid);core=load(artifact(ds,sid,'p2','default'))['p2_features'];names=p['feature_names'];con=p['contribution_scores'];maxcon=max(max(con.values()),1e-10);cfgs=['default']+P3KEYS;pairsets={};union=set()
 for key in cfgs:
  cfg=CONFIGS[key];gate=float(cfg['value']) if cfg['group']=='residual_gate' else 1.;high=set(p['high_contrib_features']) if gate==1 else {n for n in names if con[n]>=gate*p['high_contrib_threshold']};res=sorted((set(names)-set(core))&high);pairs=sorted([(a,b) for a in sorted(core) for b in res if a!=b],key=lambda x:(-con[x[0]]*con[x[1]],x[0],x[1]))[:200];pairsets[key]=(res,pairs);union.update(pairs)
 records={}
 if union:
  tf,sg,c,p=setup_p1_context(ds,sid)
  hist=ROOT/'code_reviewer_sensitivity_20261022/p3_scan.py';tree=ast.parse(hist.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='full_js');ns=dict(np=np,itertools=itertools);exec(compile(ast.Module(body=[node],type_ignores=[]),str(hist),'exec'),ns)
  for index,(a,b) in enumerate(sorted(union)):
   salt=int.from_bytes(hashlib.sha256((a+'\0'+b).encode()).digest()[:8],'big');pairseed=(int(p['seed'])+salt)%(2**63-1);js,count,sampled=ns['full_js'](c,a,b,pairseed);records[a,b]=dict(pair=[a,b],max_joint_js=float(js),forward_rows=int(count),sampled=bool(sampled),pair_seed=pairseed)
   if index%10==0:save(case(ds,sid)/'progress/p3sweep.json',dict(pairs_done=index+1,pairs_total=len(union)))
 for key in cfgs:
  cfg=CONFIGS[key];lam=float(cfg['value']) if cfg['group']=='coefficient' else .4;res,pairs=pairsets[key];recovered=set();details=[];allpassing=0
  for a,b in pairs:
   total=con[a]+con[b];threshold=1e-6 if total<1e-9 else total*(1-lam*total/(2*maxcon));rec=records[a,b];passed=rec['max_joint_js']>threshold;allpassing+=int(passed)
   if b in recovered:continue
   details.append(dict(rec,threshold=threshold,accepted=passed))
   if passed:recovered.add(b)
  features=[n for n in names if n in set(core)|recovered];result=stamp(ds,sid,key,features);result.update(core=core,eligible_residual=res,recovered_features=sorted(recovered),candidate_pairs=len(pairs),evaluated_pairs=len(details),passing_pairs_all_fixed_candidates=allpassing,pair_details=details,coefficient=lam,sampling_protocol='common fixed pair-seeded support; full 10000-point budget; all coefficients share scores; not online timing')
  if key=='default':
   expected=load(artifact(ds,sid,'selection','default'))['features'];assert features==expected,('fixed-support default mismatch',ds,sid);save(case(ds,sid)/'p3_fixed_default_check.json',dict(protocol=PROTOCOL,default_parity=True,features=features,pair_records=list(records.values()),candidate_pairs=len(pairs),active=bool(res)))
  else:save(artifact(ds,sid,'selection',key),result)
 save(case(ds,sid)/'p3sweep_done.json',dict(protocol=PROTOCOL,default_parity=True,configs=P3KEYS))
def evaluate(ds,sid,key):
 if reuse_eval(ds,sid,key):return
 import ablation_gpu as ag
 ag.evaluate(ds,sid,key)
 f=artifact(ds,sid,'evaluation',key);z=load(f);z.update(protocol=PROTOCOL);save(f,z)
if __name__=='__main__':
 mode,ds,sid,*rest=sys.argv[1:];scratch=case(ds,sid)/'scratch'/mode/(rest[0] if rest else 'batch');scratch.mkdir(parents=True,exist_ok=True);os.chdir(scratch)
 if mode=='p1batch':p1batch(ds,sid)
 elif mode=='shallow':shallow(ds,sid,float(rest[0]))
 elif mode=='p2':p2(ds,sid,rest[0])
 elif mode=='verifydefault':p2(ds,sid,'default',force=True)
 elif mode=='recover':recover(ds,sid,rest[0])
 elif mode=='p3sweep':p3sweep(ds,sid)
 elif mode=='evaluate':evaluate(ds,sid,rest[0])
 else:raise ValueError(mode)
 print('DONE',*sys.argv[1:],flush=True);sys.stdout.flush();sys.stderr.flush();os._exit(0)
