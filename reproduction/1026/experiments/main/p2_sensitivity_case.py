#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse,json,sys,time
import numpy as np,pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_frame(path):
    df=pd.read_csv(path).copy();x,y=df.iloc[:,:-1].copy(),df.iloc[:,-1].copy();le=LabelEncoder()
    for c in x.select_dtypes(exclude=['number']).columns:x[c]=le.fit_transform(x[c])
    if x.isnull().any().any():x=x.fillna(x.mean())
    y=LabelEncoder().fit_transform(pd.Series(y))
    return pd.concat([x,pd.Series(y,index=df.index,name=df.columns[-1])],axis=1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--code-dir',required=True);ap.add_argument('--case-dir',required=True)
    ap.add_argument('--alpha',type=float,default=.05);ap.add_argument('--max-level',default='adaptive')
    ap.add_argument('--workers',type=int,default=4);ap.add_argument('--out',required=True)
    a=ap.parse_args()
    code=Path(a.code_dir);sys.path.insert(0,str(code));import p2_parallel_pc as v22
    case=Path(a.case_dir);p1=json.loads((case/'p1_cache.json').read_text())
    df=encode_frame(p1['selector_input']);raw=list(p1['feature_names']);target=p1['target_name']
    nunique=df[raw].nunique(dropna=False);constants=[n for n in raw if int(nunique[n])<=1];pc=[n for n in raw if n not in constants]
    names,ti=pc+[target],len(pc);mat=df[names].to_numpy(float);corr=np.corrcoef(mat,rowvar=False)
    n,d=len(df),len(names);reliable=np.zeros(d,float);rmap=p1['reliable_scores']
    for i,name in enumerate(pc):reliable[i]=float(rmap.get(name,0.0))
    cutoff=float(np.percentile(reliable[:ti],75.0)) if ti else 0.0
    max_level=None if a.max_level=='adaptive' else int(a.max_level)
    adj=np.ones((d,d),dtype=np.int8)-np.eye(d,dtype=np.int8);sep=[[[] for _ in range(d)] for _ in range(d)]
    evidence,shallow,margins={},{},{};t0=time.perf_counter()
    shallow_max=3 if max_level is None else min(3,max_level)
    for level in range(shallow_max+1):
        q=time.perf_counter();st=v22.parallel_level(corr,n,adj,sep,evidence,level,a.alpha,a.workers,mode='global',record_margin=(level==3))
        shallow[str(level)]={'ci_tests':int(st['ci_tests']),'deletions':int(st['deletions']),'any_testable':bool(st['any_testable']),'sec':time.perf_counter()-q}
        if level==3:margins=st['margins']
        if not st['any_testable']:break
    frontier=v22.build_frontier(adj,ti,reliable,cutoff,margins)
    deep={};stopped=False;stop_level=None
    if max_level is None or max_level>=4:
        deep_end=d-2 if max_level is None else min(max_level,d-2)
        for level in range(4,deep_end+1):
            q=time.perf_counter();st=v22.parallel_level(corr,n,adj,sep,evidence,level,a.alpha,a.workers,
                 edge_candidates=frontier['candidate_edges'],mode='restricted',allowed_nodes=frontier['frontier_nodes'],rescue_k=min(level,8))
            deep[str(level)]={'ci_tests':int(st['ci_tests']),'deletions':int(st['deletions']),'local_tests':int(st['local_tests']),'rescue_tests':int(st['rescue_tests']),'any_testable':bool(st['any_testable']),'sec':time.perf_counter()-q}
            if not st['any_testable']:break
            if max_level is None:
                stopped,stop_level=v22.should_adaptive_stop({int(k):v for k,v in deep.items()})
                if stopped:break
    all_paths,task_paths=v22.task_paths_from_adj(adj,ti,reliable,cutoff)
    nodes=set()
    for path in task_paths:nodes.update(int(x) for x in path if int(x)!=ti)
    p2=[x for x in raw if x in {names[i] for i in nodes}]
    target_neighbors=[names[i] for i in range(ti) if adj[i,ti]!=0]
    truth={'X001','X002','X005'} if all(x in raw for x in ['X001','X002','X005']) else set()
    direct_tp=len(set(p2)&truth);direct_precision=direct_tp/len(p2) if p2 else 0;direct_recall=direct_tp/len(truth) if truth else None
    sk_tp=len(set(target_neighbors)&truth);sk_fp=len(set(target_neighbors)-truth);sk_prec=sk_tp/len(target_neighbors) if target_neighbors else 0;sk_rec=sk_tp/len(truth) if truth else None
    out={
      'dataset':p1['dataset'],'seed':p1['seed'],'alpha':a.alpha,'max_level':a.max_level,'workers':a.workers,
      'raw_dim':len(raw),'p2_features':p2,'p2_dim':len(p2),'target_neighbors':target_neighbors,
      'direct_tp':direct_tp,'direct_precision':direct_precision,'direct_recall':direct_recall,
      'skeleton_tp':sk_tp,'skeleton_fp':sk_fp,'skeleton_precision':sk_prec,'skeleton_recall':sk_rec,
      'target_path_count':len(all_paths),'task_path_count':len(task_paths),'frontier_nodes':len(frontier['frontier_nodes']),
      'adaptive_stopped':stopped,'adaptive_stop_level':stop_level,'shallow':shallow,'deep':deep,'sec':time.perf_counter()-t0}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);Path(a.out).write_text(json.dumps(out,indent=2))
    print('DONE',p1['dataset'],'alpha',a.alpha,'max',a.max_level,'p2',len(p2),'target',target_neighbors,'sec',round(out['sec'],2),flush=True)

if __name__=='__main__':main()
