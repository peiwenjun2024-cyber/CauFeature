"""Equivalent NumPy-batched shallow CI kernel; original PC-stable ordering is retained.

Only SoftwareQuality-sized global levels 2/3 use this kernel. Deep CI jobs remain
the historical scalar implementation. Extra look-ahead probes are audited separately.
"""
import os,json,time
from pathlib import Path
from itertools import combinations,islice
import numpy as np
from scipy.stats import t as student_t
import p2_parallel_pc as pc
ORIGINAL=pc._global_edge_job
COUNTS=dict(logical_probes=0,physical_probes=0,batched_edges=0)
def values(corr,n,i,j,conds):
    indices=np.asarray([[i,j,*c] for c in conds],dtype=np.intp)
    sub=corr[indices[:,:,None],indices[:,None,:]]
    sub=np.nan_to_num(sub,nan=0.,posinf=1.,neginf=-1.)
    sub=sub+pc.CORR_REG*np.eye(indices.shape[1])
    precision=np.linalg.pinv(sub,rcond=1e-3)
    denom=np.sqrt(np.abs(precision[:,0,0]*precision[:,1,1]))
    rho=np.divide(-precision[:,0,1],denom,out=np.zeros(len(conds)),where=denom>=1e-12)
    rho=np.clip(rho,-.999999,.999999);df=int(n-len(conds[0])-2)
    if df<=0:return rho,np.ones(len(conds))
    stat=rho*np.sqrt(df/np.maximum(1e-12,1.-rho*rho))
    return rho,2.*student_t.sf(np.abs(stat),df=df)
def unique_conditions(pools,level):
    seen=set()
    for pool in pools:
        for c in combinations(pool,level):
            if c not in seen:seen.add(c);yield c
def global_job(pair):
    if pc._W_LEVEL not in (2,3) or pc._W_SNAPSHOT.shape[0]<100:return ORIGINAL(pair)
    i,j=pair;s=pc._W_SNAPSHOT;level=pc._W_LEVEL
    if s[i,j]==0:return i,j,False,0,None,None
    a=[k for k in range(len(s)) if s[i,k] and k!=j];b=[k for k in range(len(s)) if s[j,k] and k!=i]
    pools=[]
    if len(a)>=level:pools.append(a)
    if len(b)>=level and b!=a:pools.append(b)
    if not pools:return i,j,False,0,None,None
    tests=0;maxp=-1.;it=unique_conditions(pools,level);COUNTS['batched_edges']+=1
    while True:
        conds=list(islice(it,256))
        if not conds:break
        rho,pv=values(pc._W_CORR,pc._W_N,i,j,conds);COUNTS['physical_probes']+=len(conds)
        # Use the original scalar kernel at numerical decision boundaries.
        for k in np.flatnonzero(np.abs(pv-pc._W_ALPHA)<1e-12):rho[k],pv[k]=pc._partial_corr_pvalue(pc._W_CORR,i,j,conds[k],pc._W_N)
        found=np.flatnonzero(pv>=pc._W_ALPHA)
        k=int(found[0]) if len(found) else len(conds)-1
        used=k+1;tests+=used;COUNTS['logical_probes']+=used;maxp=max(maxp,float(pv[:used].max()))
        if len(found):return i,j,True,tests,(tuple(conds[k]),float(rho[k]),float(pv[k]),'global'),maxp
    return i,j,True,tests,None,maxp
def flush():
    folder=os.environ.get('CAU_BATCH_AUDIT_DIR')
    if folder:
        p=Path(folder);p.mkdir(parents=True,exist_ok=True);(p/(str(os.getpid())+'.json')).write_text(json.dumps(COUNTS))
def install():
    pc._global_edge_job=global_job
    from multiprocessing.util import Finalize
    Finalize(None,flush,exitpriority=1)
