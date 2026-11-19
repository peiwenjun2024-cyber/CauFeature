"""Ordered first-exit restricted CI batches. Same decisions and logical counts."""
from sens_common import *
import numpy as np
import p2_parallel_pc as pc
from ablation_batched_pc_deep import values
from itertools import combinations,islice
COUNTS=dict(logical_probes=0,physical_probes=0,edges=0);LAST=0
def restricted_job(pair):
 global LAST
 i,j=pair;s=pc._W_SNAPSHOT;level=pc._W_LEVEL
 if not s[i,j]:return i,j,False,0,0,0,None,None
 gi=[k for k in range(len(s)) if s[i,k] and k!=j];gj=[k for k in range(len(s)) if s[j,k] and k!=i]
 li=[k for k in gi if k in pc._W_ALLOWED];lj=[k for k in gj if k in pc._W_ALLOWED]
 seen=set();tests=local=rescue=0;maxp=-1.;testable=False;COUNTS['edges']+=1
 for phase in ['local','rescue']:
  if phase=='local':a,b=li,lj
  else:
   a=list(dict.fromkeys(li+pc._rank_boundary(i,j,[k for k in gi if k not in pc._W_ALLOWED])[:pc._W_RESCUE_K]));b=list(dict.fromkeys(lj+pc._rank_boundary(i,j,[k for k in gj if k not in pc._W_ALLOWED])[:pc._W_RESCUE_K]))
  pools=[]
  if len(a)>=level:pools.append(a)
  if len(b)>=level and b!=a:pools.append(b)
  if pools:testable=True
  for pool in pools:
   def unique():
    for cond in combinations(pool,level):
     if cond not in seen:seen.add(cond);yield cond
   it=unique()
   while True:
    conds=list(islice(it,256))
    if not conds:break
    rho,pv=values(pc._W_CORR,pc._W_N,i,j,conds);COUNTS['physical_probes']+=len(conds)
    boundary=np.flatnonzero(np.abs(pv-pc._W_ALPHA)<1e-12)
    for k in boundary:rho[k],pv[k]=pc._partial_corr_pvalue(pc._W_CORR,i,j,conds[k],pc._W_N)
    COUNTS['physical_probes']+=len(boundary);found=np.flatnonzero(pv>=pc._W_ALPHA);last=int(found[0]) if len(found) else len(conds)-1;used=last+1;tests+=used;COUNTS['logical_probes']+=used;maxp=max(maxp,float(pv[:used].max()))
    if phase=='local':local+=used
    else:rescue+=used
    if time.monotonic()-LAST>15:
     dest=os.environ.get('SENS_KERNEL_PROGRESS')
     if dest:save(dest,dict(COUNTS,current_level=level,current_edge=[i,j],wall_unix=time.time()))
     LAST=time.monotonic()
    if len(found):return i,j,True,tests,local,rescue,(tuple(conds[last]),float(rho[last]),float(pv[last]),phase),maxp
 return i,j,testable,tests,local,rescue,None,maxp
def target_paths(skeleton,target_idx,max_length=3):
 # The skeleton is undirected. Reverse every target-rooted simple path, then
 # sort, yielding exactly the source-rooted function's paths and ordering.
 assert np.array_equal(skeleton,skeleton.T),'Target-rooted enumeration needs undirected skeleton'
 neighbors=[np.flatnonzero(row).tolist() for row in skeleton];out=set();stack=[(target_idx,[target_idx])]
 while stack:
  node,path=stack.pop()
  if len(path)>1:out.add(tuple(reversed(path)))
  if len(path)-1>=max_length:continue
  for nxt in neighbors[node]:
   if nxt not in path:stack.append((nxt,path+[nxt]))
 return [list(path) for path in sorted(out)]
def install():
 pc._restricted_edge_job=restricted_job
 pc.find_target_skeleton_paths=target_paths
