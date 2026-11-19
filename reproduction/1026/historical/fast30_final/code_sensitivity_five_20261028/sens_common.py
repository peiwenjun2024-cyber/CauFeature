"""Five-dataset reviewer sensitivity; immutable original sources."""
import sys,os,json,time,hashlib,pickle,copy
from pathlib import Path
ROOT=Path('/root/autodl-tmp/fast30_final')
CODE=ROOT/'code_sensitivity_five_20261028'
OUT=ROOT/'results/sensitivity_five_20261028'
ABL=ROOT/'code_ablation_five_20261028'
OLDOUT=ROOT/'results/ablation_five_20261028'
sys.path.insert(0,str(ABL))
import ablation_common as ac
ac.CODE=CODE;ac.OUT=OUT
from ablation_common import *
PROTOCOL='reviewer-sensitivity-five-20261028'
# All defaults map to one frozen default; individual group rows remain explicit.
GROUPS={
 'high_rule':('P1',['native','p60','p75','p90'],'native'),
 'bootstrap':('P1',[2,4,8],4),
 'aggregation':('P1',['max','mean','median'],'max'),
 'continuous_policy':('P1',['original','quantile_only','linspace_only'],'original'),
 'points':('P1',[51,101,151],101),
 'skew_cutoff':('P1',[1.5,2.,2.5],2.),
 'outlier_cutoff':('P1',[.01,.02,.05],.02),
 'alpha':('P2',[.005,.01,.05],.01),
 'path_percentile':('P2',[60,75,90],75),
 'path_length':('P2',[2,3,4],3),
 'frontier':('P2',[0.,.25,.5],.25),
 'coefficient':('P3',[0.,.2,.3,.4,.5,.6,.8],.4),
 'residual_gate':('P3',[.75,1.,1.25],1.),
}
def cfgkey(group,value):return 'default' if value==GROUPS[group][2] else group+'__'+str(value).replace('.','p')
CONFIGS={'default':dict(stage='default',group='default',value=None)}
for g,(stage,values,default) in GROUPS.items():
 for v in values:
  k=cfgkey(g,v)
  if k!='default':CONFIGS[k]=dict(stage=stage,group=g,value=v)
P1KEYS=[k for k,v in CONFIGS.items() if v['stage']=='P1']
PIPEKEYS=[k for k,v in CONFIGS.items() if v['stage'] in ('P1','P2')]
P3KEYS=[k for k,v in CONFIGS.items() if v['stage']=='P3']
def artifact(ds,sid,stage,key):return case(ds,sid)/stage/(key+'.json')
def getp1(ds,sid,key):return load(artifact(ds,sid,'p1',key)) if CONFIGS[key]['stage']=='P1' else p1(ds,sid)
def stamp(ds,sid,key,features):
 z=base_result(ds,sid,key,features);z.update(protocol=PROTOCOL,config=CONFIGS[key]);return z
def reuse_eval(ds,sid,key):
 dest=artifact(ds,sid,'evaluation',key);sel=load(artifact(ds,sid,'selection',key))
 for folder in [case(ds,sid)/'evaluation',OLDOUT/'cases'/ds/sid/'evaluation']:
  for f in sorted(folder.glob('*.json')):
   x=load(f)
   if x['features']==sel['features'] and x['split_hash']==sel['split_hash']:
    x.update(kind=key,protocol=PROTOCOL,reuse='exact_ordered_features_same_seed_split',source_evaluation=str(f),source_evaluation_sha256=sha(f));save(dest,x);return True
 return False
