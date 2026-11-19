#!/usr/bin/env python3
from pathlib import Path
import argparse,json,sys,os,time,hashlib
import numpy as np,pandas as pd

ap=argparse.ArgumentParser()
ap.add_argument('--code-dir',required=True)
ap.add_argument('--case-dir',required=True)
ap.add_argument('--dataset-file',required=True)
ap.add_argument('--p2-json',required=True)
ap.add_argument('--out',required=True)
a=ap.parse_args()
code=Path(a.code_dir); case=Path(a.case_dir); datafile=Path(a.dataset_file); p2j=Path(a.p2_json); out=Path(a.out)
out.parent.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(code))

import tensorflow as tf
import shared_globals
from model_train_v6 import load_and_preprocess_data
from feature_perturbation import CausalFeaturePerturbation
from p3_recovery import ResidualInteractionRecovery
from fixed_split_evaluator import evaluate_fixed_split

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

p1=json.loads((case/'p1_cache.json').read_text())
frozen=json.loads((case/'record.json').read_text())
p2=json.loads(p2j.read_text())
seed=int(p1['seed'])
assert p2['alpha']==0.01
assert p2['dataset']==p1['dataset']
assert sha(datafile)==p1['dataset_sha256']

tf.keras.utils.set_random_seed(seed); np.random.seed(seed)
try: tf.config.experimental.enable_op_determinism()
except Exception: pass
shared_globals.init(); shared_globals.random_seed=seed
load_and_preprocess_data(p1['selector_input'],None)
model=tf.keras.models.load_model(p1['selector_model'],compile=False)
shared_globals.model=model
names=list(shared_globals.feature_names)
assert names==list(p1['feature_names'])
con=[float(p1['contribution_scores'][n]) for n in names]
shared_globals.con_list=con
shared_globals.p1_contribution_reliability={k:float(v['reliability']) for k,v in p1['reliability_records'].items()}
shared_globals.p1_reliable_score={k:float(v) for k,v in p1['reliable_scores'].items()}
shared_globals.high_contrib_features=list(p1['high_contrib_features'])

pert=CausalFeaturePerturbation(model,shared_globals.data,dataset_name=p1['dataset'])
pert.perturb_values={n:list(p1['perturbation_values'][n]) for n in names}
pert.con_list=con
shared_globals.perturbation_instance=pert
shared_globals.x_list=[pert.perturb_values[n] for n in names]
shared_globals.feature_values=pert.feature_values
shared_globals.filtered_features=list(p2['p2_features'])

t0=time.perf_counter()
p3info=ResidualInteractionRecovery(candidate_top_k=200).run()
p3_sec=time.perf_counter()-t0
final=list(shared_globals.filtered_features or [])
assert final
split=pd.read_csv(p1['split_file'])
idx={k:split.loc[split.partition==k,'row_index'].to_numpy(int) for k in ['train','validation','test']}
t1=time.perf_counter()
met=evaluate_fixed_split(datafile,idx['train'],idx['validation'],idx['test'],seed,selected_features=final)
eval_sec=time.perf_counter()-t1

fmet=frozen['metrics']
full=frozen['paired_full_feature_baseline']
payload={
 'status':'complete','dataset':p1['dataset'],'seed_id':p1['seed_id'],'seed':seed,'split_hash':p1['split_hash'],
 'alpha':0.01,'default_alpha':0.05,
 'p2':{'dimension':len(p2['p2_features']),'features':p2['p2_features'],'sec':p2['sec'],
       'target_path_count':p2['target_path_count'],'task_path_count':p2['task_path_count']},
 'p3':{'candidate_pairs':p3info['candidate_pairs'],'evaluated_pairs':p3info['evaluated_pairs'],
       'eligible_residual_features':p3info['eligible_residual_features'],
       'recovered_features':p3info['recovered_features'],'sec':p3_sec},
 'final':{'dimension':len(final),'features':final},
 'metrics_alpha001':{'accuracy':float(met['accuracy']),'f1_macro':float(met['f1_macro'])},
 'frozen_alpha005':{'dimension':int(frozen['selection']['final_dimension']),
                    'features':frozen['selection']['final_features'],
                    'accuracy':float(fmet['accuracy']),'f1_macro':float(fmet['f1_macro'])},
 'paired_alpha001_minus_alpha005_pp':{
      'accuracy':100*(float(met['accuracy'])-float(fmet['accuracy'])),
      'f1_macro':100*(float(met['f1_macro'])-float(fmet['f1_macro']))},
 'full_feature_baseline':{'accuracy':float(full['accuracy']),'f1_macro':float(full['f1_macro'])},
 'alpha001_minus_full_pp':{
      'accuracy':100*(float(met['accuracy'])-float(full['accuracy'])),
      'f1_macro':100*(float(met['f1_macro'])-float(full['f1_macro']))},
 'timing':{'p2_sec':float(p2['sec']),'p3_sec':p3_sec,'evaluator_sec':eval_sec},
 'provenance':{'case_dir':str(case),'p2_json':str(p2j),'dataset_sha256':sha(datafile)}
}
out.write_text(json.dumps(payload,indent=2))
print('DONE',p1['dataset'],p1['seed_id'],'alpha001 dim',len(final),
      'acc',met['accuracy'],'vs alpha005 dAcc pp',payload['paired_alpha001_minus_alpha005_pp']['accuracy'],
      'out',out,flush=True)
sys.stdout.flush();sys.stderr.flush();os._exit(0)
