#!/usr/bin/env python3
from pathlib import Path
import json,sys,time,traceback,os
import numpy as np,pandas as pd
ROOT=Path('/root/autodl-tmp/fast30_final')
CODE=ROOT/'code_flat_profilefix_20260929'
MAIN=ROOT/'results/alpha001_full30_20261009'
OLD=ROOT/'results/fast30_semanticperturb_flat_v6_20260930'
DATA=ROOT/'code_revised_20260910/dataset'
OUT=ROOT/'results/scaled_evaluator_full_vs_selected_20261010'
OUT.mkdir(parents=True,exist_ok=True); (OUT/'cases').mkdir(exist_ok=True)
sys.path.insert(0,str(CODE))
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score
from fixed_split_evaluator import _prepare_fixed_splits,_one_hot
from model_train_v6 import smote_enn_resample,build_model,train_model
import shared_globals

manifest=pd.read_csv(MAIN/'official_main_results_15x30_manifest.csv')

def scaled_eval(ds,sid,features,mode):
    seed=int(sid[1:]); p1=json.load(open(OLD/ds/sid/'p1_cache.json'))
    split=pd.read_csv(p1['split_file'])
    idx={k:split.loc[split.partition==k,'row_index'].to_numpy(int) for k in ['train','validation','test']}
    data_path=DATA/f'{ds}.csv'
    xtr,xv,xt,ytr,yv,yt,meta=_prepare_fixed_splits(data_path,idx['train'],idx['validation'],idx['test'],features)
    actual_features=list(xtr.columns)
    sc=StandardScaler(); xtr_s=sc.fit_transform(xtr); xv_s=sc.transform(xv); xt_s=sc.transform(xt)
    try:
        xr,yr=smote_enn_resample(pd.DataFrame(xtr_s,columns=list(xtr.columns)),ytr,seed); xr=np.asarray(xr); yr=np.asarray(yr)
    except ValueError:
        xr=np.asarray(xtr_s); yr=np.asarray(ytr)
    classes=int(meta['target_class_count'])
    xr=xr.reshape(xr.shape[0],xr.shape[1],1); xv_s=xv_s.reshape(xv_s.shape[0],xv_s.shape[1],1); xt_s=xt_s.reshape(xt_s.shape[0],xt_s.shape[1],1)
    yre=_one_hot(yr,classes); yve=_one_hot(yv,classes)
    tf.keras.backend.clear_session(); tf.keras.utils.set_random_seed(seed)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass
    shared_globals.init(); shared_globals.random_seed=seed; shared_globals.sample_num=xr.shape[0]
    t=time.perf_counter(); model=build_model(xr.shape[1:],classes,xr.shape[0]); model,h=train_model(model,xr,yre,xv_s,yve,xr.shape[0]); train_sec=time.perf_counter()-t
    t=time.perf_counter(); pred=np.argmax(model.predict(xt_s,verbose=0),axis=1); test_sec=time.perf_counter()-t
    out={'dataset':ds,'seed_id':sid,'seed':seed,'mode':mode,'dimension':int(len(actual_features)),'features':actual_features,
         'accuracy':float(accuracy_score(yt,pred)),'f1_macro':float(f1_score(yt,pred,average='macro',zero_division=0)),
         'train_seconds':float(train_sec),'test_seconds':float(test_sec),'best_epoch':int(np.argmin(h.history['val_loss'])+1),'epochs_ran':int(len(h.history['loss'])),
         'resampled_train_n':int(xr.shape[0]),'scaler_policy':'StandardScaler fit on train only; transform validation/test; same SMOTEENN + CNN',
         'scaler_mean':sc.mean_.tolist(),'scaler_scale':sc.scale_.tolist(),'split_hash':p1['split_hash'],'preprocessing':meta}
    tf.keras.backend.clear_session(); return out

shards=int(os.environ.get('EVAL_SHARDS','1')); shard_id=int(os.environ.get('EVAL_SHARD_ID','0')); status_file=OUT/f'status_worker{shard_id}.json'
status={'total_cases':len(manifest),'shards':shards,'shard_id':shard_id,'expected_evaluations':2*sum(1 for i in range(len(manifest)) if i%shards==shard_id),'completed_evaluations':0,'failed':[],'started':time.strftime('%Y-%m-%dT%H:%M:%S')}
for i,r in manifest.iterrows():
    if i % shards != shard_id: continue
    ds,sid=r.dataset,r.seed_id; rec=json.load(open(r.record_json)); selected=list(rec['selection']['final_features'])
    case_dir=OUT/'cases'/ds/sid; case_dir.mkdir(parents=True,exist_ok=True)
    for mode,features in [('full',None),('selected',selected)]:
        fp=case_dir/f'{mode}_scaled.json'
        if fp.exists(): status['completed_evaluations']+=1; continue
        try:
            result=scaled_eval(ds,sid,features,mode); fp.write_text(json.dumps(result,indent=2)); status['completed_evaluations']+=1
            print('DONE',ds,sid,mode,'dim',result['dimension'],'acc',round(result['accuracy'],6),'f1',round(result['f1_macro'],6),flush=True)
        except Exception as e:
            err={'dataset':ds,'seed_id':sid,'mode':mode,'error':repr(e),'traceback':traceback.format_exc()}; status['failed'].append(err); print('FAILED',ds,sid,mode,repr(e),flush=True)
        status['current']={'dataset':ds,'seed_id':sid,'mode':mode}; status_file.write_text(json.dumps(status,indent=2))
# aggregate
rows=[]
for _,r in manifest.iterrows():
    ds,sid=r.dataset,r.seed_id; case=OUT/'cases'/ds/sid
    fp,sp=case/'full_scaled.json',case/'selected_scaled.json'
    if not(fp.exists() and sp.exists()): continue
    f=json.load(open(fp)); s=json.load(open(sp)); rec=json.load(open(r.record_json))
    rows.append({'dataset':ds,'seed_id':sid,'source':r.source,'selected_dimension':s['dimension'],
      'raw_full_acc':float(r.full_accuracy),'raw_selected_acc':float(r.accuracy),'raw_full_f1':float(r.full_f1_macro),'raw_selected_f1':float(r.f1_macro),
      'scaled_full_acc':f['accuracy'],'scaled_selected_acc':s['accuracy'],'scaled_delta_acc_pp':100*(s['accuracy']-f['accuracy']),
      'scaled_full_f1':f['f1_macro'],'scaled_selected_f1':s['f1_macro'],'scaled_delta_f1_pp':100*(s['f1_macro']-f['f1_macro']),
      'full_scaling_gain_acc_pp':100*(f['accuracy']-float(r.full_accuracy)),'selected_scaling_gain_acc_pp':100*(s['accuracy']-float(r.accuracy)),
      'full_scaling_gain_f1_pp':100*(f['f1_macro']-float(r.full_f1_macro)),'selected_scaling_gain_f1_pp':100*(s['f1_macro']-float(r.f1_macro))})
df=pd.DataFrame(rows); df.to_csv(OUT/'case_summary.csv',index=False)
if len(df):
    agg=[]
    for ds,q in df.groupby('dataset',sort=False):
      row={'dataset':ds,'n':len(q)}
      for c in ['selected_dimension','scaled_full_acc','scaled_selected_acc','scaled_delta_acc_pp','scaled_full_f1','scaled_selected_f1','scaled_delta_f1_pp','full_scaling_gain_acc_pp','selected_scaling_gain_acc_pp','full_scaling_gain_f1_pp','selected_scaling_gain_f1_pp']:
       row[c+'_mean']=float(q[c].mean()); row[c+'_sd']=float(q[c].std(ddof=1)) if len(q)>1 else 0.0
      agg.append(row)
    pd.DataFrame(agg).to_csv(OUT/'dataset_summary.csv',index=False)
status['finished']=time.strftime('%Y-%m-%dT%H:%M:%S'); status['current']=None; status_file.write_text(json.dumps(status,indent=2)); print('COMPLETE',status['completed_evaluations'],'failed',len(status['failed']),flush=True)
