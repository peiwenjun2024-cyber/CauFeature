#!/usr/bin/env python3
from pathlib import Path
import json,sys,time,traceback,os
import numpy as np,pandas as pd
from fresh_cases import load_fresh_case
PACKAGE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PACKAGE/'scripts'))
from isolated_worker import finish_worker, required_output
ROOT=Path(os.environ.get('FAST30_SOURCE_ROOT','/root/autodl-tmp/fast30_final')).resolve()
CODE=PACKAGE/'CauFeature'
MAIN=ROOT/'results/alpha001_full30_20261009'
OLD=ROOT/'results/fast30_semanticperturb_flat_v6_20260930'
DATA=ROOT/'code_revised_20260910/dataset'
OUT=required_output('FAST30_EVAL_OUTPUT_ROOT',forbidden=(ROOT/'results',))
OUT.mkdir(parents=True,exist_ok=True); (OUT/'cases').mkdir(exist_ok=True)
sys.path.insert(0,str(CODE))
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,f1_score
from fixed_split_evaluator import _prepare_fixed_splits,_one_hot
from model_train_v6 import smote_enn_resample,build_model,train_model
import shared_globals

def archived_case(row):
    ds, sid = row.dataset, row.seed_id
    with Path(row.record_json).open(encoding='utf-8') as stream:
        record = json.load(stream)
    with (OLD / ds / sid / 'p1_cache.json').open(encoding='utf-8') as stream:
        p1 = json.load(stream)
    raw = {
        'raw_full_acc': float(row.full_accuracy), 'raw_selected_acc': float(row.accuracy),
        'raw_full_f1': float(row.full_f1_macro), 'raw_selected_f1': float(row.f1_macro),
    }
    return DATA / f'{ds}.csv', p1, record, row.source, raw


def fresh_case(cases_root, ds, sid):
    data, p1, record = load_fresh_case(ROOT, cases_root, ds, sid)
    full, selected = record['paired_full_feature_baseline'], record['metrics']
    raw = {
        'raw_full_acc': float(full['accuracy']), 'raw_selected_acc': float(selected['accuracy']),
        'raw_full_f1': float(full['f1_macro']), 'raw_selected_f1': float(selected['f1_macro']),
    }
    return data, p1, record, 'fresh_rq1', raw


def case_loaders():
    ds = os.environ.get('FAST30_EVAL_DATASET')
    sid = os.environ.get('FAST30_EVAL_SEED')
    fresh_root = os.environ.get('FAST30_FRESH_CASES_ROOT')
    if fresh_root:
        if not ds or not sid:
            raise SystemExit('Set FAST30_EVAL_DATASET and FAST30_EVAL_SEED for a fresh RQ1 case.')
        if ds == 'Optdigits':
            raise SystemExit('Use experiments/paper/rq1_optdigits/run_case.py for the frozen Optdigits evaluator.')
        return [(ds, sid, lambda: fresh_case(Path(fresh_root).resolve(), ds, sid))]
    manifest = pd.read_csv(os.environ.get('FAST30_EVAL_MANIFEST', MAIN/'official_main_results_15x30_manifest.csv'))
    if ds:
        manifest = manifest[manifest.dataset == ds]
    if sid:
        manifest = manifest[manifest.seed_id == sid]
    if manifest.empty:
        raise SystemExit('No matching fixed-split cases')
    return [(r.dataset, r.seed_id, lambda r=r: archived_case(r)) for r in manifest.itertuples()]


def scaled_eval(ds,sid,features,mode,data_path,p1):
    seed=int(sid[1:])
    split=pd.read_csv(p1['split_file'])
    idx={k:split.loc[split.partition==k,'row_index'].to_numpy(int) for k in ['train','validation','test']}
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
cases=case_loaders()
status={'total_cases':len(cases),'shards':shards,'shard_id':shard_id,'expected_evaluations':2*sum(1 for i in range(len(cases)) if i%shards==shard_id),'completed_evaluations':0,'failed':[],'started':time.strftime('%Y-%m-%dT%H:%M:%S')}
for i,(ds,sid,load_case) in enumerate(cases):
    if i % shards != shard_id: continue
    data_path,p1,rec,_,_=load_case()
    selected=list(rec['selection']['final_features'])
    case_dir=OUT/'cases'/ds/sid; case_dir.mkdir(parents=True,exist_ok=True)
    for mode,features in [('full',None),('selected',selected)]:
        fp=case_dir/f'{mode}_scaled.json'
        if fp.exists(): status['completed_evaluations']+=1; continue
        try:
            result=scaled_eval(ds,sid,features,mode,data_path,p1); fp.write_text(json.dumps(result,indent=2)); status['completed_evaluations']+=1
            print('DONE',ds,sid,mode,'dim',result['dimension'],'acc',round(result['accuracy'],6),'f1',round(result['f1_macro'],6),flush=True)
        except Exception as e:
            err={'dataset':ds,'seed_id':sid,'mode':mode,'error':repr(e),'traceback':traceback.format_exc()}; status['failed'].append(err); print('FAILED',ds,sid,mode,repr(e),flush=True)
        status['current']={'dataset':ds,'seed_id':sid,'mode':mode}; status_file.write_text(json.dumps(status,indent=2))
# aggregate
rows=[]
for ds,sid,load_case in cases:
    case=OUT/'cases'/ds/sid
    fp,sp=case/'full_scaled.json',case/'selected_scaled.json'
    if not(fp.exists() and sp.exists()): continue
    f=json.load(open(fp)); s=json.load(open(sp)); _,_,_,source,raw=load_case()
    rows.append({'dataset':ds,'seed_id':sid,'source':source,'selected_dimension':s['dimension'],**raw,
      'scaled_full_acc':f['accuracy'],'scaled_selected_acc':s['accuracy'],'scaled_delta_acc_pp':100*(s['accuracy']-f['accuracy']),
      'scaled_full_f1':f['f1_macro'],'scaled_selected_f1':s['f1_macro'],'scaled_delta_f1_pp':100*(s['f1_macro']-f['f1_macro']),
      'full_scaling_gain_acc_pp':100*(f['accuracy']-raw['raw_full_acc']),'selected_scaling_gain_acc_pp':100*(s['accuracy']-raw['raw_selected_acc']),
      'full_scaling_gain_f1_pp':100*(f['f1_macro']-raw['raw_full_f1']),'selected_scaling_gain_f1_pp':100*(s['f1_macro']-raw['raw_selected_f1'])})
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
finish_worker(1 if status['failed'] or status['completed_evaluations']!=status['expected_evaluations'] else 0)
