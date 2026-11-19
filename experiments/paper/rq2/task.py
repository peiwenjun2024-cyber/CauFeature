#!/usr/bin/env python3
"""Isolated train-only native selection and scaled CNN evaluation."""
import os
for k,v in {'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','TF_CPP_MIN_LOG_LEVEL':'2','CUDA_VISIBLE_DEVICES':('0' if len(__import__('sys').argv)>1 and __import__('sys').argv[1]=='evaluate' else '-1')}.items():os.environ[k]=v
import argparse,hashlib,json,sys,time,traceback,tempfile,subprocess
from pathlib import Path
import numpy as np,pandas as pd
PACKAGE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PACKAGE/'scripts'))
from isolated_worker import finish_worker, required_output
ROOT=Path(os.environ.get('FAST30_SOURCE_ROOT','/root/autodl-tmp/fast30_final')).resolve()
MANIFEST=Path(os.environ.get('RQ2_MANIFEST',ROOT/'explainable_5ds_s30_20261012/manifest.json'))
HOME=required_output('RQ2_OUTPUT_ROOT',forbidden=(ROOT/'results',ROOT/'explainable_5ds_s30_20261012'))
HOME.mkdir(parents=True,exist_ok=True)
CODE=PACKAGE/'CauFeature'
sys.path[:0]=[str(CODE),str(PACKAGE/'experiments/baselines/explainable')]
from fixed_split_evaluator import _prepare_fixed_splits,_one_hot
from sklearn.preprocessing import StandardScaler
def atomic(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str));tmp.replace(path)
def locate(ds,sid,method):
    manifest=json.loads(MANIFEST.read_text())
    return next(r for r in manifest['rows'] if r['dataset']==ds and r['seed_id']==sid and r['method']==method)
def load(row):
    split=pd.read_csv(row['split_path'])
    ix=tuple(split.loc[split.partition==p,'row_index'].to_numpy(int) for p in ('train','validation','test'))
    x,v,t,y,yv,yt,meta=_prepare_fixed_splits(Path(row['data_path']),*ix,None)
    return ix,x,v,t,y,yv,yt,meta
def native(method,row,ix,x,v,y,seed):
    if method in ('SHAP','LIME','FeatureX'):
        import tensorflow as tf,shared_globals
        from model_train_v6 import smote_enn_resample,build_model,train_model
        from run_explainable_experiments import _select_shap,_select_lime,_select_featurex
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
        scaler=StandardScaler();a=scaler.fit_transform(x);b=scaler.transform(v)
        xx=pd.DataFrame(a,columns=x.columns);vv=pd.DataFrame(b,columns=x.columns)
        try:rx,ry=smote_enn_resample(xx,y,seed)
        except ValueError:rx,ry=a,y
        rx=np.asarray(rx).reshape(-1,x.shape[1],1);ry=np.asarray(ry)
        shared_globals.init();shared_globals.random_seed=seed;shared_globals.sample_num=len(rx)
        m=build_model(rx.shape[1:],int(max(y)+1),len(rx))
        m,_=train_model(m,rx,_one_hot(ry,int(max(y)+1)),b.reshape(-1,x.shape[1],1),_one_hot(yv_cache,int(max(y)+1)),len(rx))
        if method=='SHAP':selected,scores=_select_shap(m,xx,vv,seed)
        elif method=='LIME':selected,scores=_select_lime(m,xx,vv,seed)
        else:
            # Spearman drops constant columns in some scipy versions; preserve feature indices.
            import run_explainable_experiments as fx
            from scipy.stats import spearmanr
            def stable_components(frame):
                cols=list(frame.columns); adjacency={name:set() for name in cols}
                values=frame.to_numpy(dtype=float)
                for i in range(len(cols)):
                    if not np.isfinite(values[:,i]).all() or np.ptp(values[:,i])==0:continue
                    for j in range(i+1,len(cols)):
                        if not np.isfinite(values[:,j]).all() or np.ptp(values[:,j])==0:continue
                        corr=spearmanr(values[:,i],values[:,j]).statistic
                        if np.isfinite(corr) and abs(corr)>=0.7:
                            adjacency[cols[i]].add(cols[j]);adjacency[cols[j]].add(cols[i])
                components=[];seen=set()
                for name in cols:
                    if name in seen or not adjacency[name]:continue
                    stack=[name];component=set()
                    while stack:
                        n=stack.pop()
                        if n in component:continue
                        component.add(n);stack.extend(adjacency[n]-component)
                    seen.update(component);components.append(component)
                return components
            fx._featurex_components=stable_components
            selected,scores=fx._select_featurex(xx,y,m,seed)
        del m;tf.keras.backend.clear_session()
        return list(selected),{'score_count':len(np.asarray(scores).reshape(-1)),'selector_model':'CNN surrogate'}
    from revised_protocol.run_s10_external_mb_baselines import ppfs_select,octave_select
    import argparse as ap
    params=ap.Namespace(ppfs_num_simul=5,ppfs_n_jobs=1,bamb_max_k=3,octave_timeout=1800)
    if method=='PPFS':
        result=ppfs_select(x,y,np.arange(len(x)),seed,params)
    else:
        # History helper uses native Octave code. Its temporary directory must exist.
        from revised_protocol import run_s10_external_mb_baselines as mb
        mb.RAW=HOME/'scratch';mb.RAW.mkdir(parents=True,exist_ok=True)
        vendor=Path(os.environ.get('RQ2_THIRD_PARTY_ROOT',ROOT/'code_revised_20260910/third_party'))
        mb.CAUSAL=vendor/'CausalLearner/source/Causal_Learner'
        mb.EAMB=vendor/'EAMB/EAMB-code (matlab)'
        entry=mb.CAUSAL/'alg_MB/BAMB/BAMB_Z.m' if method=='BAMB-Z' else mb.EAMB/'EAMB.m'
        if not entry.is_file():
            raise FileNotFoundError(entry)
        result=octave_select(method,x,y,np.arange(len(x)),params)
    return list(result),{'selector_model':'native Markov blanket'}
def select(row):
    ds,sid,m=row['dataset'],row['seed_id'],row['method']
    out=HOME/'results'/m/ds/sid/'selection.json'
    if out.exists():
        try:
            q=json.loads(out.read_text())
            if q['status']=='complete' and len(q['features'])==q['selected_dimension'] and q['features']:return
        except Exception:pass
    ts=time.time();seed=int(sid[1:])
    try:
        ix,x,v,t,y,yv,yt,meta=load(row)
        global yv_cache;yv_cache=yv
        selected,extra=native(m,row,ix,x,v,y,seed)
        selected=[str(f) for f in selected]
        if not selected or len(set(selected))!=len(selected) or not set(selected)<=set(x.columns):
            raise ValueError('empty, duplicate or unknown selected features')
        obj=dict(status='complete',dataset=ds,seed_id=sid,method=m,features=selected,selected_dimension=len(selected),original_dimension=len(x.columns),reduction_rate=1-len(selected)/len(x.columns),selection_seconds=time.time()-ts,split_file_sha256=row['split_file_sha256'],selection_scope='train_only',selection_policy='native_method_no_budget_matching',details=extra)
        atomic(out,obj);print('SELECTED',ds,sid,m,len(selected),flush=True)
    except Exception as e:
        atomic(out.with_name('selection_failed.json'),dict(status='failed',error=str(e),traceback=traceback.format_exc()))
        print('SELECTION_FAILED',ds,sid,m,str(e),flush=True);raise
def evaluate(row):
    ds,sid,m=row['dataset'],row['seed_id'],row['method']
    root=HOME/'results'/m/ds/sid;out=root/'scaled_eval.json'
    if out.exists():
        try:
            q=json.loads(out.read_text())
            if q['status']=='complete':return
        except Exception:pass
    selection=json.loads((root/'selection.json').read_text())
    assert selection['status']=='complete' and selection['split_file_sha256']==row['split_file_sha256']
    selected=selection['features'];seed=int(sid[1:])
    # Identical ordered inputs use the canonical Original evaluation, not a second fit.
    original_root = os.environ.get('RQ2_ORIGINAL_EVAL_ROOT')
    if original_root:
        original_path = Path(original_root) / ds / sid / 'full_scaled.json'
    elif ds == 'Optdigits':
        original_path = ROOT / f'results/alpha001_full30_20261009/Optdigits/{sid}/full_scaled.json'
    else:
        original_path = ROOT / f'results/scaled_evaluator_full_vs_selected_20261010/cases/{ds}/{sid}/full_scaled.json'
    if original_path.exists():
        original = json.loads(original_path.read_text())
        if selected == original['features']:
            assert original['dataset'] == ds and original['seed_id'] == sid
            assert original['split_hash'] == row['split_file_sha256']
            obj = dict(status='complete', dataset=ds, seed_id=sid, method=m,
                       accuracy=original['accuracy'], f1_macro=original['f1_macro'],
                       selected_dimension=len(selected), original_dimension=selection['original_dimension'],
                       reduction_rate=selection['reduction_rate'], selected_features=selected,
                       selection_seconds=selection['selection_seconds'], evaluation_seconds=0.0,
                       split_file_sha256=row['split_file_sha256'],
                       preprocessing='train_only StandardScaler; train_only SMOTEENN',
                       evaluation_reuse=dict(reused_original=True, original_record=str(original_path),
                                             rule='identical ordered features and seed-matched split'))
            atomic(out, obj)
            print('REUSED_ORIGINAL', ds, sid, m, flush=True)
            return
    ts=time.time()
    try:
        import tensorflow as tf,shared_globals
        from model_train_v6 import smote_enn_resample,build_model,train_model
        from sklearn.metrics import accuracy_score,f1_score
        ix=tuple(pd.read_csv(row['split_path']).query('partition == @p')['row_index'].to_numpy(int) for p in ('train','validation','test'))
        x,v,t,y,yv,yt,meta=_prepare_fixed_splits(Path(row['data_path']),*ix,selected)
        sc=StandardScaler();a=sc.fit_transform(x);b=sc.transform(v);c=sc.transform(t)
        try:rx,ry=smote_enn_resample(pd.DataFrame(a,columns=selected),y,seed)
        except ValueError:rx,ry=a,y
        rx=np.asarray(rx);ry=np.asarray(ry);classes=int(meta['target_class_count'])
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
        shared_globals.init();shared_globals.random_seed=seed;shared_globals.sample_num=len(rx)
        model=build_model((len(selected),1),classes,len(rx))
        model,_=train_model(model,rx.reshape(-1,len(selected),1),_one_hot(ry,classes),b.reshape(-1,len(selected),1),_one_hot(yv,classes),len(rx))
        pred=np.argmax(model.predict(c.reshape(-1,len(selected),1),verbose=0),axis=1)
        obj=dict(status='complete',dataset=ds,seed_id=sid,method=m,accuracy=float(accuracy_score(yt,pred)),f1_macro=float(f1_score(yt,pred,average='macro',zero_division=0)),selected_dimension=len(selected),original_dimension=selection['original_dimension'],reduction_rate=selection['reduction_rate'],selected_features=selected,selection_seconds=selection['selection_seconds'],evaluation_seconds=time.time()-ts,split_file_sha256=row['split_file_sha256'],preprocessing='train_only StandardScaler; train_only SMOTEENN')
        atomic(out,obj);tf.keras.backend.clear_session();print('EVALUATED',ds,sid,m,obj['accuracy'],flush=True)
    except Exception as e:
        atomic(root/'evaluation_failed.json',dict(status='failed',error=str(e),traceback=traceback.format_exc()))
        print('EVALUATION_FAILED',ds,sid,m,str(e),flush=True);raise
def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['select','evaluate']);p.add_argument('dataset');p.add_argument('seed');p.add_argument('method');a=p.parse_args()
    row=locate(a.dataset,a.seed,a.method)
    with Path(row['split_path']).open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest() != row['split_file_sha256']:
            raise RuntimeError('Fixed split hash mismatch')
    if a.stage=='select':select(row)
    else:
        import fcntl
        with (HOME/'gpu_evaluation.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:evaluate(row)
            finally:fcntl.flock(lock,fcntl.LOCK_UN)
if __name__=='__main__':
    main()
    finish_worker(0)
