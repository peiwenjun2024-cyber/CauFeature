import sys,json,os,argparse,traceback,time
from pathlib import Path
import numpy as np,pandas as pd
from fresh_cases import load_fresh_case
PACKAGE=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PACKAGE/'scripts'))
from isolated_worker import finish_worker, required_output
R=Path(os.environ.get('FAST30_SOURCE_ROOT','/root/autodl-tmp/fast30_final')).resolve()
fresh_root=os.environ.get('FAST30_FRESH_CASES_ROOT')
C=PACKAGE/'CauFeature'
sys.path.insert(0,str(C))
from fixed_split_evaluator import _prepare_fixed_splits,evaluate_fixed_split
from sklearn.feature_selection import mutual_info_classif,chi2,RFE
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score,f1_score
from sklearn.preprocessing import StandardScaler
from fixed_split_evaluator import _one_hot
from model_train_v6 import smote_enn_resample,build_model,train_model
import tensorflow as tf,shared_globals
try:
 for g in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(g,True)
except Exception:pass
ap=argparse.ArgumentParser();ap.add_argument('--datasets',nargs='+',default=['StudentsAcademic','SoftwareQuality','OnlineShoppers','Optdigits','BankFull']);ap.add_argument('--seeds',nargs='+',default=['S01']);ap.add_argument('--methods',nargs='+',default=['VT','MI','CHI','RFC','RANDOM','RFE']);ap.add_argument('--modes',nargs='+',default=['scaled']);a=ap.parse_args()
O=required_output('FAST30_MATCHED_OUTPUT_ROOT',forbidden=(R/'results',));O.mkdir(parents=True,exist_ok=True)
failed=False

def run_scaled(path,ix,features,seed):
 xt,xv,xe,yt,yv,ye,meta=_prepare_fixed_splits(path,*ix,features)
 sc=StandardScaler();xts=sc.fit_transform(xt);xvs=sc.transform(xv);xes=sc.transform(xe)
 try: xr,yr=smote_enn_resample(pd.DataFrame(xts,columns=xt.columns),yt,seed);xr=np.asarray(xr);yr=np.asarray(yr)
 except ValueError:xr=np.asarray(xts);yr=np.asarray(yt)
 n=int(meta['target_class_count']);xr=xr.reshape(-1,xr.shape[1],1);xvs=xvs.reshape(-1,xvs.shape[1],1);xes=xes.reshape(-1,xes.shape[1],1)
 tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed)
 try:tf.config.experimental.enable_op_determinism()
 except Exception:pass
 shared_globals.init();shared_globals.random_seed=seed;shared_globals.sample_num=len(xr)
 model=build_model(xr.shape[1:],n,len(xr));model,h=train_model(model,xr,_one_hot(yr,n),xvs,_one_hot(yv,n),len(xr))
 pred=np.argmax(model.predict(xes,verbose=0),axis=1)
 return {'accuracy':float(accuracy_score(ye,pred)),'f1_macro':float(f1_score(ye,pred,average='macro',zero_division=0)),'selected_dimension':len(features)}

for ds in a.datasets:
 for sid in a.seeds:
  try:
   if fresh_root:
    data,old,main=load_fresh_case(R,Path(fresh_root).resolve(),ds,sid)
   else:
    old=json.load(open((R/'results/alpha001_full30_20261009/Optdigits' if ds=='Optdigits' else R/'results/fast30_semanticperturb_flat_v6_20260930'/ds)/sid/'p1_cache.json'))
    manifest=pd.read_csv(R/'results/alpha001_full30_20261009/official_main_results_15x30_manifest.csv'); matched=manifest[(manifest.dataset==ds)&(manifest.seed_id==sid)];assert len(matched)==1 or ds=='Optdigits';main=json.load(open(R/'results/alpha001_full30_20261009/Optdigits'/sid/'record.json' if ds=='Optdigits' else matched.iloc[0].record_json))
    data=(R/'pilot_optdigits_30_20261011/dataset/Optdigits.csv' if ds=='Optdigits' else R/'code_revised_20260910/dataset'/f'{ds}.csv')
   k=len(main['selection']['final_features']);seed=int(sid[1:]);split=pd.read_csv(old['split_file']);ix=tuple(split.loc[split.partition==part,'row_index'].to_numpy(int) for part in ['train','validation','test'])
   xtr,xv,xe,ytr,yv,ye,meta=_prepare_fixed_splits(data,*ix,None)
   x_selector=xtr;y_selector=ytr
   # Fit rankers only on training partition, not validation/test.
   for method in a.methods:
    out=O/method/ds/sid;out.mkdir(parents=True,exist_ok=True);sel=out/'selection.json'
    if sel.exists():features=json.load(open(sel))['features']
    else:
     X=x_selector.to_numpy(float)
     if method=='VT':rank=np.argsort(-np.var(X,axis=0),kind='stable')
     elif method=='MI':rank=np.argsort(-mutual_info_classif(X,y_selector,random_state=seed),kind='stable')
     elif method=='CHI':rank=np.argsort(-chi2(MinMaxScaler().fit_transform(X),y_selector)[0],kind='stable')
     elif method=='RANDOM':rank=np.random.default_rng(seed).permutation(X.shape[1])
     elif method=='RFC':
      clf=RandomForestClassifier(n_estimators=100,random_state=seed,n_jobs=4);clf.fit(X,y_selector);rank=np.argsort(-clf.feature_importances_,kind='stable')
     elif method=='RFE':
      clf=RandomForestClassifier(n_estimators=100,random_state=seed,n_jobs=4)
      fitted=RFE(clf,n_features_to_select=k,step=1).fit(X,y_selector);rank=np.flatnonzero(fitted.support_)
     else:raise ValueError(method)
     features=[x_selector.columns[i] for i in rank[:k]];assert len(features)==k and len(set(features))==k
     sel.write_text(json.dumps({'dataset':ds,'seed_id':sid,'method':method,'budget_k':k,'features':features,'split_hash':old['split_hash'],'selection_fit':'train partition only','selection_protocol':'native VT/MI/CHI/RFE/RFC/RANDOM; exact CauFeature seed-wise topk','scoring_preprocessing':'CHI only uses MinMax; other methods use original encoded values','status':'complete'},indent=2))
     print('SELECTED',ds,sid,method,k,flush=True)
    for mode in a.modes:
     fp=out/f'{mode}_eval.json'
     if fp.exists():
      try:
       j=json.load(open(fp)); assert j['budget_k']==k and len(j['selected_features'])==k and j['split_hash']==old['split_hash'];continue
      except Exception: fp.rename(fp.with_suffix('.invalid.json'))
     print('START',ds,sid,method,mode,flush=True)
     try:
      if mode=='raw':res=evaluate_fixed_split(data,*ix,seed,features)
      else:res=run_scaled(data,ix,features,seed)
      res.update({'dataset':ds,'seed_id':sid,'method':method,'mode':mode,'budget_k':k,'selected_features':features,'split_hash':old['split_hash']})
      fp.write_text(json.dumps(res,indent=2));print('DONE',ds,sid,method,mode,round(res['accuracy'],6),round(res['f1_macro'],6),flush=True)
     except Exception as exc:failed=True;print('EVALUATION_FAILED',ds,sid,method,mode,str(exc),traceback.format_exc(),flush=True)
  except Exception as exc:failed=True;print('CASE_FAILED',ds,sid,str(exc),traceback.format_exc(),flush=True)
if any(not (O/m/d/s/f'{mode}_eval.json').is_file() for m in a.methods for d in a.datasets for s in a.seeds for mode in a.modes):
 failed=True
finish_worker(int(failed))
