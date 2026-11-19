import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import contextlib, io, time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.combine import SMOTEENN
from common import *

def prepare_case(ds, sid):
    start=time.perf_counter(); dest=case_dir(ds,sid)
    if (dest/'shared.json').exists(): return str(dest)
    p=source(ds,sid); p1=load(p); seed=int(p1['seed'])
    assert sha(p1['selector_model']) == p1['selector_model_sha256']
    raw=pd.read_csv(p1['selector_input'])
    x,y=raw.iloc[:,:-1].copy(),raw.iloc[:,-1].copy()
    assert x.columns.tolist() == p1['feature_names']
    encoders={}
    for col in x.select_dtypes(exclude=['number']).columns:
        le=LabelEncoder();x[col]=le.fit_transform(x[col])
        encoders[col]=[str(v) for v in le.classes_]
    if x.isnull().any().any():x=x.fillna(x.mean())
    target=LabelEncoder(); y=pd.Series(target.fit_transform(y),index=y.index,name=y.name)
    # train_test_split row order and absence of stratify match train() exactly.
    tr,va=train_test_split(np.arange(len(x)),test_size=.2,random_state=seed)
    fn=import_function(source_code(ds)/'model_train_v6.py','smote_enn_resample',
        dict(SMOTE=SMOTE,EditedNearestNeighbours=EditedNearestNeighbours,SMOTEENN=SMOTEENN,pd=pd))
    with contextlib.redirect_stdout(io.StringIO()):
        xt,yt=fn(x.iloc[tr],y.iloc[tr],random_state=seed)
    refs=[]
    def reference(indices):
        frame=x if indices is None else x.iloc[indices]
        vals=[]
        for name in x.columns:
            v=pd.to_numeric(frame[name],errors='coerce').dropna()
            if p1['baseline_policy'][name]=='mode':
                m=v.mode(dropna=True);value=float(m.iloc[0]) if not m.empty else np.nan
            elif p1['baseline_policy'][name]=='mean':value=float(v.mean()) if not v.empty else np.nan
            else:raise RuntimeError('Unknown baseline policy')
            if not np.isfinite(value):value=float(pd.to_numeric(x[name],errors='coerce').dropna().mean())
            vals.append(np.float32(value))
        return np.asarray(vals,dtype=np.float32)
    original=reference(None)
    frozen=np.asarray([p1['baseline_values'][n] for n in x.columns],dtype=np.float32)
    if not np.array_equal(original,frozen): raise RuntimeError('Frozen baseline reconstruction mismatch')
    refs.append(original)
    rng=np.random.default_rng(seed)
    bootstrap=np.asarray([rng.integers(0,len(x),size=max(2,int(round(len(x)*.8)))) for _ in range(4)])
    refs.extend(reference(idx) for idx in bootstrap)
    # Constant mask matches stage2_runtime's encoded development data.
    nonconstant=(x.nunique(dropna=False).to_numpy()>1)
    arrays=dict(x_train=xt.to_numpy(),y_train=yt.to_numpy(),
                x_val=x.iloc[va].to_numpy(),y_val=y.iloc[va].to_numpy(),
                train_indices=tr,validation_indices=va,bootstrap_indices=bootstrap,
                references=np.asarray(refs),nonconstant=nonconstant)
    dest.mkdir(parents=True,exist_ok=True)
    tmp=dest/'shared.tmp.npz';np.savez(tmp,**arrays);os.replace(tmp,dest/'shared.npz')
    meta=dict(protocol=PROTOCOL,dataset=ds,seed_id=sid,seed=seed,
              feature_names=p1['feature_names'],target_classes=[str(v) for v in target.classes_],
              source_p1=str(p),source_p1_sha256=sha(p),source_model=p1['selector_model'],
              source_model_sha256=p1['selector_model_sha256'],
              selector_input=p1['selector_input'],selector_input_sha256=sha(p1['selector_input']),
              split_file=p1['split_file'],split_hash=p1['split_hash'],
              source_dataset_sha256=p1['dataset_sha256'],
              input_scaling='none; identical encoded unscaled input for all models',
              sample_num=len(x),resampled_train_num=len(xt),validation_num=len(va),
              encoders=encoders,perturbation_values=p1['perturbation_values'],
              baseline_policy=p1['baseline_policy'],
              nonconstant_features=[n for n,b in zip(x.columns,nonconstant) if b],
              hashes={k:array_sha(v) for k,v in arrays.items()},
              shared_npz_sha256=sha(dest/'shared.npz'),outer_test_rows_used=False,
              production_train_code_sha256=sha(source_code(ds)/'model_train_v6.py'),
              preparation_sec=time.perf_counter()-start)
    save_json(dest/'shared.json',meta)
    return str(dest)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('dataset');p.add_argument('seed_id');a=p.parse_args()
    print(prepare_case(a.dataset,a.seed_id))
