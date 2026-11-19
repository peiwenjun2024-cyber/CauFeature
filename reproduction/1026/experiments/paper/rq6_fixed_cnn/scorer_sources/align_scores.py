"""After all fits, align JS numeric precision with production; never retrain."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,concurrent.futures as cf,multiprocessing as mp,time,contextlib
from common import *
from exact_scoring import exact_scores

def align(ds,sid,model):
    from worker import validate_shared,high_fn,init_gpu
    dest=case_dir(ds,sid);path=dest/(model+'.json');record=load(path)
    if record.get('numeric_precision_alignment',{}).get('completed'):return 'cached'
    with open(OUT/'logs'/f'{ds}_{sid}_{model}_alignment.log','a',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        meta,z=validate_shared(dest);t=time.perf_counter()
        if model=='MLP':
            tf=init_gpu();fitted=tf.keras.models.load_model(dest/'MLP.keras',compile=False)
            def predict(x):return fitted(np.asarray(x,dtype=np.float32),training=False).numpy()
        else:
            import joblib
            fitted=joblib.load(dest/'RF.joblib');fitted.n_jobs=4
            def predict(x):return fitted.predict_proba(x)
        scores,queries=exact_scores(predict,z,meta,batch_size=8192 if model=='MLP' else 16384)
        ev=evidence(scores,meta['feature_names'],z['nonconstant'],high_fn(ds))
        oldscores=np.asarray(record['baseline_scores'])
        previous_h=set(record['high_contrib_features']);previous_a=set(record['p2_gate_features'])
        # Keep original preliminary vectorized score record for a transparent audit.
        save_json(dest/'audit'/(model+'_before_precision_alignment.json'),record)
        record.update(ev)
        record['numeric_precision_alignment']=dict(completed=True,model_retrained=False,
            policy='original reference preserves predictor probability dtype; bootstrap JS casts float64, as production',
            max_absolute_score_change=float(np.max(np.abs(scores-oldscores))),
            fh_symmetric_difference=sorted(previous_h^set(ev['high_contrib_features'])),
            p2_gate_symmetric_difference=sorted(previous_a^set(ev['p2_gate_features'])),
            elapsed_sec=time.perf_counter()-t,query_count=queries,
            alignment_script_sha256=sha(__file__),scoring_script_sha256=sha(CODE/'exact_scoring.py'))
        save_json(path,record)
        if model=='MLP':tf.keras.backend.clear_session()
        return dict(dataset=ds,seed_id=sid,model=model,seconds=time.perf_counter()-t,
                    max_change=record['numeric_precision_alignment']['max_absolute_score_change'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true')
    ap.add_argument('--models',nargs='+',choices=['MLP','RF'],default=['MLP','RF']);a=ap.parse_args()
    if not a.execute:raise SystemExit('Pass --execute')
    ctx=mp.get_context('spawn');pending={};errors=[];done=0;t=time.perf_counter()
    with cf.ProcessPoolExecutor(4,mp_context=ctx) as gpu,cf.ProcessPoolExecutor(2,mp_context=ctx) as cpu:
        for ds in DATASETS:
            for seed in range(1,31):
                sid=f'S{seed:02d}'
                for model,pool in [('MLP',gpu),('RF',cpu)]:
                    if model not in a.models:continue
                    future=pool.submit(align,ds,sid,model);pending[future]=(ds,sid,model)
        while pending:
            ready,_=cf.wait(pending,timeout=15,return_when=cf.FIRST_COMPLETED)
            for future in ready:
                case=pending.pop(future)
                try:print('ALIGNED',case,future.result(),flush=True);done+=1
                except Exception as e:errors.append(dict(case=case,error=repr(e)));print('FAILED',case,repr(e),flush=True)
            progress=dict(completed=done,pending=len(pending),errors=errors,
                complete=not pending and not errors,models=a.models,elapsed_sec=time.perf_counter()-t)
            save_json(OUT/('alignment_'+'_'.join(a.models)+'_status.json'),progress)
    finished=0
    for ds in DATASETS:
        for seed in range(1,31):
            for model in ['MLP','RF']:
                p=case_dir(ds,f'S{seed:02d}')/(model+'.json')
                if p.exists() and load(p).get('numeric_precision_alignment',{}).get('completed'):finished+=1
    save_json(OUT/'alignment_status.json',dict(completed=finished,expected=300,
        complete=finished==300 and not errors,errors=errors))
    if errors:raise SystemExit(str(errors))

if __name__=='__main__':main()
