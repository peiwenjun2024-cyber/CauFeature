#!/usr/bin/env python3
"""S01 smoke gate then full set; crash-safe per-case records and single evaluator."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,concurrent.futures,json,subprocess,sys,time,math
from pathlib import Path
PACKAGE=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(PACKAGE/'scripts'))
from isolated_worker import required_output
TASK=Path(__file__).resolve().with_name('task.py')
ROOT=Path(os.environ.get('FAST30_SOURCE_ROOT','/root/autodl-tmp/fast30_final')).resolve()
MANIFEST=Path(os.environ.get('RQ2_MANIFEST',ROOT/'explainable_5ds_s30_20261012/manifest.json'))
HOME=required_output('RQ2_OUTPUT_ROOT',forbidden=(ROOT/'results',ROOT/'explainable_5ds_s30_20261012'))
HOME.mkdir(parents=True,exist_ok=True)
METHODS=('SHAP','LIME','FeatureX','PPFS','BAMB-Z','EAMB-Z')
def valid(row,stage):
    name='selection.json' if stage=='select' else 'scaled_eval.json'
    f=HOME/'results'/row['method']/row['dataset']/row['seed_id']/name
    try:
        j=json.loads(f.read_text())
        if j['status']!='complete' or j['split_file_sha256']!=row['split_file_sha256']:return False
        if j.get('dataset')!=row['dataset'] or j.get('seed_id')!=row['seed_id'] or j.get('method')!=row['method']:return False
        if stage=='select':
            fs=j.get('features',[])
            return bool(fs) and len(fs)==len(set(fs))==j['selected_dimension'] and 0<=j['reduction_rate']<1 and j.get('selection_policy')=='native_method_no_budget_matching' and math.isfinite(j['selection_seconds'])
        sf=HOME/'results'/row['method']/row['dataset']/row['seed_id']/'selection.json'
        s=json.loads(sf.read_text());fs=s['features']
        return valid(row,'select') and j['selected_features']==fs and j['selected_dimension']==len(fs) and all(math.isfinite(j[k]) and 0<=j[k]<=1 for k in ('accuracy','f1_macro')) and math.isfinite(j['evaluation_seconds']) and j['evaluation_seconds']>=0
    except Exception:return False
def run(row,stage):
    name=f"{row['dataset']}_{row['seed_id']}_{row['method']}_{stage}.log"
    log=HOME/'logs'/name;log.parent.mkdir(exist_ok=True)
    if valid(row,stage):return True
    command=[sys.executable,'-u',str(TASK),stage,row['dataset'],row['seed_id'],row['method']]
    with log.open('a') as f:
        f.write(f"START {time.strftime('%F %T')} {command}\n");f.flush()
        r=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,cwd=HOME,timeout=7200)
        f.write(f"EXIT {time.strftime('%F %T')} code={r.returncode}\n")
    data_valid=valid(row,stage)
    audit=HOME/'exit_audit.jsonl'
    with audit.open('a') as f:
        f.write(json.dumps(dict(dataset=row['dataset'],seed_id=row['seed_id'],method=row['method'],stage=stage,exit_code=r.returncode,result_valid=data_valid,classification=('clean_success' if data_valid and r.returncode==0 else 'valid_result_abnormal_exit' if data_valid else 'invalid_or_missing_result'),log=str(log),at=time.strftime('%F %T')))+'\n')
    if data_valid and r.returncode!=0:print('POST_SAVE_ABNORMAL_EXIT',row['dataset'],row['seed_id'],row['method'],stage,'code',r.returncode,flush=True)
    return data_valid and r.returncode==0
def phase(rows,workers):
    failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(run,r,'select'):r for r in rows}
        for f in concurrent.futures.as_completed(futures):
            r=futures[f]
            try:ok=f.result()
            except Exception as e:ok=False;print('SELECTION_EXCEPTION',r,e,flush=True)
            print('SELECT',r['dataset'],r['seed_id'],r['method'],'PASS' if ok else 'FAIL',flush=True)
            if not ok:failures.append(r)
    # Single evaluator process per case; never parallelize evaluation.
    for r in rows:
        if not valid(r,'select'):continue
        try:ok=run(r,'evaluate')
        except Exception as e:ok=False;print('EVAL_EXCEPTION',r,e,flush=True)
        print('EVAL',r['dataset'],r['seed_id'],r['method'],'PASS' if ok else 'FAIL',flush=True)
        if not ok:failures.append(r)
    return failures
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,choices=range(1,5),default=2)
    p.add_argument('--smoke-only',action='store_true')
    p.add_argument('--dataset')
    p.add_argument('--seed',default='S01')
    p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    a=p.parse_args()
    DATA=json.loads(MANIFEST.read_text())['rows']
    if a.dataset:
        chosen=[r for r in DATA if r['dataset']==a.dataset and r['seed_id']==a.seed and r['method'] in a.methods]
        if len(chosen)!=len(set(a.methods)):
            p.error('chosen rows do not match the requested methods in the frozen manifest')
        failed=phase(chosen,a.workers)
        if failed or not all(valid(r,'select') and valid(r,'evaluate') for r in chosen):
            sys.exit(2)
        print(f'TARGETED_COMPLETE {len(chosen)}/{len(chosen)}',flush=True)
        return
    smoke=[r for r in DATA if r['seed_id']=='S01']
    assert len(smoke)==30 and len(DATA)==900
    failed=phase(smoke,a.workers)
    smoke_pass=not failed and all(valid(r,'select') and valid(r,'evaluate') for r in smoke)
    (HOME/'smoke_verdict.json').write_text(json.dumps({'passed':smoke_pass,'selection_complete':sum(valid(r,'select') for r in smoke),'evaluation_complete':sum(valid(r,'evaluate') for r in smoke),'failures':failed},indent=2))
    if not smoke_pass:
        print('SMOKE_GATE_FAILED: Full experiment NOT started',flush=True);sys.exit(2)
    print('SMOKE_GATE_PASSED 30/30',flush=True)
    if a.smoke_only:return
    remaining=[r for r in DATA if r['seed_id']!='S01']
    bad=phase(remaining,a.workers)
    complete=sum(valid(r,'select') and valid(r,'evaluate') for r in DATA)
    (HOME/'full_verdict.json').write_text(json.dumps({'complete':complete,'expected':900,'failures':bad},indent=2))
    print('FULL_COMPLETE',complete,'/900',flush=True)
    if bad or complete!=900:sys.exit(3)
if __name__=='__main__':main()
