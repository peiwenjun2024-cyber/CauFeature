#!/usr/bin/env python3
import argparse, json, os, signal, subprocess, sys, time, traceback
from pathlib import Path

ROOT=Path('/root/autodl-tmp/fast30_final')
BASE=ROOT/'RQ4_EXPERIMENT_HUB_20261014/deep3_extension_20261015'
MAN=BASE/'synthetic_manifest.jsonl'
DATA=BASE/'datasets'
OUT=ROOT/'RQ4_EXPERIMENT_HUB_20261014/deep3_extension_20261015/runs'

# Must be set before wrappers spawn runtime children.
os.environ['FAST30_MANIFEST']=str(MAN)
os.environ['FAST30_DATASET_DIR']=str(DATA)
os.environ['FAST30_OUTPUT_ROOT']=str(OUT)
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['NUMEXPR_NUM_THREADS']='1'

from stage1 import run as stage1_run
from stage3 import run as stage3_run

def cleanup_pg(pgid):
    try: os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError,PermissionError): return
    time.sleep(.2)
    try: os.killpg(pgid,0)
    except ProcessLookupError: return
    try: os.killpg(pgid,signal.SIGKILL)
    except (ProcessLookupError,PermissionError): pass

def complete(rd):
    p=rd/'record.json'
    if not p.exists(): return False
    try: return str(json.load(open(p)).get('status','')).lower()=='complete'
    except: return False

def run_one(rec, workers=16):
    ds=rec['dataset']; sid=rec['split_seed_id']
    rd=OUT/Path(ds).stem/sid
    rd.mkdir(parents=True,exist_ok=True)
    if complete(rd):
        print('SKIP_COMPLETE',ds,sid,flush=True); return 'skip'
    print('START_SYN',ds,sid,'profile',rec['synthetic_profile'],'mech',rec['mechanism'],time.strftime('%F %T'),flush=True)
    try:
        stage1_run(ds,sid,str(MAN),str(OUT))
        env=os.environ.copy()
        log=rd/'stage2.log'
        with log.open('a') as fh:
            p=subprocess.Popen(
                [sys.executable,'stage2_runtime.py','--dataset',ds,'--split-seed-id',sid,'--workers',str(workers)],
                env=env,stdout=fh,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            try: rc=p.wait()
            finally: cleanup_pg(p.pid)
        if rc!=0:
            raise RuntimeError('stage2 failed\n'+'\n'.join(log.read_text(errors='replace').splitlines()[-80:]))
        stage3_run(ds,sid,str(MAN),str(OUT))
        (rd/'synthetic_status.json').write_text(json.dumps({'status':'complete','profile':rec['synthetic_profile'],'mechanism':rec['mechanism']}))
        print('COMPLETE_SYN',ds,sid,time.strftime('%F %T'),flush=True)
        return 'complete'
    except Exception as e:
        (rd/'synthetic_status.json').write_text(json.dumps({
            'status':'failed','profile':rec['synthetic_profile'],'mechanism':rec['mechanism'],
            'error':str(e),'traceback':traceback.format_exc()
        },indent=2))
        print('FAILED_SYN',ds,sid,repr(e),flush=True)
        return 'failed'

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--contains',default='')
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--workers',type=int,default=16)
    a=ap.parse_args()
    recs=[json.loads(x) for x in MAN.read_text().splitlines() if x.strip()]
    if a.contains: recs=[r for r in recs if a.contains in r['dataset']]
    if a.limit: recs=recs[:a.limit]
    stat={'complete':0,'skip':0,'failed':0}
    for r in recs: stat[run_one(r,a.workers)]+=1
    print('SYN_SUMMARY',json.dumps(stat),flush=True)

if __name__=='__main__':
    main()
