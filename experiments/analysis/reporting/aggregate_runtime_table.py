#!/usr/bin/env python3
from pathlib import Path
import argparse,csv,json,statistics

def timing(record):
    t=record.get('timing_seconds',{})
    selector=t.get('p1_selector_model_sec_reused',t.get('selector_model_sec'))
    p1c=t.get('p1_contribution_sec_reused',t.get('p1_contribution_sec'))
    p1r=t.get('p1_reliability_sec_reused',t.get('p1_reliability_sec'))
    p2=t.get('p2'); p3=t.get('p3')
    vals=[selector,p1c,p1r,p2,p3]
    if any(v is None for v in vals):
        raise ValueError('record is missing one or more required timing_seconds fields')
    return {'Selector':float(selector),'P1':float(p1c)+float(p1r),'P2':float(p2),'P3':float(p3)}

def main():
    ap=argparse.ArgumentParser(description='Aggregate existing CauFeature stage timings; does not run experiments.')
    ap.add_argument('--results-root',required=True,help='Root containing DATASET/Sxx/record.json')
    ap.add_argument('--datasets',nargs='+',required=True)
    ap.add_argument('--expected-seeds',type=int,default=30)
    ap.add_argument('--output',required=True)
    a=ap.parse_args(); root=Path(a.results_root); rows=[]
    for ds in a.datasets:
        vals=[]; missing=[]
        for i in range(1,a.expected_seeds+1):
            sid=f'S{i:02d}'; path=root/ds/sid/'record.json'
            if not path.exists(): missing.append(sid); continue
            rec=json.loads(path.read_text())
            if rec.get('status')!='complete': missing.append(sid); continue
            try: vals.append(timing(rec))
            except Exception as e: raise RuntimeError(f'{ds}/{sid}: {e}')
        if missing or len(vals)!=a.expected_seeds:
            raise RuntimeError(f'{ds}: expected {a.expected_seeds} complete timing records; got {len(vals)}; missing={missing}')
        means={k:statistics.mean(v[k] for v in vals) for k in ['Selector','P1','P2','P3']}
        sds={k:statistics.stdev(v[k] for v in vals) if len(vals)>1 else 0.0 for k in means}
        overall=[sum(v[k] for k in means) for v in vals]
        means['Overall']=statistics.mean(overall)
        sds['Overall']=statistics.stdev(overall) if len(overall)>1 else 0.0
        row={'Dataset':ds,'N':len(vals)}
        for k in ['Selector','P1','P2','P3','Overall']:
            row[k+'_mean_s']=means[k]; row[k+'_sd_s']=sds[k]
        rows.append(row)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print('WROTE',out)
    for r in rows: print(r['Dataset'],*(f"{r[k+'_mean_s']:.6f}" for k in ['Selector','P1','P2','P3','Overall']))

if __name__=='__main__': main()