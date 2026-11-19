from ablation_common import *
import csv,shutil,statistics
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=ROOT/'code_flat_profilefix_20260929'
    BASE.mkdir(parents=True,exist_ok=True)
    files=['p2_parallel_pc.py','stage2_runtime.py','feature_perturbation.py','high_contribution.py','model_train_v6.py','fixed_split_evaluator.py','shared_globals.py','p1_reliability.py','p3_recovery.py','feature_semantic_metadata.json']
    hashes={}
    for name in files:
        src=source/name;dst=BASE/name
        if dst.exists():assert sha(src)==sha(dst),(src,dst)
        else:shutil.copy2(src,dst)
        hashes[str(src)]=sha(src)
    for name in ['online_shoppers_mechanisms_paired.py','stage1_runtime_mech.py','stage3_runtime_mech.py']:
        src=ROOT/'code_mechanism_ablation_20261013'/name;dst=CODE/('historical_'+name)
        if not dst.exists():shutil.copy2(src,dst)
        assert sha(src)==sha(dst);hashes[str(src)]=sha(src)
    old={ (x['dataset'],x['seed_id']):x for x in csv.DictReader((ROOT/'results/alpha001_full30_20261009/official_main_results_15x30_manifest.csv').open())}
    sources=[]
    for ds in DS:
      for n in range(1,31):
        sid=f'S{n:02d}';d=case(ds,sid)
        rp=ROOT/'results/alpha001_full30_20261009'/ds/sid/'record.json' if ds=='Optdigits' else Path(old[ds,sid]['record_json'])
        rec=load(rp);pp=rp.parent/'p1_cache.json' if ds=='Optdigits' else Path(rec['source_policy']['p1_cache']);p=load(pp)
        ref=rp.parent if ds=='Optdigits' else ROOT/'results/scaled_evaluator_full_vs_selected_20261010/cases'/ds/sid
        full,selected=ref/'full_scaled.json',ref/'selected_scaled.json'
        f,s=load(full),load(selected)
        assert p['split_hash']==rec['split_hash']==f['split_hash']==s['split_hash']==sha(p['split_file'])
        assert rec['selection']['final_features']==s['features']
        assert sha(p['selector_model'])==p['selector_model_sha256']
        data=ROOT/'pilot_optdigits_30_20261011/dataset/Optdigits.csv' if ds=='Optdigits' else ROOT/'code_revised_20260910/dataset'/(ds+'.csv')
        assert sha(data)==p['dataset_sha256']
        assert sha(p['semantic_metadata_file'])==p['semantic_metadata_sha256']
        m=dict(dataset=ds,seed_id=sid,split_hash=p['split_hash'],record=str(rp),record_sha256=sha(rp),p1=str(pp),p1_sha256=sha(pp),data=str(data),data_sha256=sha(data),full_eval=str(full),selected_eval=str(selected),full_eval_sha256=sha(full),selected_eval_sha256=sha(selected),selector_model_sha256=p['selector_model_sha256'],split_file=p['split_file'],metadata_file=p['semantic_metadata_file'],metadata_sha256=p['semantic_metadata_sha256'])
        dest=d/'manifest.json'
        if dest.exists():assert load(dest)==m
        else:save(dest,m)
        for kind,features,reference in [('phase_original',f['features'],full),('phase_p1',p['high_contrib_features'],None),('phase_p1p2',rec['selection']['p2_features'],None),('phase_full',s['features'],selected),('mechanism_wo_recovery',rec['selection']['p2_features'],None)]:
            z=base_result(ds,sid,kind,features);z.update(p1_sha256=sha(pp),record_sha256=sha(rp));save(selection_path(ds,sid,kind),z)
            # Existing StudentsAcademic stage records exactly define Table 6.
            archived={'phase_original':'Original_Features','phase_p1':'P1-only','phase_p1p2':'P1_P2','phase_full':'CauFeature'}.get(kind)
            ap=ROOT/'results/stage_ablation_scaled_20261015/StudentsAcademic'/sid/((archived or '')+'.json')
            if ds=='StudentsAcademic' and archived and ap.exists() and load(ap)['features']==features:reference=ap
            if reference is None:
                reference=next((q for q in (selected,full) if load(q)['features']==features),None)
            if reference:
                ev=load(reference);assert ev['split_hash']==m['split_hash'];ev.update(kind=kind,source_evaluation=str(reference),source_evaluation_sha256=sha(reference),reuse='verified_existing_scaled',protocol=z['protocol']);save(evaluation_path(ds,sid,kind),ev)
        sources.append(m)
    # Preserve full P1 scorer study; never train the already-complete 450 models again.
    v=ROOT/'results/p1_model_sensitivity_scaled_20261024/verification.json';assert load(v)['passed'] and load(v)['model_records']==450
    c=ROOT/'results/p1_model_sensitivity_scaled_20261024/reports/comparisons_summary.csv'
    save(OUT/'source_audit.json',dict(cases=150,datasets=list(DS),seeds=30,sources=sources,code_sha256=hashes,p1_model_verification=load(v),p1_comparison_summary=str(c),p1_comparison_sha256=sha(c),historical_outputs_immutable=True))
    print('PREPARED 150 source cases; source hashes and standardized evaluator membership verified',flush=True)
if __name__=='__main__':main()
