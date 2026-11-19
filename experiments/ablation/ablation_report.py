"""Validate and aggregate complete records; incomplete summaries remain explicitly partial."""
from ablation_common import *
import csv,statistics,collections,math,sys
def writecsv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:return
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    import fcntl
    report_lock=(OUT/'report.lock').open('w')
    try:fcntl.flock(report_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another report is running; no duplicate aggregation',flush=True);return
    audit=load(OUT/'source_audit.json');errors=[];rows=[];selrows=[]
    for m in audit['sources']:
        ds,sid=m['dataset'],m['seed_id'];rec=load(m['record']);p=load(m['p1'])
        for key,h in [('record','record_sha256'),('p1','p1_sha256'),('data','data_sha256'),('metadata_file','metadata_sha256'),('full_eval','full_eval_sha256'),('selected_eval','selected_eval_sha256')]:
            if sha(m[key])!=m[h]:errors.append('source changed '+m[key])
        if sha(p['selector_model'])!=m['selector_model_sha256']:errors.append('source changed '+p['selector_model'])
        for fp in (case(ds,sid)/'selection').glob('*.json'):
            x=load(fp)
            if x['split_hash']!=m['split_hash'] or len(x['features'])!=len(set(x['features'])):errors.append('selection invalid '+str(fp))
            if not set(x['features']).issubset(set(p['feature_names'])):errors.append('unknown feature '+str(fp))
            ev=evaluation_path(ds,sid,x['kind']);e=load(ev) if ev.exists() else None
            if e:
                if e['features']!=x['features'] or e['split_hash']!=m['split_hash']:errors.append('evaluation mismatch '+str(ev))
                rows.append(dict(dataset=ds,seed_id=sid,kind=x['kind'],dimension=len(x['features']),accuracy=e['accuracy'],f1_macro=e['f1_macro'],reuse=e.get('reuse'),selection_path=str(fp),evaluation_path=str(ev),selection_sha256=sha(fp),evaluation_sha256=sha(ev)))
            if x['kind'].startswith('p3_'):
                selrows.append(dict(dataset=ds,seed_id=sid,policy=x['kind'][3:],candidate_pairs=x['candidate_pairs'],evaluated_pairs=x['evaluated_pairs'],recovered=len(x['recovered_features']),dimension=x['dimension'],forward_rows=x['forward_rows'],pair_loop_seconds=x['pair_loop_seconds'],default_parity=x.get('default_p3_parity')))
        # Existing OnlineShoppers is an independent compatibility oracle for the historical mechanism controls.
        for v in MECH:
            new=selection_path(ds,sid,'mechanism_'+v)
            old=ROOT/'results/mechanism_ablation_onlineshoppers_paired_20261013'/sid/v/'mechanism_record.json'
            if ds=='OnlineShoppers' and new.exists():
                x,y=load(new),load(old)
                if x['features']!=y['final_features']:errors.append('historical mechanism membership mismatch '+str(new))
                if x['p2']['task_path_count']!=y['p2']['task_path_count']:errors.append('historical path count mismatch '+str(new))
                for l,stat in x['p2']['deep'].items():
                    for k in ('ci_tests','deletions'):
                        if stat[k]!=y['p2']['deep'][l][k]:errors.append('historical deep count mismatch '+str(new))
    groups=collections.defaultdict(list)
    for x in rows:groups[x['dataset'],x['kind']].append(x)
    summary=[]
    for (ds,kind),xs in sorted(groups.items()):
        z=dict(dataset=ds,kind=kind,n=len(xs),complete=len(xs)==30)
        for k in ('dimension','accuracy','f1_macro'):
            z[k+'_mean']=statistics.mean(x[k] for x in xs);z[k+'_sd']=statistics.stdev(x[k] for x in xs) if len(xs)>1 else None
        summary.append(z)
    writecsv(OUT/'reports/per_seed_results.csv',rows);writecsv(OUT/'reports/dataset_summary.csv',summary);writecsv(OUT/'reports/p3_search_per_seed.csv',selrows)
    p2rows=[]
    for fp in (OUT/'cases').glob('*/*/p2/*.json'):
        x=load(fp);folder=case(x['dataset'],x['seed_id'])/'deep_kernel_audit'/x['kind'];stats=[load(p) for p in folder.glob('*.json')];logical=sum(v['ci_tests'] for v in x['deep'].values())
        if 'deep_kernel_counters' in x:stats=[x['deep_kernel_counters']]
        extra=sum(v['physical_probes']-v['logical_probes'] for v in stats)
        missing_extra=x['dataset']=='SoftwareQuality' and (x['seed_id'] in ('S02','S03') or x.get('promoted_vector_reference')) and x['kind']=='wo_target_local_pc' and 'deep_kernel_counters' not in x and not any(v['batched_edges'] for v in stats)
        p2rows.append(dict(dataset=x['dataset'],seed_id=x['seed_id'],variant=x['kind'],dimension=x['dimension'],paths=x['task_path_count'],deep_ci_tests_logical=logical,deep_ci_probes_computed=None if missing_extra else logical+extra,deep_batched_lookahead_extra=None if missing_extra else extra,deep_kernel='batched_global_physical_audit_unavailable' if missing_extra else ('batched_global' if any(v['batched_edges'] for v in stats) else 'historical_scalar'),p2_cpu_seconds=x['p2_cpu_seconds'],accounted_wall_seconds=x['p2_wall_seconds_accounted'],shallow_reused=x['reused_common_shallow']))
    writecsv(OUT/'reports/p2_mechanism_per_seed.csv',p2rows)
    comparisons=[]
    from scipy.stats import wilcoxon,t
    lookup={(x['dataset'],x['seed_id'],x['kind']):x for x in rows}
    for ds in DS:
      for kind in ['phase_p1','phase_p1p2','mechanism_wo_recovery']+['mechanism_'+v for v in MECH if v!='full_control']+['p3_all_residual','p3_all_pairs']:
        diffs=[]
        for i in range(1,31):
            sid=f'S{i:02d}';a=lookup.get((ds,sid,'phase_full'));b=lookup.get((ds,sid,kind))
            if a and b:diffs.append(100*(a['f1_macro']-b['f1_macro']))
        if len(diffs)!=30:continue
        mean=statistics.mean(diffs);se=statistics.stdev(diffs)/math.sqrt(30);ci=float(t.ppf(.975,29))*se
        pv=1. if all(v==0 for v in diffs) else float(wilcoxon(diffs,alternative='two-sided').pvalue)
        comparisons.append(dict(dataset=ds,control=kind,n=30,full_minus_control_f1_pp=mean,ci_low=mean-ci,ci_high=mean+ci,wilcoxon_p=pv))
    # Define one family across all planned dataset/control F1 comparisons; only report adjusted p after completion.
    kinds=13;expected_evals=150*kinds
    complete=len(rows)==expected_evals and len(p2rows)==150*len(MECH) and len(selrows)==450 and not errors
    if complete:
        order=sorted(range(len(comparisons)),key=lambda i:comparisons[i]['wilcoxon_p']);prev=0
        for rank,i in enumerate(order):
            prev=max(prev,min(1.,comparisons[i]['wilcoxon_p']*(len(order)-rank)));comparisons[i]['holm_p']=prev
    writecsv(OUT/'reports/paired_f1_comparisons.csv',comparisons)
    import shutil
    src=ROOT/'results/p1_model_sensitivity_scaled_20261024/reports'
    for name in ('comparisons_summary.csv','comparisons_per_seed.csv','model_diagnostics_summary.csv'):
        shutil.copy2(src/name,OUT/'reports'/('p1_'+name))
    # Emit publication tables only when every required cell in that family has all 30 seeds.
    tabledir=OUT/'reports/tables';tabledir.mkdir(exist_ok=True)
    index={(x['dataset'],x['kind']):x for x in summary}
    def pm(x,key,scale=1):return f"${float(x[key+'_mean'])*scale:.2f} \\pm {float(x[key+'_sd'])*scale:.2f}$"
    labels={'phase_original':'Original Features','phase_p1':'P1-only','phase_p1p2':'P1+P2','phase_full':'CauFeature','mechanism_full_control':'Full CauFeature','mechanism_wo_reliability':'w/o Reliability','mechanism_wo_semantic_perturbation':'w/o Semantic Perturbation','mechanism_wo_contribution_guided_paths':'w/o Contribution-guided Paths','mechanism_wo_target_local_pc':'w/o Target-local PC','mechanism_wo_recovery':'No recovery','p3_original':'Original candidate scope','p3_all_residual':'All core--residual pairs','p3_all_pairs':'Including residual--residual pairs'}
    for family,ks in [('phase_ablation',['phase_original','phase_p1','phase_p1p2','phase_full']),('p1_internal',['mechanism_full_control','mechanism_wo_reliability','mechanism_wo_semantic_perturbation']),('p2_internal',['mechanism_full_control','mechanism_wo_contribution_guided_paths','mechanism_wo_target_local_pc']),('p3_internal',['mechanism_wo_recovery','p3_original','p3_all_residual','p3_all_pairs'])]:
        if not all((ds,k) in index and index[ds,k]['n']==30 for ds in DS for k in ks):continue
        tex=['% Complete five-dataset, 30-seed family only.','\\begin{tabular}{llrrr}','\\hline',r'Dataset & Configuration & Dim. & Acc. (\%) & F-measure (\%) \\',r'\hline']
        for ds in DS:
            for k in ks:
                x=index[ds,k];tex.append(' & '.join([ds,labels[k],pm(x,'dimension'),pm(x,'accuracy',100),pm(x,'f1_macro',100)])+r' \\')
        tex.extend([r'\hline',r'\end{tabular}']);(tabledir/(family+'.tex')).write_text('\n'.join(tex))
    p1comp=list(csv.DictReader((src/'comparisons_summary.csv').open()))
    tex=[r'\begin{tabular}{llrr}',r'\hline',r'Dataset & Comparison & $\rho_{\rm Con}$ & $J(F_H)$ \\',r'\hline']
    for ds in DS:
        for x in p1comp:
            if x['dataset']==ds and x['first_model']=='CNN':
                tex.append(' & '.join([ds,'CNN--'+x['second_model'],pm(x,'con_spearman'),pm(x,'fh_jaccard')])+r' \\')
    tex.extend([r'\hline',r'\end{tabular}']);(tabledir/'p1_models_all_five.tex').write_text('\n'.join(tex))
    parity=sum(load(p).get('default_p2_parity',False) for p in (OUT/'cases').glob('*/*/p2/full_control.json'))
    verification=dict(complete=complete,passed=complete and not errors,errors=errors,evaluations=len(rows),expected_evaluations=expected_evals,p2_records=len(p2rows),p3_policy_records=len(selrows),default_p2_parity_cases=parity,p1_existing_model_records=450,p1_verification_source=audit['p1_model_verification'],source_files_unchanged=not any('source changed' in x for x in errors),timing_policy='Shared shallow CPU cost accounted for every variant; concurrent wall times are not standalone speedup evidence. Batched global CI retains historical logical first-exit counts; physical lookahead probes are separate. Scalar and batched reference timings must not be pooled as homogeneous wall-clock speedups.',synthetic_truth_experiments='Preserved unchanged; real datasets have no generating interaction truth labels.')
    cancellation_path=OUT/'user_cancelled_software_global.json'
    if cancellation_path.exists():
        cancellation=load(cancellation_path)
        active_expected=expected_evals-cancellation['excluded_evaluation_records']
        active_p2_expected=150*len(MECH)-cancellation['excluded_p2_records']
        active_complete=len(rows)==active_expected and len(p2rows)==active_p2_expected and len(selrows)==450 and not errors
        verification.update(original_plan_complete=complete,requested_scope_complete=active_complete,requested_scope_passed=active_complete and not errors,requested_scope_expected_evaluations=active_expected,requested_scope_expected_p2_records=active_p2_expected,cancelled_experiments=[dict(dataset='SoftwareQuality',variant='wo_target_local_pc',status='cancelled_by_user',evaluation_records=30)])
    user_failed_path=OUT/'user_failed_software_unguided.json'
    if user_failed_path.exists():
        failure_policy=load(user_failed_path)
        scheduler_state=load(OUT/'status.json')
        verification.update(failed_experiments=[dict(dataset='SoftwareQuality',seed_id=sid,variant='wo_contribution_guided_paths',status='failed',reason='user_requested_stop',evaluation_status='blocked') for sid in failure_policy['seeds']],available_scope_expected_evaluations=1918,available_scope_expected_p2_records=718,available_results_complete=len(rows)==1918 and len(p2rows)==718 and len(selrows)==450 and not errors,all_runnable_jobs_finished=not scheduler_state.get('running') and scheduler_state.get('pending')==0,requested_scope_complete=False,requested_scope_passed=False)
    save(OUT/'verification.json',verification)
    lines=['# Five-dataset ablations — '+('COMPLETE' if complete else 'PARTIAL'),'',str(verification),'','| Dataset | Configuration | n | Dim. | Acc. (%) | F1 (%) |','|---|---|---:|---:|---:|---:|']
    for x in summary:lines.append(f"|{x['dataset']}|{x['kind']}|{x['n']}|{x['dimension_mean']:.2f}|{x['accuracy_mean']*100:.2f}|{x['f1_macro_mean']*100:.2f}|")
    (OUT/'reports/report.md').write_text('\n'.join(lines))
    print(json.dumps(verification,indent=2),flush=True)
    if errors:raise SystemExit('Validation errors; results are not publication-ready')
if __name__=='__main__':main()
