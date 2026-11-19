"""Paired RQ3 statistics from the standardized records underlying the native-rule table."""
import csv
import json
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, t, wilcoxon

OUT = Path(__file__).resolve().parent
PAPER = OUT.parents[1]
ROOT = PAPER.parent
DATASETS = ['BankFull', 'OnlineShoppers', 'StudentsAcademic', 'Optdigits', 'SoftwareQuality']
METHODS = ['VT', 'MI', 'CHI', 'RFE', 'RFC', 'RANDOM']
SUMMARY = {r['dataset']: r for r in csv.DictReader((PAPER/'revision_optdigits_main_20261012/figures/data/rq1_main_scaled_15.csv').open())}
rows, paired = [], []
for ds in DATASETS:
    cau = []
    for i in range(1, 31):
        seed = f'S{i:02d}'
        path = (ROOT/f'results/alpha001_full30_20261009/Optdigits/{seed}/selected_scaled.json' if ds == 'Optdigits' else ROOT/f'results/scaled_evaluator_full_vs_selected_20261010/cases/{ds}/{seed}/selected_scaled.json')
        rec = json.loads(path.read_text())
        assert rec['dataset'] == ds and rec['seed_id'] == seed
        cau.append(rec)
    for metric, key in [('accuracy', 'scaled_selected_acc_mean'), ('f1_macro', 'scaled_selected_f1_mean')]:
        assert abs(np.mean([r[metric] for r in cau])-float(SUMMARY[ds][key])) < 1e-12
    for method in METHODS:
        baseline = []
        for i, c in enumerate(cau, 1):
            seed = f'S{i:02d}'
            b = json.loads((ROOT/f'results/matched_baselines_5ds_s30_trainonly_20261011/{method}/{ds}/{seed}/scaled_eval.json').read_text())
            assert b['mode'] == 'scaled' and b['dataset'] == ds and b['seed_id'] == seed
            assert c['split_hash'] == b['split_hash'], (ds, method, seed)
            assert c['dimension'] == b['selected_dimension'] == b['budget_k']
            baseline.append(b)
            paired.append(dict(dataset=ds, baseline=method.replace('-Z',''), seed=seed, split_hash=c['split_hash'], cau_accuracy=c['accuracy'], baseline_accuracy=b['accuracy'], cau_f1=c['f1_macro'], baseline_f1=b['f1_macro']))
        for metric in ['accuracy', 'f1_macro']:
            diff = np.array([(c[metric]-b[metric])*100 for c,b in zip(cau,baseline)])
            # Round to 12 decimal places in pp to avoid floating-point artificial rank ties.
            ranked_diff = np.round(diff, 12)
            nz = ranked_diff[ranked_diff != 0]
            ranks = rankdata(np.abs(nz)) if len(nz) else np.array([])
            effect = float(np.sum(ranks*np.sign(nz))/np.sum(ranks)) if len(nz) else 0.0
            p = float(wilcoxon(ranked_diff, zero_method='wilcox', alternative='two-sided', method='approx').pvalue) if len(nz) else 1.0
            mean = float(diff.mean()); half = float(t.ppf(.975,29)*diff.std(ddof=1)/np.sqrt(30))
            rows.append(dict(dataset=ds, baseline=method.replace('-Z',''), metric=metric, n=30, nonzero_pairs=len(nz), mean_difference_pp=mean, ci_low_pp=mean-half, ci_high_pp=mean+half, p_raw=p, rank_biserial=effect))
# Holm family: all 5 datasets x 6 baselines x 2 metrics.
order = np.argsort([r['p_raw'] for r in rows], kind='stable'); running = 0.0
for k, idx in enumerate(order):
    running = max(running, min(1.0, (len(rows)-k)*rows[idx]['p_raw']))
    rows[idx]['p_holm'] = running
for name, data in [('paired_seed_records.csv',paired),('paired_statistics.csv',rows)]:
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)

def pformat(p):
    value = '<0.001' if p < .001 else f'{p:.4f}'
    return r'$\mathbf{'+value+'}$' if p < .05 else '$'+value+'$'
lines = [r'\begin{table*}[!tbp]',r'\centering\scriptsize',r'\setlength{\tabcolsep}{3pt}',r'\renewcommand{\arraystretch}{1.10}',r'\caption{Paired Comparisons of CauFeature with Matched-Budget Baselines}',r'\label{tab:rq3_paired_statistics}',r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}llcrrcrr@{}}',r'\toprule',r'\multirow{2}{*}{\textbf{Dataset}} & \multirow{2}{*}{\textbf{Baseline}} & \multicolumn{3}{c}{\textbf{Accuracy}} & \multicolumn{3}{c}{\textbf{F-measure}} \\',r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',r' & & $\Delta$ [95\% CI] (pp) & $p_{\mathrm{Holm}}$ & $r_{\mathrm{rb}}$ & $\Delta$ [95\% CI] (pp) & $p_{\mathrm{Holm}}$ & $r_{\mathrm{rb}}$ \\',r'\midrule']
for ds in DATASETS:
    for j, method in enumerate(METHODS):
        label=method.replace('-Z',''); cells=[ds if j==0 else '',label]
        for metric in ['accuracy','f1_macro']:
            r=next(r for r in rows if r['dataset']==ds and r['baseline']==label and r['metric']==metric)
            cells += [rf"${r['mean_difference_pp']:+.2f}\;[{r['ci_low_pp']:+.2f},{r['ci_high_pp']:+.2f}]$",pformat(r['p_holm']),f"${r['rank_biserial']:+.3f}$"]
        lines.append(' & '.join(cells)+r' \\')
    if ds != DATASETS[-1]: lines.append(r'\midrule')
lines += [r'\bottomrule',r'\end{tabular*}',r'\par\smallskip',r'{\raggedright\scriptsize\noindent\textit{Note:} $\Delta$ is CauFeature minus the named baseline over 30 paired seeds. CIs are unadjusted paired 95\% $t$ intervals for mean differences. Two-sided Wilcoxon signed-rank tests use nonzero differences, average ranks for ties, and the normal approximation without continuity correction. Holm correction covers all 60 dataset--baseline--metric comparisons. Bold $p$ values indicate $p_{\mathrm{Holm}}<0.05$; positive $r_{\mathrm{rb}}$ favors CauFeature. Significance follows the adjusted test rather than CI exclusion of zero.\par}',r'\end{table*}']
(OUT/'appendix_table.tex').write_text('\n'.join(lines)+'\n')
print('Verified 900 baseline records and 150 CauFeature records; all split hashes match; CauFeature means match manuscript source.')
for ds in DATASETS:
    for metric in ['accuracy','f1_macro']:
        q=[r for r in rows if r['dataset']==ds and r['metric']==metric]
        print(ds,metric,'wins',[r['baseline'] for r in q if r['p_holm']<.05 and r['mean_difference_pp']>0],'losses',[r['baseline'] for r in q if r['p_holm']<.05 and r['mean_difference_pp']<0])
