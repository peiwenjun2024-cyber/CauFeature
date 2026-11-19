"""Real P1 reliability ablation: recompute raw-contribution task ranking from frozen P1 contributions."""
import json
from pathlib import Path
ROOT = Path('/root/autodl-tmp/fast30_final')

def run_stage1_mechanism(dataset, seed_id, use_reliability=True):
    record = json.loads((ROOT/'results/alpha001_full30_20261009'/dataset/seed_id/'record.json').read_text())
    src = Path(record['source_policy']['p1_cache'])
    p1 = json.loads(src.read_text())
    if p1['split_hash'] != record['split_hash']:
        raise RuntimeError('P1/Full split hash mismatch')
    p1['p1_source_cache'] = str(src)
    if use_reliability:
        p1['mechanism_score_source'] = 'frozen_bootstrap_reliability_scores'
        return p1
    names = p1['feature_names']
    raw = p1['contribution_scores']
    if any(k not in raw for k in names):
        raise RuntimeError('Missing raw P1 contributions')
    # Preserve the original contributions, but rederive P2 ranking without
    # bootstrap reliability corrections. P2 computes the 75th percentile anew.
    p1['reliable_scores'] = {k: float(raw[k]) for k in names}
    p1['mechanism_score_source'] = 'raw_unweighted_P1_contributions'
    p1['mechanism_ablation'] = 'wo_reliability'
    return p1
