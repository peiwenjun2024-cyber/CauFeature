"""Isolated E1 depth-3 SCM generator & length-stratified path auditor.

Default: print a plan only. --self-check: CPU-only in-memory validation.
--execute: explicitly write synthetic data; NEVER runs training.
--evaluate: read existing truths + audit sidecars, write no files.
Frozen E1 generator/runtime remain untouched.
"""
import argparse
import collections
import json
from pathlib import Path
import numpy as np

BASE = Path("/root/autodl-tmp/fast30_final")
HERE = BASE / "RQ4_EXPERIMENT_HUB_20261014"
DEST = HERE / "deep3_extension_20261015"
KINDS = ("s4_two_chains_deep3",)
SEEDS = range(1, 31)
DIMS = (20, 100, 241)

def generate(n=5000, d=20, seed=1):
    """Two independent true 3-edge paths X003-X002-X001-Y and X004-X005-X006-Y."""
    if d < 10 or n < 200:
        raise ValueError("require d>=10 and n>=200")
    rng = np.random.default_rng(202610150000 + 1007 * seed + d * 53 + n)
    x = rng.normal(size=(n, d))
    # Two linear-Gaussian chains, preserving uncorrelated exogenous noise.
    # Chain A: X003 -> X002 -> X001 -> Y
    x[:, 1] = 0.85 * x[:, 2] + np.sqrt(1-0.85**2) * rng.normal(size=n)
    x[:, 0] = 0.85 * x[:, 1] + np.sqrt(1-0.85**2) * rng.normal(size=n)
    # Chain B: X004 -> X005 -> X006 -> Y
    x[:, 4] = 0.85 * x[:, 3] + np.sqrt(1-0.85**2) * rng.normal(size=n)
    x[:, 5] = 0.85 * x[:, 4] + np.sqrt(1-0.85**2) * rng.normal(size=n)
    logit = 1.5*x[:, 0] - 1.25*x[:, 5]
    y = rng.binomial(1, 1/(1+np.exp(-np.clip(logit, -30, 30))))
    names = [f"X{i+1:03d}" for i in range(d)]
    edges = [
        ["X001", "target"], ["X002", "X001"], ["X003", "X002"],
        ["X006", "target"], ["X005", "X006"], ["X004", "X005"]
    ]
    truth = dict(kind=KINDS[0], seed=seed, n=n, d=d,
                 direct_target_parents=["X001", "X006"],
                 indirect_ancestors=["X002", "X003", "X004", "X005"],
                 correlated_proxies=[],
                 true_interaction_pairs=[],
                 true_skeleton_edges=sorted([sorted(e) for e in edges]),
                 noise_features=names[6:],
                 label_generation="Bernoulli(sigmoid(1.5*X001 - 1.25*X006))",
                 truth_interpretation="undirected target paths; no direction recovery claim",
                 max_path_edges=3)
    return names, x, y, truth

def paths_to_target(edges, max_edges=3):
    """Enumerate simple node sequences ending at target; order matters."""
    graph = collections.defaultdict(set)
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)
    result = set()
    for src in graph:
        if src == "target":
            continue
        stack = [(src, (src,))]
        while stack:
            node, path = stack.pop()
            if node == "target":
                if 1 <= len(path)-1 <= max_edges:
                    result.add(path)
                continue
            if len(path)-1 >= max_edges:
                continue
            for nxt in graph[node]:
                if nxt not in path:
                    stack.append((nxt, path+(nxt,)))
    return result

def metrics(truth_paths, predicted_paths, depth):
    actual = {p for p in truth_paths if len(p)-1 == depth}
    observed = {p for p in predicted_paths if len(p)-1 == depth}
    hit = len(actual & observed)
    precision = hit/len(observed) if observed else (1.0 if not actual else 0.0)
    recall = hit/len(actual) if actual else None
    f1 = 2*precision*recall/(precision+recall) if recall is not None and precision+recall else (0.0 if recall is not None else None)
    return dict(depth=depth, true_paths=len(actual), predicted_paths=len(observed),
                true_positive=hit, false_positive=len(observed)-hit,
                false_negative=len(actual)-hit, precision=precision, recall=recall, f1=f1,
                complete_coverage=actual.issubset(observed))

def validate():
    for d in DIMS:
        _, x, y, truth = generate(300, d, 1)
        assert x.shape == (300, d) and y.shape == (300,)
        p = paths_to_target(truth["true_skeleton_edges"])
        assert {i:sum(len(v)-1==i for v in p) for i in (1,2,3)} == {1:2,2:2,3:2}
        assert set(truth["direct_target_parents"]) == {"X001","X006"}
        assert len(truth["noise_features"]) == d-6
    print("SELF_CHECK_PASS: d=20/100/241, two real paths at each depth 1/2/3; no files written.")

def evaluate(truth_dir, audit_dir):
    by_depth = collections.defaultdict(list)
    cases = 0
    for truthfile in sorted(truth_dir.glob("*.truth.json")):
        truth = json.loads(truthfile.read_text())
        stem = truthfile.name.removesuffix(".truth.json")
        sid = f"S{int(truth['seed']):02d}"
        auditfile = audit_dir/stem/sid/"ddg_skeleton_audit.json"
        if not auditfile.exists():
            continue
        audit = json.loads(auditfile.read_text())
        reference = paths_to_target(truth["true_skeleton_edges"], 3)
        cases += 1
        for stage,key in (("raw","all_target_paths"), ("screened","screened_target_paths")):
            output = set(tuple(p) for p in audit[key])
            for depth in (1,2,3):
                by_depth[(stage,depth)].append(metrics(reference,output,depth))
    print("MATCHED_CASES",cases)
    for (stage,depth),records in sorted(by_depth.items()):
        import statistics
        def avg(k):
            vals = [z[k] for z in records if z[k] is not None]
            return round(statistics.mean(vals), 5) if vals else None
        print(json.dumps(dict(stage=stage, depth=depth, cases=len(records),
                              precision_macro=avg("precision"), recall_macro=avg("recall"),
                              f1_macro=avg("f1"), complete_count=sum(r["complete_coverage"] for r in records),
                              mean_true_paths=avg("true_paths"),mean_predicted_paths=avg("predicted_paths"))))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check",action="store_true")
    parser.add_argument("--execute",action="store_true",help="Generate datasets only; requires pandas, never trains")
    parser.add_argument("--evaluate",action="store_true",help="Read-only evaluation of existing outputs")
    parser.add_argument("--truth-dir",type=Path,default=DEST/"datasets")
    parser.add_argument("--audit-dir",type=Path,default=DEST/"runs")
    args=parser.parse_args()
    if args.self_check:
        validate()
    if args.execute:
        import pandas as pd
        dst = DEST/"datasets"
        dst.mkdir(parents=True,exist_ok=True)
        for d in DIMS:
            for seed in SEEDS:
                stem=f"{KINDS[0]}_n5000_d{d}_s{seed:02d}"
                csv=dst/(stem+".csv");tf=dst/(stem+".truth.json")
                if csv.exists() or tf.exists():
                    raise FileExistsError(f"Never overwrite frozen artifacts: {stem}")
                names,x,y,truth=generate(5000,d,seed)
                frame=pd.DataFrame(x,columns=names);frame["target"]=y
                frame.to_csv(csv,index=False)
                tf.write_text(json.dumps(truth,ensure_ascii=False,indent=2)+"\n")
        print("DATA_ONLY_COMPLETE",len(DIMS)*len(SEEDS),"NO_TRAINING")
    if args.evaluate:
        evaluate(args.truth_dir,args.audit_dir)
    if not (args.execute or args.evaluate or args.self_check):
        print("PLAN_ONLY: 3 dimensions x 30 seeds = 90 cases; two real paths at depths 1/2/3.")
        print("Will not generate data, train models, or use GPU without --execute.")

if __name__=="__main__":
    main()
