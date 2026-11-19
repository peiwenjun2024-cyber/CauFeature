"""E1 S4 Deep-3 isolated integration. Default --check is CPU-only and read-only.
--prepare generates data/splits/manifest and clones a frozen runtime into an
isolated subtree. It never starts training or GPU jobs.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path
BASE=Path("/root/autodl-tmp/fast30_final")
H=BASE/"RQ4_EXPERIMENT_HUB_20261014"
CFG=H/"configs"/"e1_deep3_extension_20261015.py"
DEST=H/"deep3_extension_20261015"
SOURCE=BASE/"code_synthetic_recovery_validation_20261014/rq4_v3_e1_highdim_frozen_runtime_20261014"
ORIG=H/"highdim"
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda:f.read(1048576), b""):h.update(chunk)
    return h.hexdigest()
def module():
    s=importlib.util.spec_from_file_location("e1_deep3_ext",CFG)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def check():
    assert CFG.is_file() and SOURCE.is_dir() and (ORIG/"freeze.json").is_file()
    old=json.loads((ORIG/"freeze.json").read_text())
    for k,v in old["runtime_hashes"].items():
        assert sha(SOURCE/k)==v, "frozen source drift: "+k
    module().validate()
    print("PREP_CHECK_PASS: source runtime hashes verified, generator checked, NO GPU/NO TRAINING")
def prepare():
    check()
    from sklearn.model_selection import train_test_split
    import pandas as pd
    ext=module()
    if DEST.exists():
        raise FileExistsError("Fail closed: destination exists; won't overwrite: "+str(DEST))
    # Preflight disk and destination, generate only after explicit --prepare.
    if shutil.disk_usage(BASE).free < 3*1024**3:
        raise RuntimeError("Less than 3 GiB free: refuse data generation.")
    datasets=DEST/"datasets"
    datasets.mkdir(parents=True)
    rows=[]
    for dim in ext.DIMS:
        for seed in ext.SEEDS:
            stem=f"{ext.KINDS[0]}_n5000_d{dim}_s{seed:02d}"
            names,x,y,truth=ext.generate(n=5000,d=dim,seed=seed)
            csv=datasets/(stem+".csv")
            split=datasets/(stem+".split.csv")
            tp=datasets/(stem+".truth.json")
            frame=pd.DataFrame(x,columns=names);frame["target"]=y
            frame.to_csv(csv,index=False)
            indices=list(range(5000))
            train,test=train_test_split(indices,test_size=.2,random_state=seed)
            train,valid=train_test_split(train,test_size=.25,random_state=seed)
            split_df=pd.DataFrame([{"row_index":int(idx),"partition":kind}
                for kind,arr in (("train",train),("validation",valid),("test",test)) for idx in arr])
            split_df.to_csv(split,index=False)
            truth.update(dataset=csv.name,dataset_sha256=sha(csv),split_sha256=sha(split))
            tp.write_text(json.dumps(truth,indent=2)+"\n")
            rows.append(dict(dataset=csv.name,separator=",",n_samples=5000,
                n_features=dim,sha256=sha(csv),split_seed_id=f"S{seed:02d}",
                split_seed=seed,split_file=str(split),split_hash=sha(split),
                synthetic_profile="RQ4V3_E1_DEEP3",mechanism=ext.KINDS[0],
                generation_seed=seed))
    manifest=DEST/"synthetic_manifest.jsonl"
    manifest.write_text("".join(json.dumps(z)+"\n" for z in rows))
    runtime=DEST/"runtime"
    shutil.copytree(SOURCE,runtime,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    # The runner uses module constants; path-adjust ONLY cloned runtime, not original.
    for fname in ("synthetic_runner.py","stage1_runtime.py","stage2_runtime.py","stage3_runtime.py","stage1.py","stage2.py","stage3.py"):
        p=runtime/fname
        if p.exists():
            content=p.read_text()
            content=content.replace("RQ4_EXPERIMENT_HUB_20261014/highdim","RQ4_EXPERIMENT_HUB_20261014/deep3_extension_20261015")
            p.write_text(content)
    # All source runtime files remain unchanged; provide audit trail of cloned hashes.
    hashes={f.name:sha(f) for f in runtime.iterdir() if f.is_file()}
    freeze=dict(description="E1 S4 independent 3-hop path extension",
        source_freeze=old_freeze_id(),source_runtime=str(SOURCE),
        generator_sha256=sha(CFG),preparer_sha256=sha(Path(__file__)),
        cases=len(rows),manifest_sha256=sha(manifest),runtime_hashes=hashes,
        split_protocol="sklearn train_test_split 60/20/20, random_state=seed",
        execution_status="DATA_PREPARED_NO_TRAINING")
    (DEST/"freeze.json").write_text(json.dumps(freeze,indent=2)+"\n")
    print("PREPARED_ONLY",len(rows),"manifest",manifest,"runtime",runtime)
    print("No training process launched. Run future execution only after GPU available and separate runtime smoke verification.")
def old_freeze_id():
    return sha(ORIG/"freeze.json")
if __name__=="__main__":
    p=argparse.ArgumentParser()
    g=p.add_mutually_exclusive_group()
    g.add_argument("--check",action="store_true")
    g.add_argument("--prepare",action="store_true")
    a=p.parse_args()
    if a.prepare: prepare()
    else: check()
