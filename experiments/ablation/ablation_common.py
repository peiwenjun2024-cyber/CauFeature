"""Five-dataset extension of the manuscript's frozen ablation protocols."""
import os
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_k]='1'
os.environ.update(TF_ENABLE_ONEDNN_OPTS='0',TF_DETERMINISTIC_OPS='1',TF_CPP_MIN_LOG_LEVEL='2')
from pathlib import Path
import json,hashlib,time
ROOT=Path('/root/autodl-tmp/fast30_final')
CODE=ROOT/'code_ablation_five_20261028'
OUT=ROOT/'results/ablation_five_20261028'
BASE=CODE/'frozen_runtime'
DS=('BankFull','OnlineShoppers','StudentsAcademic','Optdigits','SoftwareQuality')
MECH=('full_control','wo_contribution_guided_paths','wo_target_local_pc','wo_reliability','wo_semantic_perturbation')
POLICIES=('original','all_residual','all_pairs')
def load(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);assert p.resolve().is_relative_to(OUT.resolve())
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_name(p.name+f'.{os.getpid()}.tmp');q.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False));q.replace(p)
def case(ds,sid):return OUT/'cases'/ds/sid
def manifest(ds,sid):return load(case(ds,sid)/'manifest.json')
def selection_path(ds,sid,kind):return case(ds,sid)/'selection'/(kind+'.json')
def evaluation_path(ds,sid,kind):return case(ds,sid)/'evaluation'/(kind+'.json')
def main_record(ds,sid):return load(manifest(ds,sid)['record'])
def p1(ds,sid):return load(manifest(ds,sid)['p1'])
def base_result(ds,sid,kind,features):
    m=manifest(ds,sid)
    return dict(protocol='ablation-five-frozen-20261028',dataset=ds,seed_id=sid,kind=kind,split_hash=m['split_hash'],features=features,dimension=len(features),source_manifest_sha256=sha(case(ds,sid)/'manifest.json'))
def tf_setup():
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    devices=tf.config.list_physical_devices('GPU')
    if not devices:raise RuntimeError('RTX GPU unavailable; refusing implicit CPU training')
    tf.config.set_logical_device_configuration(devices[0],[tf.config.LogicalDeviceConfiguration(memory_limit=4608)])
    return tf
