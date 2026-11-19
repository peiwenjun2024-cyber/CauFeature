"""P3 recovery recomputed against the actual recomputed P2 subset."""
import sys
from pathlib import Path
ROOT=Path('/root/autodl-tmp/fast30_final')
BASE=ROOT/'code_flat_profilefix_20260929'
if str(BASE) not in sys.path:sys.path.insert(0,str(BASE))

def run_stage3_mechanism(p2, p1, dataset, seed_id, use_recovery=True):
    base=list(p2['p2_features'])
    if not use_recovery:
        return base, {'recovered_features':[],'candidate_pairs':0,'evaluated_pairs':0,'mode':'disabled'}
    import tensorflow as tf
    import shared_globals
    from model_train_v6 import load_and_preprocess_data
    from feature_perturbation import CausalFeaturePerturbation
    from p3_recovery import ResidualInteractionRecovery
    seed=int(p1['seed'])
    tf.keras.utils.set_random_seed(seed)
    try: tf.config.experimental.enable_op_determinism()
    except Exception: pass
    shared_globals.init()
    shared_globals.random_seed=seed
    load_and_preprocess_data(p1['selector_input'], None)
    model=tf.keras.models.load_model(str(p1['selector_model']),compile=False)
    shared_globals.model=model
    if list(shared_globals.feature_names)!=list(p1['feature_names']):
        raise RuntimeError('P3 feature order mismatch')
    scores=[float(p1['contribution_scores'][n]) for n in shared_globals.feature_names]
    shared_globals.con_list=scores
    shared_globals.high_contrib_features=list(p1['high_contrib_features'])
    shared_globals.p1_reliable_score=dict(p1['reliable_scores'])
    perturb=CausalFeaturePerturbation(model,shared_globals.data,dataset_name=dataset+'.csv' if not dataset.endswith('.csv') else dataset)
    perturb.perturb_values={n:list(p1['perturbation_values'][n]) for n in shared_globals.feature_names}
    perturb.con_list=scores
    shared_globals.perturbation_instance=perturb
    shared_globals.filtered_features=base
    info=ResidualInteractionRecovery(candidate_top_k=200).run()
    return list(shared_globals.filtered_features),info
