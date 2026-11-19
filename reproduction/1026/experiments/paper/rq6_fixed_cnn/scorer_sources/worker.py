import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_DETERMINISTIC_OPS']='1'
os.environ['TF_GPU_ALLOCATOR']='cuda_malloc_async'
import time,sys,gc
import numpy as np
from common import *

TF=None
def init_gpu():
    global TF
    if TF is not None:return TF
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    gpu=tf.config.list_physical_devices('GPU')
    if not gpu:raise RuntimeError('GPU required for neural worker')
    tf.config.set_logical_device_configuration(gpu[0],[tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    tf.config.experimental.enable_op_determinism()
    TF=tf
    return tf

def high_fn(ds):
    from scipy.stats import gaussian_kde
    from scipy.ndimage import gaussian_filter1d
    return import_function(source_code(ds)/'high_contribution.py','select_high_contribution_features',
                           dict(np=np,gaussian_kde=gaussian_kde,gaussian_filter1d=gaussian_filter1d))

def validate_shared(dest):
    meta=load(dest/'shared.json')
    if sha(dest/'shared.npz')!=meta['shared_npz_sha256']:raise RuntimeError('Shared matrix hash mismatch')
    return meta,np.load(dest/'shared.npz',allow_pickle=False)

def base_record(meta, model):
    return dict(protocol=PROTOCOL,status='complete',dataset=meta['dataset'],seed_id=meta['seed_id'],
                seed=meta['seed'],model=model,source_p1_sha256=meta['source_p1_sha256'],
                shared_npz_sha256=meta['shared_npz_sha256'],shared_hashes=meta['hashes'],
                outer_test_rows_used=False,feature_names=meta['feature_names'],
                validation_role='internal development validation; used for neural early stopping',
                script_sha256=sha(__file__),common_code_sha256=sha(CODE/'common.py'))

def cnn_case(ds,sid,tf,meta,z):
    dest=case_dir(ds,sid);out=dest/'CNN.json'
    if out.exists():return
    t=time.perf_counter();p1=load(meta['source_p1'])
    if sha(meta['source_model'])!=meta['source_model_sha256']:raise RuntimeError('CNN model changed')
    model=tf.keras.models.load_model(meta['source_model'],compile=False)
    def predict(x):
        return model(np.asarray(x,dtype=np.float32).reshape(-1,len(meta['feature_names']),1),training=False).numpy()
    prob=np.concatenate([predict(z['x_val'][i:i+4096]) for i in range(0,len(z['x_val']),4096)])
    replay,queries=compute_scores(predict,z,meta,batch_size=4096,
                                 reference_context=lambda i:tf.device('/GPU:0' if i==0 else '/CPU:0'))
    cached=np.asarray([p1['reliability_records'][n]['baseline_scores'] for n in meta['feature_names']]).T
    error=np.abs(replay-cached)
    save_json(dest/'CNN_replay_check.json',dict(max_absolute_error_by_reference=error.max(axis=1).tolist(),
        score_matrix=replay.tolist(),frozen_score_matrix=cached.tolist()))
    # Production computed original scores on GPU and bootstrap scores on CPU.
    if not np.allclose(replay,cached,rtol=2e-4,atol=2e-6):
        raise RuntimeError(f'CNN frozen score replay mismatch: max_abs={error.max()}')
    ev=evidence(cached,meta['feature_names'],z['nonconstant'],high_fn(ds))
    if ev['high_contrib_features']!=p1['high_contrib_features']:raise RuntimeError('CNN threshold parity mismatch')
    if not np.allclose(list(ev['reliable_scores'].values()),[p1['reliable_scores'][n] for n in meta['feature_names']],rtol=1e-12,atol=1e-12):
        raise RuntimeError('CNN reliability parity mismatch')
    ev['contribution_scores']=p1['contribution_scores'];ev['reliable_scores']=p1['reliable_scores']
    result=base_record(meta,'CNN')
    result.update(ev,configuration='frozen original unscaled selector; no retraining',
                  frozen_model_sha256=meta['source_model_sha256'],
                  frozen_score_reused=True,validation_metrics=model_metrics(z['y_val'],prob),
                  replay_audit=dict(passed=True,atol=2e-6,rtol=2e-4,
                      max_absolute_error=float(error.max()),mean_absolute_error=float(error.mean()),
                      query_count=queries,score_matrix=replay.tolist()),
                  elapsed_sec=time.perf_counter()-t)
    np.savez(dest/'CNN_validation_predictions.npz',y=z['y_val'],probability=prob)
    save_json(out,result);del model;tf.keras.backend.clear_session();gc.collect()

def neural_case(ds,sid):
    tf=init_gpu();dest=case_dir(ds,sid);meta,z=validate_shared(dest)
    cnn_case(ds,sid,tf,meta,z)
    out=dest/'MLP.json'
    if out.exists():return dict(dataset=ds,seed_id=sid,kind='neural',cached=True)
    t=time.perf_counter();tf.keras.utils.set_random_seed(meta['seed'])
    d=len(meta['feature_names']);k=len(meta['target_classes'])
    model=tf.keras.Sequential([
        tf.keras.layers.Input((d,)),
        tf.keras.layers.Dense(128),tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(64),tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(k,activation='softmax')])
    model.compile(optimizer=tf.keras.optimizers.Adam(.001),
                  loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    bs=64 if meta['sample_num']<=10000 else 256 if meta['sample_num']<=100000 else 512
    options=tf.data.Options();options.threading.private_threadpool_size=1
    options.threading.max_intra_op_parallelism=1;options.experimental_deterministic=True
    train=tf.data.Dataset.from_tensor_slices((z['x_train'].astype(np.float32),z['y_train']))
    train=train.shuffle(len(z['y_train']),seed=meta['seed'],reshuffle_each_iteration=True).batch(bs).with_options(options).prefetch(2)
    val=tf.data.Dataset.from_tensor_slices((z['x_val'].astype(np.float32),z['y_val'])).batch(1024).with_options(options).prefetch(1)
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,min_delta=.001,restore_best_weights=True),
               tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=3,min_lr=1e-6)]
    tt=time.perf_counter();history=model.fit(train,validation_data=val,epochs=100,callbacks=callbacks,verbose=0)
    training=time.perf_counter()-tt
    model.save(dest/'MLP.keras')
    def predict(x):return model(np.asarray(x,dtype=np.float32),training=False).numpy()
    prob=np.concatenate([predict(z['x_val'][i:i+4096]) for i in range(0,len(z['x_val']),4096)])
    tt=time.perf_counter();scores,queries=compute_scores(predict,z,meta);scoring=time.perf_counter()-tt
    result=base_record(meta,'MLP')
    result.update(evidence(scores,meta['feature_names'],z['nonconstant'],high_fn(ds)),
                  configuration=dict(hidden_units=[128,64],batch_normalization='after hidden Dense, before ReLU',
                    dropout=.2,optimizer='Adam',initial_lr=.001,max_epochs=100,batch_size=bs,
                    early_stopping_patience=10,min_delta=.001,restore_best_weights=True,
                    reduce_lr_factor=.5,reduce_lr_patience=3,min_lr=1e-6,
                    input_scaling='none'),
                  model_sha256=sha(dest/'MLP.keras'),validation_metrics=model_metrics(z['y_val'],prob),
                  epochs=len(history.history['loss']),best_validation_loss=float(min(history.history['val_loss'])),
                  history={n:list(map(float,v)) for n,v in history.history.items()},
                  timing=dict(training_sec=training,scoring_sec=scoring,total_sec=time.perf_counter()-t),
                  query_count=queries)
    np.savez(dest/'MLP_validation_predictions.npz',y=z['y_val'],probability=prob)
    save_json(out,result);tf.keras.backend.clear_session();gc.collect()
    return dict(dataset=ds,seed_id=sid,kind='neural',seconds=time.perf_counter()-t)

def rf_case(ds,sid):
    from sklearn.ensemble import RandomForestClassifier
    import joblib
    dest=case_dir(ds,sid);out=dest/'RF.json'
    if out.exists():return dict(dataset=ds,seed_id=sid,kind='RF',cached=True)
    t=time.perf_counter();meta,z=validate_shared(dest)
    config=dict(n_estimators=500,max_features='sqrt',min_samples_leaf=1,max_depth=None,
                class_weight=None,random_state=meta['seed'],n_jobs=4)
    model=RandomForestClassifier(**config)
    tt=time.perf_counter();model.fit(z['x_train'],z['y_train']);training=time.perf_counter()-tt
    if not np.array_equal(model.classes_,np.arange(len(meta['target_classes']))):raise RuntimeError('RF class order mismatch')
    joblib.dump(model,dest/'RF.joblib',compress=1)
    def predict(x):return model.predict_proba(x)
    prob=predict(z['x_val'])
    tt=time.perf_counter();scores,queries=compute_scores(predict,z,meta,batch_size=16384);scoring=time.perf_counter()-tt
    result=base_record(meta,'RF')
    result.update(evidence(scores,meta['feature_names'],z['nonconstant'],high_fn(ds)),
                  configuration=config,model_sha256=sha(dest/'RF.joblib'),
                  validation_metrics=model_metrics(z['y_val'],prob),query_count=queries,
                  timing=dict(training_sec=training,scoring_sec=scoring,total_sec=time.perf_counter()-t))
    np.savez(dest/'RF_validation_predictions.npz',y=z['y_val'],probability=prob)
    save_json(out,result)
    return dict(dataset=ds,seed_id=sid,kind='RF',seconds=time.perf_counter()-t)

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('kind',choices=['neural','RF']);ap.add_argument('dataset');ap.add_argument('seed_id')
    a=ap.parse_args();print((neural_case if a.kind=='neural' else rf_case)(a.dataset,a.seed_id),flush=True)
