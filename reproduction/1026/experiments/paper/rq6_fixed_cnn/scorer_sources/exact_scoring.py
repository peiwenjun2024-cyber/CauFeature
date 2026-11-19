"""Mirror production's distinct original-reference/ bootstrap numeric precision."""
import numpy as np
from scipy.stats import entropy

def exact_scores(predict,shared,meta,batch_size=8192):
    names=meta['feature_names'];refs=shared['references']
    scores=np.zeros((5,len(names)),dtype=float);queries=0
    for bi,ref in enumerate(refs):
        raw_base=np.asarray(predict(ref[None,:]))[0]
        # feature_perturbation._js_divergence preserves probability dtype;
        # p1_reliability._js_divergence explicitly converts to float64.
        base=np.asarray(raw_base,dtype=float) if bi else raw_base
        base=np.clip(base,1e-10,1.)
        jobs=[];indices=[]
        def flush():
            nonlocal queries
            if not jobs:return
            xx=np.repeat(ref[None,:],len(jobs),axis=0).astype(np.float32)
            jj=np.asarray(indices);xx[np.arange(len(jobs)),jj]=np.asarray(jobs,dtype=np.float32)
            raw=np.asarray(predict(xx))
            q=np.asarray(raw,dtype=float) if bi else raw
            q=np.clip(q,1e-10,1.)
            mid=.5*(q+base)
            js=.5*(entropy(q,mid,axis=1)+entropy(base[None,:],mid,axis=1))
            np.maximum.at(scores[bi],jj,js)
            queries+=len(jobs);jobs.clear();indices.clear()
        for j,name in enumerate(names):
            for val in meta['perturbation_values'][name]:
                jobs.append(float(val));indices.append(j)
                if len(jobs)>=batch_size:flush()
        flush()
    return scores,queries
