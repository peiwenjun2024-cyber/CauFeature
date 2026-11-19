"""Durable dependency scheduler: 14 CPU slots, reserved shallow capacity."""
from ablation_common import *
import sys,subprocess,fcntl,collections,signal,shutil
def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    lock=(OUT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    seeds=[f'S{i:02d}' for i in range(1,31)];jobs=[];running={};failed={};completed={};started=time.time()
    def add(key,pool,args,dest,deps=(),cost=1):jobs.append(dict(key=key,pool=pool,args=args,dest=str(dest),deps=list(deps),cost=cost))
    # Seed-major scheduling prevents omitting a dataset regime from an interrupted run.
    for sid in seeds:
      for ds in DS:
        p=case(ds,sid);tag=f'{ds}/{sid}'
        shallow_deps=['SoftwareQuality/S01/deep/full_control'] if ds=='SoftwareQuality' and sid!='S01' else []
        add(tag+'/shallow','cpu',['shallow',ds,sid],p/'shallow.pkl',shallow_deps,cost=2 if ds=='SoftwareQuality' else 1)
        add(tag+'/generic','gpu',['generic',ds,sid],p/'generic_p1.json')
        for v in MECH:
            dep=[tag+'/shallow']
            if v!='full_control':dep.append(tag+'/deep/full_control')
            if v=='wo_semantic_perturbation':dep.append(tag+'/generic')
            if ds=='SoftwareQuality' and sid=='S01' and v=='wo_target_local_pc':
                add(tag+'/deep/'+v,'cpu',['deepcheck',ds,sid,v],p/'p2_vector_equivalence'/(v+'.json'),dep)
            else:add(tag+'/deep/'+v,'cpu',['deep',ds,sid,v],p/'p2'/(v+'.json'),dep)
            add(tag+'/recover/'+v,'gpu',['recover',ds,sid,'mechanism_'+v],selection_path(ds,sid,'mechanism_'+v),[tag+'/deep/'+v])
        for v in POLICIES:
            dep=[] if v=='original' else [tag+'/policy/original']
            add(tag+'/policy/'+v,'gpu',['policy',ds,sid,v],selection_path(ds,sid,'p3_'+v),dep)
        kinds=['phase_original','phase_p1','phase_p1p2','phase_full','mechanism_wo_recovery']+['mechanism_'+v for v in MECH]+['p3_'+v for v in POLICIES]
        for kind in kinds:
            if kind.startswith('phase_') or kind=='mechanism_wo_recovery':deps=[]
            elif kind.startswith('mechanism_'):deps=[tag+'/recover/'+kind.removeprefix('mechanism_')]
            else:deps=[tag+'/policy/'+kind.removeprefix('p3_')]
            add(tag+'/eval/'+kind,'gpu',['evaluate',ds,sid,kind],evaluation_path(ds,sid,kind),deps)
    cancellation_path=OUT/'user_cancelled_software_global.json'
    if cancellation_path.exists():
        cancelled=set(load(cancellation_path)['excluded_job_keys'])
        jobs=[j for j in jobs if j['key'] not in cancelled]
    save(OUT/'execution_plan.json',dict(datasets=DS,seeds=30,total_jobs=len(jobs),cpu_slots=14,gpu_slots=4,gpu_memory_mib_per_worker=4608,p2_shallow_workers={'SoftwareQuality':2,'others':1},software_shallow_cpu_cap=8,software_global_deep_cap=10,mechanism_variants=MECH,p3_policies=POLICIES,job_definitions=jobs))
    statuspath=OUT/'status.json'
    def valid_dest(j):
        p=Path(j['dest'])
        if not p.exists():return False
        if p.suffix=='.json':
            x=load(p)
            if '/eval/' in j['key']:assert all(k in x for k in ('features','accuracy','f1_macro','split_hash'))
            if j['args'][0]=='deepcheck':
                ds,sid,variant=j['args'][1:];assert x['dataset']==ds and x['seed_id']==sid and x['split_hash']==manifest(ds,sid)['split_hash']
                assert all(v['mode']=='global' for v in x['deep'].values())
                canonical=case(ds,sid)/'p2'/(variant+'.json')
                if not canonical.exists():
                    x.update(promoted_vector_reference=True,vector_reference_source=str(p),vector_reference_sha256=sha(p),deep_kernel='verified_batched_global');save(canonical,x)
        return True
    for j in jobs:
        if valid_dest(j):completed[j['key']]='existing'
    user_failed_path=OUT/'user_failed_software_unguided.json'
    if user_failed_path.exists():
        failure_policy=load(user_failed_path)
        for key in failure_policy['failed_job_keys']:
            if key not in completed:failed[key]=dict(reason='user_requested_stop',status='failed',audit=str(user_failed_path))
        for key in failure_policy['blocked_job_keys']:
            if key not in completed:failed[key]=dict(reason='dependency_failed_user_requested_stop',status='blocked',audit=str(user_failed_path))
    pending=[j for j in jobs if j['key'] not in completed and j['key'] not in failed]
    # Reattach surviving workers after a scheduler restart, without rerunning scientific jobs.
    class Adopted:
        def __init__(self,pid,command):self.pid=pid;self.command=command
        def poll(self):
            try:return None if (Path('/proc')/str(self.pid)/'cmdline').read_bytes()==self.command else -999
            except FileNotFoundError:return -999
    for procdir in Path('/proc').iterdir():
        if not procdir.name.isdigit():continue
        try:command=(procdir/'cmdline').read_bytes();args=command.decode().split('\0')
        except (OSError,UnicodeError):continue
        for j in list(pending):
            script=str(CODE/('ablation_cpu.py' if j['pool']=='cpu' else 'ablation_gpu.py'))
            if script not in args:continue
            index=args.index(script)
            if args[index+1:]==j['args']+['']:
                if j['args'][0]=='shallow' and j['args'][1]=='SoftwareQuality':
                    progress=case(j['args'][1],j['args'][2])/'shallow_progress.json'
                    if progress.exists():j['cost']=load(progress).get('workers',j['cost'])
                log=(OUT/'logs'/(j['key'].replace('/','__')+'.log')).open('a');running[j['key']]=(j,Adopted(int(procdir.name),command),log);pending.remove(j);print('ADOPTED',j['key'],procdir.name,flush=True);break
    last_status=0
    try:
      while pending or running:
        for key,(j,proc,log) in list(running.items()):
            rc=proc.poll()
            if rc is None:continue
            log.close();running.pop(key)
            if valid_dest(j):completed[key]='new' if rc==0 else 'artifact_saved_exit_warning_'+str(rc)
            else:failed[key]=dict(exit_code=rc,log=str(OUT/'logs'/(key.replace('/','__')+'.log')))
            print('FINISHED',key,'exit',rc,'artifact',key in completed,flush=True)
        available={'cpu':14-sum(j['cost'] for j,p,l in running.values() if j['pool']=='cpu'),'gpu':4-sum(j['cost'] for j,p,l in running.values() if j['pool']=='gpu')}
        remaining=[]
        for j in pending:
            key=j['key']
            if any(d in failed for d in j['deps']):failed[key]=dict(blocked_by=[d for d in j['deps'] if d in failed]);continue
            if valid_dest(j):completed[key]='existing';continue
            if not all(d in completed for d in j['deps']) or available[j['pool']]<j['cost']:remaining.append(j);continue
            if j['args'][0] in ('deep','deepcheck') and j['args'][1]=='SoftwareQuality' and j['args'][-1]=='wo_target_local_pc':
                global_jobs=sum(z['args'][0] in ('deep','deepcheck') and z['args'][1]=='SoftwareQuality' and z['args'][-1]=='wo_target_local_pc' for z,p,l in running.values())
                if global_jobs>=10:remaining.append(j);continue

            if j['args'][0]=='shallow' and j['args'][1]=='SoftwareQuality':
                swcost=sum(z['cost'] for z,p,l in running.values() if z['args'][0]=='shallow' and z['args'][1]=='SoftwareQuality')
                if swcost+j['cost']>8:remaining.append(j);continue
            # Reuse exact ordered feature sets only within the same dataset/seed/split.
            if '/eval/' in key:
                ds,sid=key.split('/')[:2];kind=j['args'][-1];sel=load(selection_path(ds,sid,kind));features=sel['features'];reused=False
                inflight=False
                for z,proc,log in running.values():
                    if z['args'][0]=='evaluate' and z['args'][1:3]==[ds,sid] and load(selection_path(ds,sid,z['args'][-1]))['features']==features:inflight=True;break
                if inflight:remaining.append(j);continue
                for fp in sorted((case(ds,sid)/'evaluation').glob('*.json')):
                    x=load(fp)
                    if x['features']==features and x['split_hash']==sel['split_hash']:
                        x.update(kind=kind,reuse='exact_ordered_features_same_seed',source_evaluation=str(fp),source_evaluation_sha256=sha(fp));save(j['dest'],x);completed[key]='deduplicated';reused=True;break
                if reused:continue
            script=CODE/('ablation_cpu.py' if j['pool']=='cpu' else 'ablation_gpu.py')
            logpath=OUT/'logs'/(key.replace('/','__')+'.log');log=logpath.open('a',buffering=1)
            env=os.environ.copy()
            if j['pool']=='cpu':env['CUDA_VISIBLE_DEVICES']='-1'
            proc=subprocess.Popen([sys.executable,'-u',str(script),*j['args']],cwd=str(ROOT),env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            running[key]=(j,proc,log);available[j['pool']]-=j['cost'];print('START',key,proc.pid,flush=True)
        pending=remaining
        if time.time()-last_status>10 or not pending and not running:
            save(statuspath,dict(started_unix=started,updated_unix=time.time(),elapsed_seconds=time.time()-started,total=len(jobs),completed=len(completed),completion_methods=dict(collections.Counter(completed.values())),running=[dict(key=k,pid=p.pid,pool=j['pool'],cost=j['cost']) for k,(j,p,l) in running.items()],pending=len(pending),failed=failed,complete=not pending and not running and not failed,free_disk_bytes=shutil.disk_usage(ROOT).free))
            last_status=time.time()
        if pending and not running and not any(all(d in completed for d in j['deps']) for j in pending):raise RuntimeError('Unresolvable dependency state')
        if running:time.sleep(2)
      result=subprocess.run([sys.executable,str(CODE/'ablation_report.py')],cwd=str(ROOT))
      if result.returncode:raise RuntimeError('Final aggregation/verification failed')
    finally:
      # Child jobs are isolated and left intact on scheduler interruption; resume must wait for them to finish.
      for j,p,log in running.values():log.close()
    if failed:raise SystemExit(f'{len(failed)} jobs failed or blocked; inspect status.json')
if __name__=='__main__':main()
