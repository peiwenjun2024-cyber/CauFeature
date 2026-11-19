"""Durable seeded dependency queue, 14 CPU slots, four 4608MiB GPU workers."""
from sens_common import *
import subprocess,fcntl,collections,shutil
def definitions():
 jobs=[]
 def add(ds,sid,stage,key,pool,dest,deps=(),cost=1):
  ident='/'.join([ds,sid,stage,key]);jobs.append(dict(key=ident,args=[stage,ds,sid]+([] if stage in ['p1batch','p3sweep','verifydefault'] else [key]),pool=pool,dest=str(dest),deps=list(deps),cost=cost))
 for sid in [f'S{i:02d}' for i in range(1,31)]:
  for ds in DS:
   tag=ds+'/'+sid
   add(ds,sid,'p1batch','batch','gpu',case(ds,sid)/'p1batch_done.json')
   add(ds,sid,'p3sweep','batch','gpu',case(ds,sid)/'p3sweep_done.json')
   for alpha in [.005,.05]:add(ds,sid,'shallow',str(alpha),'cpu',case(ds,sid)/'shallow'/('alpha_'+str(alpha)+'.pkl'),cost=2 if ds=='SoftwareQuality' else 1)
   for key in PIPEKEYS:
    conf=CONFIGS[key];alpha=float(conf['value']) if conf['group']=='alpha' else .01;dep=[]
    if conf['stage']=='P1':dep.append(tag+'/p1batch/batch')
    if alpha!=.01:dep.append(tag+'/shallow/'+str(alpha))
    add(ds,sid,'p2',key,'cpu',artifact(ds,sid,'p2',key),dep)
    add(ds,sid,'recover',key,'gpu',artifact(ds,sid,'selection',key),[tag+'/p2/'+key])
   for key in PIPEKEYS+P3KEYS:
    dep=tag+'/recover/'+key if key in PIPEKEYS else tag+'/p3sweep/batch'
    add(ds,sid,'evaluate',key,'gpu',artifact(ds,sid,'evaluation',key),[dep])
 return jobs
def main():
 lock=(OUT/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert load(OUT/'pilot_verification.json')['passed'],'Cannot run full campaign before pilot verification'
 jobs=definitions();save(OUT/'execution_plan.json',dict(protocol=PROTOCOL,total_jobs=len(jobs),cpu_slots=14,gpu_slots=4,configs=CONFIGS,groups=GROUPS,datasets=DS,seeds=30,job_definitions=jobs));finished={};failed={};running={};start=time.time()
 def valid(j):
  p=Path(j['dest'])
  if not p.exists():return False
  if p.suffix=='.json':load(p)
  return True
 for j in jobs:
  if valid(j):finished[j['key']]='existing'
 user_failure_path=OUT/'user_failed_path_length4.json'
 if user_failure_path.exists():
  policy=load(user_failure_path)
  for key in policy['failed_job_keys']:
   if key not in finished:failed[key]=dict(status='failed',reason='user_requested_stop',audit=str(user_failure_path))
  for key in policy['blocked_job_keys']:
   if key not in finished:failed[key]=dict(status='blocked',reason='dependency_failed_user_requested_stop',audit=str(user_failure_path))
 pending=[j for j in jobs if j['key'] not in finished and j['key'] not in failed]
 class Adopted:
  def __init__(self,pid,args):self.pid=pid;self.args=args
  def poll(self):
   try:return None if (Path('/proc')/str(self.pid)/'cmdline').read_bytes()==self.args else -999
   except FileNotFoundError:return -999
 for q in Path('/proc').iterdir():
  if not q.name.isdigit():continue
  try:raw=(q/'cmdline').read_bytes();args=raw.decode().split('\0')
  except (OSError,UnicodeError):continue
  script=str(CODE/'sens_worker.py')
  if script not in args:continue
  tail=args[args.index(script)+1:]
  for j in list(pending):
   if tail==j['args']+['']:
    running[j['key']]=(j,Adopted(int(q.name),raw),None);pending.remove(j);print('ADOPTED',j['key'],q.name,flush=True);break
 last=0
 def status():
  save(OUT/'status.json',dict(protocol=PROTOCOL,started_unix=start,updated_unix=time.time(),total=len(jobs),completed=len(finished),completion_methods=dict(collections.Counter(finished.values())),running=[dict(key=k,pid=p.pid,pool=j['pool'],cost=j['cost']) for k,(j,p,l) in running.items()],pending=len(pending),failed=failed,complete=not pending and not running and not failed,free_disk_bytes=shutil.disk_usage(ROOT).free))
 while pending or running:
  for key,(j,p,log) in list(running.items()):
   rc=p.poll()
   if rc is None:continue
   if log:log.close()
   del running[key]
   if valid(j):finished[key]='new' if rc==0 else 'saved_exit_warning_'+str(rc)
   else:failed[key]=dict(exit_code=rc,log=str(OUT/'logs'/(key.replace('/','__')+'.log')))
   print('FINISHED',key,'artifact',key in finished,'exit',rc,flush=True)
  gpu_count=sum(1 for z,p,l in running.values() if z['pool']=='gpu')
  available={pool:cap-sum(j['cost'] for j,p,l in running.values() if j['pool']==pool) for pool,cap in [('cpu',min(14,16-gpu_count)),('gpu',4)]};keep=[]
  for j in pending:
   key=j['key']
   if any(k in failed for k in j['deps']):failed[key]=dict(blocked_by=[k for k in j['deps'] if k in failed]);continue
   if valid(j):finished[key]='existing';continue
   if not all(k in finished for k in j['deps']) or available[j['pool']]<j['cost']:keep.append(j);continue
   if j['pool']=='gpu' and sum(z['cost'] for z,p,l in running.values() if z['pool']=='cpu')+sum(1 for z,p,l in running.values() if z['pool']=='gpu')>=16:keep.append(j);continue
   # Separate inference and training waves: prevent GPU memory/CPU oversubscription.
   if j['pool']=='gpu':
    is_eval=j['args'][0]=='evaluate';active_eval=[z['args'][0]=='evaluate' for z,p,l in running.values() if z['pool']=='gpu']
    if active_eval and any(v!=is_eval for v in active_eval):keep.append(j);continue
   if j['args'][0]=='evaluate':
    ds,sid,cfg=j['args'][1:];sel=load(artifact(ds,sid,'selection',cfg))
    if reuse_eval(ds,sid,cfg):finished[key]='exact_features_reuse';continue
    dup=any(z['args'][0]=='evaluate' and z['args'][1:3]==[ds,sid] and load(artifact(ds,sid,'selection',z['args'][3]))['features']==sel['features'] for z,p,l in running.values())
    if dup:keep.append(j);continue
   log=(OUT/'logs'/(key.replace('/','__')+'.log')).open('a');env=os.environ.copy()
   if j['pool']=='cpu':env['CUDA_VISIBLE_DEVICES']='-1'
   proc=subprocess.Popen([sys.executable,'-u',str(CODE/'sens_worker.py'),*j['args']],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);running[key]=(j,proc,log);available[j['pool']]-=j['cost'];print('START',key,proc.pid,flush=True)
  pending=keep
  if time.time()-last>10:status();last=time.time()
  if pending and not running:raise RuntimeError('No runnable dependency; inspect state')
  if running:time.sleep(2)
 status();subprocess.run([sys.executable,str(CODE/'sens_report.py')],cwd=ROOT,check=True)
 if failed:raise SystemExit('Campaign has failed jobs; no silent omission')
if __name__=='__main__':main()
