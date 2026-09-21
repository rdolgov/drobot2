import json,os,subprocess,time
from pathlib import Path
root=Path('C:/Users/roman/Documents/dev/drobot2')
out=root/'simulation/reviews/ankle-v27-overnight/20260920-continue'
cmd=['C:/isaacsim/python.bat',str(root/'simulation/isaac/rl/ankle_v26_feasible/run.py'),'train','--output',str(out/'train'),'--checkpoint',str(root/'simulation/reviews/ankle-v26-balanced/20260920-train-r2/checkpoints/model_549.pt'),'--iterations','2000','--num-envs','512','--seed','2721','--resume-optimizer','--learning-rate','0.00005']
if (out/'train').exists():
    raise RuntimeError('Training output already exists; refusing duplicate launch')
state={'status':'launching','launcher_pid':os.getpid(),'command':cmd,'started_unix':time.time()}
(out/'train-launch.json').write_text(json.dumps(state,indent=2))
with (out/'train.log').open('w',encoding='utf-8') as log:
    process=subprocess.Popen(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONIOENCODING':'utf-8'},creationflags=subprocess.CREATE_NO_WINDOW)
    state.update(status='running',process_pid=process.pid)
    (out/'train-launch.json').write_text(json.dumps(state,indent=2))
    code=process.wait()
state.update(status='exited',returncode=code,finished_unix=time.time())
(out/'train-launch.json').write_text(json.dumps(state,indent=2))
