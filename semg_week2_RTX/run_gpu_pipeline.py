"""Run full GPU workflow with a persistent console log and fail-fast status."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'output/week2'


def run(epochs=5,batch_size=16,threads=4):
    OUT.mkdir(parents=True,exist_ok=True)
    checkpoint=OUT/'checkpoints/last.pt'
    train_args=['--epochs',str(epochs),'--batch-size',str(batch_size),'--threads',str(threads),'--require-cuda']
    if checkpoint.exists(): train_args.append('--resume')
    stages=[('gpu_check',['check_gpu.py','--require-cuda']),('prepare',['week2_pipeline.py']),
            ('data_checks',['test_week2.py']),('training',['train_week2.py',*train_args]),
            ('evaluation',['evaluate_week2.py','--batch-size',str(batch_size),'--threads',str(threads),'--require-cuda']),
            ('build_report',['build_week2_notebook.py']),('execute_report',['run_week2_notebook.py'])]
    def status(stage,state):
        (OUT/'gpu_run_status.json').write_text(json.dumps(dict(stage=stage,status=state,target_epochs=epochs,
            batch_size=batch_size,utc=datetime.now(timezone.utc).isoformat()),indent=2),encoding='utf-8')
    with (OUT/'gpu_console.log').open('a',encoding='utf-8',buffering=1) as log:
        for name,args in stages:
            status(name,'running')
            heading=f'\n=== {name} ===\n';print(heading,flush=True);log.write(heading)
            child=subprocess.Popen([sys.executable,'-u','-X','utf8',*args],cwd=ROOT,stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
            try:
                for line in child.stdout:
                    print(line,end='',flush=True);log.write(line)
                code=child.wait()
            except KeyboardInterrupt:
                child.terminate();child.wait();status(name,'interrupted');raise
            if code:
                status(name,'failed')
                raise RuntimeError(f'{name} failed (exit {code}). See output/week2/gpu_console.log. Later stages were not run.')
    status('complete','complete')
    print('DONE: output/week2/week2_report.html and week2_semg.ipynb',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--threads',type=int,default=4)
    a=p.parse_args();run(a.epochs,a.batch_size,a.threads)
