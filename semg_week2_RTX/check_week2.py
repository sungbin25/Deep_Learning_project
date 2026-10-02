"""Report readiness without starting model training or evaluation."""
import importlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parent

def check():
    dependencies={}
    for package,module in [('numpy','numpy'),('scipy','scipy'),('PyWavelets','pywt'),('torch','torch'),('torchvision','torchvision')]:
        try:
            version=importlib.metadata.version(package)
            importlib.import_module(module)
            dependencies[package]={'ready':True,'version':version}
        except Exception as error:
            dependencies[package]={'ready':False,'reason':str(error)}
    free=shutil.disk_usage(ROOT).free
    cache=ROOT/'data/processed_week2'
    data_ready=all((cache/p).exists() for p in ['metadata.json','train_windows.npy','train_y.npy','test_windows.npy','test_y.npy'])
    exe=Path(sys.executable)
    python_path=str(exe.relative_to(ROOT)) if exe.is_relative_to(ROOT) else exe.name
    result={'python':python_path,'free_disk_gib':round(free/1024**3,3),
            'dependencies':dependencies,'preprocessing_ready':data_ready,
            'ready_to_train':all(v['ready'] for v in dependencies.values()) and data_ready and free>=1024**3,
            'training_requested_now':False,
            'saved_checkpoint_exists':(ROOT/'output/week2/checkpoints/last.pt').exists(),
            'evaluation_result_exists':(ROOT/'output/week2/test_metrics.json').exists()}
    target=ROOT/'output/week2';target.mkdir(parents=True,exist_ok=True)
    (target/'readiness.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result

if __name__=='__main__':check()
