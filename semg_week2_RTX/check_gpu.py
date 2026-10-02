"""Verify CUDA installation and actual CUDA kernels without dataset training."""
import argparse
import json
from pathlib import Path
import shutil
import sys


def check(require_cuda=False):
    import torch,torchvision
    result=dict(python=sys.version,torch=str(torch.__version__),torchvision=str(torchvision.__version__),
                cuda_runtime=torch.version.cuda,cuda_available=torch.cuda.is_available(),
                disk_free_gib=round(shutil.disk_usage(Path(__file__).resolve().parent).free/1024**3,2),
                kernel_check_passed=False)
    if result['cuda_available']:
        result['gpu']=torch.cuda.get_device_name(0)
        result['vram_gib']=round(torch.cuda.get_device_properties(0).total_memory/1024**3,2)
        result['compute_capability']=list(torch.cuda.get_device_capability(0))
        try:
            x=torch.arange(16,dtype=torch.float32,device='cuda').reshape(4,4)
            y=x@x.T
            torch.cuda.synchronize()
            assert torch.isfinite(y).all().item()
            result['kernel_check_passed']=True
        except Exception as error:
            result['kernel_error']=str(error)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    root=Path(__file__).resolve().parent/'output/week2';root.mkdir(parents=True,exist_ok=True)
    (root/'gpu_environment.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    if require_cuda and not result['kernel_check_passed']:
        raise RuntimeError('CUDA check failed. Check the NVIDIA driver and CUDA-enabled PyTorch installation. CPU training was NOT started.')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--require-cuda',action='store_true');check(p.parse_args().require_cuda)
