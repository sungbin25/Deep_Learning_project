"""Train full DenseNet161 for complete epochs; supports epoch-boundary resume."""
import argparse
import csv
import json
import os
import random
import shutil
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import densenet161
from week2_pipeline import ROOT, OUT, CACHE, prepare, to_cwt


class CWTWindows(Dataset):
    def __init__(self, split='train'):
        self.x=np.load(CACHE/f'{split}_windows.npy',mmap_mode='r')
        self.y=np.load(CACHE/f'{split}_y.npy')
    def __len__(self): return len(self.y)
    def __getitem__(self,index):
        return torch.from_numpy(to_cwt(self.x[index])),torch.tensor(int(self.y[index]),dtype=torch.long)


def create_model():
    # Matches the lecture's unmodified DenseNet161 backbone and 5-way classifier.
    model=densenet161(weights=None, memory_efficient=True)
    model.classifier=nn.Linear(model.classifier.in_features,5)
    return model


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def train(epochs=5,batch_size=4,threads=2,resume=False,require_cuda=False):
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required. Install the CUDA PyTorch build and update the NVIDIA driver; CPU fallback is disabled.')
    if epochs < 1 or batch_size < 2: raise ValueError('epochs >= 1 and batch_size >= 2 required')
    if shutil.disk_usage(ROOT).free < 1024**3:
        raise RuntimeError('At least 1 GiB free disk space is required for checkpoint replacement; 3 GiB recommended before installation/training.')
    torch.set_num_threads(threads)
    metadata=prepare()
    seed_all(42)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=create_model().to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.001,foreach=False)
    criterion=nn.CrossEntropyLoss()
    dataset=CWTWindows('train')
    checkpoint_dir=OUT/'checkpoints'; checkpoint_dir.mkdir(parents=True,exist_ok=True)
    checkpoint=checkpoint_dir/'last.pt'
    history=[]; start_epoch=0
    config=dict(model='DenseNet161',weights=None,classes=5,learning_rate=.001,
                batch_size=batch_size,seed=42,threads=threads,device=str(device),
                dataset_id=metadata['dataset_id'],torch=str(torch.__version__),
                samples_per_epoch=len(dataset),dropout=0.0,precision='float32',
                backbone='torchvision standard stem and dense blocks',
                parameters=sum(p.numel() for p in model.parameters()),memory_efficient=True)
    if checkpoint.exists():
        if not resume: raise FileExistsError('Checkpoint exists. Use --resume to extend this run.')
        saved=torch.load(checkpoint,map_location=device,weights_only=True)
        for key in ['dataset_id','batch_size','seed','model']:
            if saved['config'][key] != config[key]: raise ValueError(f'Resume configuration mismatch: {key}')
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        history=saved['history']; start_epoch=saved['epoch']
    elif resume:
        raise FileNotFoundError('No checkpoint available for --resume')
    (OUT/'training_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(json.dumps(config,indent=2),flush=True)
    for epoch in range(start_epoch,epochs):
        # Epoch-specific RNG makes resume at an epoch boundary repeatable.
        seed_all(42+epoch)
        loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,num_workers=0,
                          generator=torch.Generator().manual_seed(42+epoch),drop_last=False)
        model.train(); loss_total=0.; correct=0; seen=0
        started=time.perf_counter()
        for step,(xb,yb) in enumerate(loader,1):
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits=model(xb); loss=criterion(logits,yb)
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            loss.backward(); opt.step()
            n=len(yb); loss_total+=loss.item()*n; seen+=n
            correct+=(logits.detach().argmax(1)==yb).sum().item()
            if step==1 or step%10==0 or step==len(loader):
                elapsed=time.perf_counter()-started
                print(f'Epoch {epoch+1}/{epochs} | batch {step}/{len(loader)} | mean loss {loss_total/seen:.6f} | elapsed {elapsed:.1f}s',flush=True)
                (OUT/'training_progress.json').write_text(json.dumps(dict(epoch=epoch+1,target_epochs=epochs,
                    batch=step,batches=len(loader),samples=seen,mean_loss=loss_total/seen,elapsed_seconds=elapsed),indent=2),encoding='utf-8')
        assert seen==len(dataset)
        record=dict(epoch=epoch+1,train_loss=loss_total/seen,train_accuracy=correct/seen,
                    samples=seen,batches=len(loader),seconds=time.perf_counter()-started)
        history.append(record)
        payload=dict(epoch=epoch+1,model=model.state_dict(),optimizer=opt.state_dict(),history=history,config=config)
        temp=checkpoint.with_suffix('.tmp')
        torch.save(payload,temp); os.replace(temp,checkpoint)
        with (OUT/'training_log.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(record)); writer.writeheader(); writer.writerows(history)
        (OUT/'training_history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        print('EPOCH COMPLETE:',json.dumps(record),flush=True)
    return history


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=5,help='Total target epochs, including any resumed epochs')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--threads',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--require-cuda',action='store_true')
    a=p.parse_args(); train(a.epochs,a.batch_size,a.threads,a.resume,a.require_cuda)
