"""Evaluate a completed Week 2 checkpoint later; never invoked by notebook Run All."""
import argparse
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from train_week2 import create_model, CWTWindows
from week2_pipeline import ROOT, OUT, CACHE


def evaluate(batch_size=4,threads=2,require_cuda=False):
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required; CPU fallback is disabled.')
    torch.set_num_threads(threads)
    checkpoint=OUT/'checkpoints/last.pt'
    if not checkpoint.exists(): raise FileNotFoundError('Train a model first: no checkpoint exists.')
    metadata=json.loads((CACHE/'metadata.json').read_text())
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if saved['config']['dataset_id'] != metadata['dataset_id']:
        raise ValueError('Checkpoint and prepared dataset IDs do not match')
    epoch=saved['epoch']
    model=create_model(); model.load_state_dict(saved['model']); del saved
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=model.to(device); model.eval()
    dataset=CWTWindows('test')
    loader=DataLoader(dataset,batch_size=batch_size,shuffle=False,num_workers=0)
    probabilities=[]; start=time.perf_counter()
    with torch.inference_mode():
        for step,(xb,_) in enumerate(loader,1):
            probabilities.append(model(xb.to(device)).softmax(1).cpu().numpy())
            if step==1 or step%25==0 or step==len(loader):
                print(f'Evaluation batch {step}/{len(loader)}',flush=True)
    probs=np.concatenate(probabilities); pred=probs.argmax(1); truth=dataset.y
    index=pd.read_csv(OUT/'test_windows.csv')
    assert len(index)==len(truth)
    for k in range(5): index[f'prob_{chr(65+k)}']=probs[:,k]
    index['true_label']=truth; index['predicted_label']=pred
    index.to_csv(OUT/'test_predictions.csv',index=False)
    by_trial=index.groupby('trial_id')
    trial_probs=by_trial[[f'prob_{s}' for s in 'ABCDE']].mean()
    trial_truth=by_trial.true_label.first().loc[trial_probs.index].to_numpy()
    trial_pred=trial_probs.to_numpy().argmax(1)
    trial_output=trial_probs.copy()
    trial_output['true_label']=trial_truth
    trial_output['predicted_label']=trial_pred
    trial_output.to_csv(OUT/'test_trial_predictions.csv')
    metrics=dict(checkpoint_epoch=epoch,dataset_id=metadata['dataset_id'],test_windows=len(truth),
                 test_trial_files=len(trial_truth),window_accuracy=float(accuracy_score(truth,pred)),
                 window_macro_f1=float(f1_score(truth,pred,average='macro',zero_division=0)),
                 trial_mean_probability_accuracy=float(accuracy_score(trial_truth,trial_pred)),
                 inference_seconds=time.perf_counter()-start,device=str(device))
    (OUT/'test_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
    report=classification_report(truth,pred,labels=list(range(5)),target_names=list('ABCDE'),output_dict=True,zero_division=0)
    pd.DataFrame(report).T.to_csv(OUT/'classification_report.csv')
    matrix=confusion_matrix(truth,pred,labels=list(range(5)))
    pd.DataFrame(matrix,index=list('ABCDE'),columns=list('ABCDE')).to_csv(OUT/'confusion_matrix.csv')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained')
    im=ax.imshow(matrix,cmap='Blues')
    for i in range(5):
        for j in range(5): ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    ax.set(xticks=range(5),yticks=range(5),xticklabels=list('ABCDE'),yticklabels=list('ABCDE'),xlabel='Predicted subject',ylabel='True subject',title=f'Test windows / checkpoint epoch {epoch}')
    fig.colorbar(im,ax=ax);fig.savefig(OUT/'confusion_matrix.png',dpi=150);plt.close(fig)
    print(json.dumps(metrics,indent=2))
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--batch-size',type=int,default=4);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--require-cuda',action='store_true')
    a=p.parse_args();evaluate(a.batch_size,a.threads,a.require_cuda)
