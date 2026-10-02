"""Week 2 preprocessing. Splits ORIGINAL TRIALS before window generation."""
from pathlib import Path
import hashlib
import json
import os
import re
import time
ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp' / 'matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from scipy.signal import butter, iirnotch, filtfilt, sosfiltfilt, welch
from analyze_week1 import load_trial

OUT = ROOT / 'output' / 'week2'
CACHE = ROOT / 'data' / 'processed_week2'
FS, WIN, HOP = 1000, 300, 150
SCALES = np.arange(1, 33)
CONFIG = dict(fs=FS, window=WIN, hop=HOP, scales=SCALES.tolist(), wavelet='morl',
              normalization='joint_time_and_channels_per_window', third_channel='mean_magnitudes',
              storage='float32_normalized_windows_cwt_on_demand', filtering='additional_filter_on_already_filtered_release', bandpass=[20,499],
              butterworth_order=4, notch_hz=60, notch_q=30, split_seed=42, test_fraction=.2, split_method='subject_stratified_exact_count_hash_groups')


def preprocess(x, fs=FS):
    """Explicit additional filtering for lecture exercise; input is NOT raw EMG."""
    x = np.asarray(x, dtype=np.float64)
    bn, an = iirnotch(60, 30, fs=fs)
    notched = filtfilt(bn, an, x, axis=0)
    sos = butter(4, [20,499], btype='bandpass', fs=fs, output='sos')
    return sosfiltfilt(sos, notched, axis=0)


def make_windows(x, win=WIN, hop=HOP):
    if x.ndim != 2 or len(x) < win or win <= 0 or hop <= 0:
        raise ValueError('Expected (time, channels) with at least one full window')
    return np.stack([x[i:i+win] for i in range(0, len(x)-win+1, hop)])


def minmax(w, eps=1e-8):
    mn = w.min(axis=(1,2), keepdims=True)
    mx = w.max(axis=(1,2), keepdims=True)
    return (w-mn)/(mx-mn+eps)


def to_cwt(window):
    # Vectorize channels while preserving the lecture's independent-channel CWT.
    coef, _ = pywt.cwt(window.T, SCALES, 'morl', sampling_period=1/FS, axis=-1)
    maps = np.abs(coef).transpose(1,0,2)
    return np.concatenate([maps, maps.mean(axis=0,keepdims=True)],axis=0).astype(np.float32)


def split_trials():
    files = sorted((ROOT/'data'/'data').glob('*/*.csv'),
                   key=lambda p:(p.parent.name,int(re.search(r'\((\d+)\)',p.name)[1])))
    # Group byte-identical files before splitting, even if filenames differ.
    records=[dict(subject=p.parent.name, label=ord(p.parent.name)-ord('A'),
                  trial_id=p.relative_to(ROOT/'data'/'data').as_posix(),
                  path=p.relative_to(ROOT).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    frame=pd.DataFrame(records)
    if len(frame)!=250 or set(frame.subject)!=set('ABCDE') or not (frame.groupby('subject').size()==50).all():
        raise ValueError('Expected 250 CSV files: 50 trials per subject A-E')
    if (frame.groupby('sha256').subject.nunique()>1).any():
        raise ValueError('Identical data have conflicting subject labels')
    frame['split']='train'
    rng=np.random.default_rng(42)
    for subject, group in frame.groupby('subject',sort=True):
        groups=list(group.groupby('sha256',sort=True).groups.values())
        order=rng.permutation(len(groups))
        target=int(round(len(group)*.2))
        # Exact subset-sum selection keeps each subject at 40/10 files while
        # placing all members of each duplicate group on the same side.
        options={0:[]}
        for gi in order:
            size=len(groups[gi])
            for total,chosen in list(options.items()):
                if total+size <= target and total+size not in options:
                    options[total+size]=chosen+[int(gi)]
        if target not in options: raise ValueError('Cannot satisfy stratified count without splitting duplicate groups')
        for gi in options[target]: frame.loc[list(groups[gi]),'split']='test'
    return frame.sort_values(['split','subject','trial_id']).reset_index(drop=True)


def filter_figures():
    x=load_trial(ROOT/'data/data/A/a (1).csv')
    xf=preprocess(x)
    np.savez(CACHE/'example_stages.npz', released=x, additionally_filtered=xf,
             windows=make_windows(xf), normalized=minmax(make_windows(xf)))
    fig, ax=plt.subplots(1,2,figsize=(12,4))
    for ch in range(2):
        for y,label in [(x,'Released (already filtered)'),(xf,'Additional lecture filter')]:
            f,p=welch(y[:,ch],fs=FS,nperseg=1000)
            ax[ch].semilogy(f,np.maximum(p,1e-15),label=label,lw=1)
        ax[ch].axvline(60,color='red',ls='--',lw=.8)
        ax[ch].set(title=f'Subject A / trial 1 / CH{ch+1}',xlabel='Frequency (Hz)',ylabel='PSD (CSV units squared / Hz)',xlim=(0,300))
        ax[ch].legend(fontsize=8); ax[ch].grid(alpha=.2)
    fig.suptitle('Additional filtering check - original raw recording is unavailable')
    fig.tight_layout(); fig.savefig(OUT/'filter_check.png',dpi=150); plt.close(fig)
    # Known frequencies are a controlled functional test, not participant data.
    t=np.arange(10000)/FS
    synthetic=np.sin(2*np.pi*100*t)+.6*np.sin(2*np.pi*60*t)+.4*np.sin(2*np.pi*5*t)
    clean=preprocess(synthetic[:,None])[:,0]
    inner=slice(1000,-1000)
    def amplitude(y,hz):
        return 2*abs(np.mean(y[inner]*np.exp(-2j*np.pi*hz*t[inner])))
    attenuation={str(hz):float(20*np.log10(amplitude(clean,hz)/amplitude(synthetic,hz))) for hz in [5,60,100]}
    fig,ax=plt.subplots(figsize=(9,3.5))
    for y,label in [(synthetic,'Synthetic: 5 + 60 + 100 Hz'),(clean,'Filtered synthetic signal')]:
        f,p=welch(y[inner],fs=FS,nperseg=2000)
        ax.semilogy(f,np.maximum(p,1e-16),label=label)
    ax.set(xlim=(0,150),xlabel='Frequency (Hz)',ylabel='PSD',title='Controlled filter validation (synthetic data)')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT/'filter_synthetic_check.png',dpi=150); plt.close(fig)
    (OUT/'filter_validation.json').write_text(json.dumps({'attenuation_db':attenuation},indent=2),encoding='utf-8')
    assert attenuation['60'] < -25 and attenuation['5'] < -25 and abs(attenuation['100']) < 1
    tensor=to_cwt(minmax(make_windows(xf))[0])
    np.save(CACHE/'cwt_example.npy',tensor)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ch,ax in enumerate(axes):
        im=ax.imshow(tensor[ch],origin='lower',aspect='auto',extent=[0,.3,.5,32.5],vmin=0,vmax=tensor.max(),cmap='magma')
        ax.set(title=['APB magnitude','ADM magnitude','Mean magnitude'][ch],xlabel='Time within window (s)',ylabel='Scale (not Hz)')
    fig.colorbar(im,ax=axes,label='Absolute CWT coefficient',shrink=.8)
    fig.suptitle('First window: (3, 32, 300) | Morlet scales 1-32')
    fig.savefig(OUT/'cwt_example.png',dpi=150); plt.close(fig)


def prepare(force=False):
    OUT.mkdir(parents=True,exist_ok=True); CACHE.mkdir(parents=True,exist_ok=True)
    manifest=split_trials()
    identity=hashlib.sha256((manifest.to_json()+json.dumps(CONFIG,sort_keys=True)).encode()).hexdigest()
    marker=CACHE/'metadata.json'
    if marker.exists() and not force:
        old=json.loads(marker.read_text(encoding='utf-8'))
        if old['dataset_id']==identity and all((CACHE/f'{s}_{k}.npy').exists() for s in ['train','test'] for k in ['windows','y']):
            print('Using verified preprocessing cache:',identity[:12],flush=True)
            return old
    start=time.perf_counter()
    manifest.to_csv(OUT/'trial_split.csv',index=False,encoding='utf-8-sig')
    manifest[manifest.sha256.duplicated(keep=False)].to_csv(OUT/'duplicate_trials.csv',index=False,encoding='utf-8-sig')
    filter_figures()
    counts=[]
    for split in ['train','test']:
        subset=manifest[manifest.split==split].reset_index(drop=True)
        n=len(subset)*19
        X=np.lib.format.open_memmap(CACHE/f'{split}_windows.npy',mode='w+',dtype=np.float32,shape=(n,300,2))
        y=np.empty(n,dtype=np.int64); indexes=[]
        for i,row in enumerate(subset.itertuples()):
            filtered=preprocess(load_trial(ROOT/row.path))
            windows=minmax(make_windows(filtered))
            assert windows.shape==(19,300,2)
            for j,w in enumerate(windows):
                idx=i*19+j
                X[idx]=w.astype(np.float32); y[idx]=row.label
                indexes.append(dict(index=idx,trial_id=row.trial_id,subject=row.subject,start_sample=j*HOP,end_sample=j*HOP+WIN))
            if (i+1)%25==0: print(f'{split}: {i+1}/{len(subset)} trials',flush=True)
        X.flush(); del X
        np.save(CACHE/f'{split}_y.npy',y)
        pd.DataFrame(indexes).to_csv(OUT/f'{split}_windows.csv',index=False)
        for subject,g in subset.groupby('subject'):
            counts.append(dict(split=split,subject=subject,trials=len(g),windows=len(g)*19))
    pd.DataFrame(counts).to_csv(OUT/'dataset_summary.csv',index=False)
    result=dict(dataset_id=identity,config=CONFIG,train_shape=[3800,3,32,300],test_shape=[950,3,32,300],stored_train_shape=[3800,300,2],stored_test_shape=[950,300,2],
                preprocessing_seconds=time.perf_counter()-start,
                pywavelets_version=pywt.__version__,scale_frequencies_hz=(pywt.scale2frequency('morl',SCALES)*FS).tolist())
    marker.write_text(json.dumps(result,indent=2),encoding='utf-8')
    (OUT/'preprocessing.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('PREPARED:',result['train_shape'],result['test_shape'],flush=True)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--force',action='store_true')
    prepare(parser.parse_args().force)
