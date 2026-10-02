"""Independent numerical/data-integrity checks for the Week 2 pipeline."""
import json
import numpy as np
import pandas as pd
import pywt
from week2_pipeline import OUT,CACHE,make_windows,minmax,to_cwt,preprocess,FS

x=np.arange(6000,dtype=float).reshape(3000,2)
w=make_windows(x)
assert w.shape==(19,300,2)
np.testing.assert_array_equal(w[-1],x[2700:3000])
np.testing.assert_array_equal(w[0,150:],w[1,:150])
assert np.all(minmax(np.ones((2,300,2)))==0)
n=minmax(w)
assert np.isfinite(n).all() and n.min()>=0 and n.max()<=1
reference=np.stack([np.abs(pywt.cwt(n[0,:,ch],np.arange(1,33),'morl')[0]) for ch in range(2)])
cwt=to_cwt(n[0])
np.testing.assert_allclose(cwt[:2],reference,rtol=1e-6,atol=1e-7)
np.testing.assert_allclose(cwt[2],cwt[:2].mean(0),rtol=1e-6,atol=1e-7)
manifest=pd.read_csv(OUT/'trial_split.csv')
tr=manifest[manifest.split=='train']; te=manifest[manifest.split=='test']
assert len(tr)==200 and len(te)==50
assert not (set(tr.trial_id)&set(te.trial_id))
assert not (set(tr.sha256)&set(te.sha256)), 'Identical signal files cross split'
assert (tr.groupby('subject').size()==40).all() and (te.groupby('subject').size()==10).all()
for split,count in [('train',3800),('test',950)]:
    x=np.load(CACHE/f'{split}_windows.npy',mmap_mode='r'); y=np.load(CACHE/f'{split}_y.npy')
    ix=pd.read_csv(OUT/f'{split}_windows.csv')
    assert x.shape==(count,300,2) and x.dtype==np.float32
    assert len(y)==count and set(y)==set(range(5))
    assert (ix.groupby('trial_id').size()==19).all()
    assert set(ix.trial_id)==set(manifest[manifest.split==split].trial_id)
    for block in range(0,count,64): assert np.isfinite(x[block:block+64]).all()
    for idx in [0,count//2,count-1]:
        tensor=to_cwt(x[idx]); assert tensor.shape==(3,32,300) and np.isfinite(tensor).all()
    np.testing.assert_array_equal(y,np.array([ord(s)-ord('A') for s in ix.subject]))
a=json.loads((OUT/'filter_validation.json').read_text())['attenuation_db']
assert a['60'] < -25 and a['5'] < -25 and abs(a['100']) < 1
print('PASS: window boundaries/overlap, constant normalization, independent CWT equivalence, trial/hash isolation, labels, shapes, finite tensors, filter response')
