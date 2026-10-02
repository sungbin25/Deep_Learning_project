"""Fixed baselines on the Week 2 split. Never selects settings using test scores."""
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'semg_week2_RTX'))
from week2_pipeline import CACHE, OUT, prepare, to_cwt  # noqa: E402
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RESULT = ROOT / 'results/comparison'
FEATURE_SPEC = 'cwt_magnitude_log1p_time_mean_std_ddof0_channels3_scales32_v1'
MODELS = {
    'svm': dict(name='SVM (RBF)', C=1.0, gamma='scale', kernel='rbf', random_state=42),
    'random_forest': dict(name='Random Forest', n_estimators=300, max_features='sqrt',
                          min_samples_leaf=1, max_depth=None, random_state=42, n_jobs=4),
}


def pooled_features(window):
    """300x2 normalized window -> same 3x32x300 CWT -> 192 time-pooled features."""
    maps = np.log1p(to_cwt(window))
    return np.concatenate([maps.mean(axis=-1).ravel(), maps.std(axis=-1, ddof=0).ravel()]).astype(np.float32)


def features(split, dataset_id):
    cache = ROOT / 'results/cache'
    cache.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(f'{dataset_id}:{FEATURE_SPEC}:{importlib.metadata.version("PyWavelets")}'.encode()).hexdigest()[:20]
    path = cache / f'{split}_{identity}.npy'
    windows = np.load(CACHE / f'{split}_windows.npy', mmap_mode='r')
    if path.exists():
        data = np.load(path)
        if data.shape == (len(windows), 192) and np.isfinite(data).all():
            return data
    data = np.empty((len(windows), 192), dtype=np.float32)
    for i, window in enumerate(windows):
        data[i] = pooled_features(window)
        if (i + 1) % 500 == 0 or i + 1 == len(windows):
            print(f'Features {split}: {i+1}/{len(windows)}', flush=True)
    if not np.isfinite(data).all():
        raise ValueError('Nonfinite pooled CWT features')
    np.save(path, data)
    return data


def dense_predictions(metadata, index, truth):
    metrics = json.loads((OUT / 'test_metrics.json').read_text(encoding='utf-8'))
    config = json.loads((OUT / 'training_config.json').read_text(encoding='utf-8'))
    history = pd.read_csv(OUT / 'training_log.csv')
    if metrics['dataset_id'] != metadata['dataset_id'] or config['dataset_id'] != metadata['dataset_id']:
        raise ValueError('DenseNet results and current preprocessing do not match')
    if metrics['checkpoint_epoch'] != 5 or history.epoch.tolist() != [1, 2, 3, 4, 5] or not (history.samples == 3800).all():
        raise ValueError('Expected exactly five full-data DenseNet epochs')
    saved = pd.read_csv(OUT / 'test_predictions.csv')
    for col in ['index', 'trial_id', 'subject', 'start_sample', 'end_sample']:
        np.testing.assert_array_equal(saved[col].to_numpy(), index[col].to_numpy())
    np.testing.assert_array_equal(saved.true_label.to_numpy(), truth)
    pred = saved.predicted_label.to_numpy()
    if not set(pred).issubset(set(range(5))):
        raise ValueError('Invalid class labels')
    if not np.isclose(accuracy_score(truth, pred), metrics['window_accuracy']):
        raise ValueError('Stored DenseNet metrics do not match predictions')
    return pred, dict(name='DenseNet161 (5 epochs)', training_seconds=float(history.seconds.sum()),
                     inference_seconds=metrics['inference_seconds'], source='semg_week2_RTX/output/week2/test_predictions.csv',
                     source_sha256=hashlib.sha256((OUT/'test_predictions.csv').read_bytes()).hexdigest(), config=config)


def save_evaluation(key, name, truth, pred, index, dataset_id):
    precision, recall, f1, _ = precision_recall_fscore_support(truth, pred, labels=range(5), average='macro', zero_division=0)
    matrix = confusion_matrix(truth, pred, labels=range(5))
    record = dict(model=name, key=key, accuracy=float(accuracy_score(truth, pred)),
                  precision_macro=float(precision), recall_macro=float(recall), f1_macro=float(f1),
                  test_windows=len(truth), dataset_id=dataset_id)
    predictions = index.copy()
    predictions['true_label'] = truth
    predictions['predicted_label'] = pred
    predictions.to_csv(RESULT / f'{key}_predictions.csv', index=False)
    pd.DataFrame(matrix, index=list('ABCDE'), columns=list('ABCDE')).to_csv(RESULT / f'{key}_confusion_matrix.csv')
    report = classification_report(truth, pred, labels=list(range(5)), target_names=list('ABCDE'), output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(RESULT / f'{key}_classification_report.csv')
    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    im = ax.imshow(matrix, cmap='Blues', vmin=0, vmax=190)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(matrix[i, j]), ha='center', va='center', color='white' if matrix[i, j] > 95 else 'black')
    ax.set(xticks=range(5), yticks=range(5), xticklabels=list('ABCDE'), yticklabels=list('ABCDE'),
           xlabel='Predicted subject', ylabel='True subject', title=f'{name}\nTest windows (n=950)')
    fig.colorbar(im, ax=ax, label='Window count')
    fig.savefig(RESULT / f'{key}_confusion_matrix.png', dpi=160)
    plt.close(fig)
    return record


def main():
    RESULT.mkdir(parents=True, exist_ok=True)
    metadata = prepare()
    index = pd.read_csv(OUT / 'test_windows.csv')
    truth = np.load(CACHE / 'test_y.npy')
    train_y = np.load(CACHE / 'train_y.npy')
    if len(truth) != 950 or len(train_y) != 3800:
        raise ValueError('Unexpected split sizes')
    pred, dense = dense_predictions(metadata, index, truth)
    records = [save_evaluation('densenet161', dense['name'], truth, pred, index, metadata['dataset_id'])]
    started = time.perf_counter()
    train_x = features('train', metadata['dataset_id'])
    test_x = features('test', metadata['dataset_id'])
    feature_seconds = time.perf_counter() - started
    fitted_dir = ROOT / 'results/models'
    fitted_dir.mkdir(parents=True, exist_ok=True)
    run = dict(dataset_id=metadata['dataset_id'], seed=42, feature_spec=FEATURE_SPEC,
               feature_count=192, feature_seconds_this_run=feature_seconds,
               test_selection='No test-based tuning, model selection or early stopping; fixed settings',
               metric_unit='window', averaging='macro', dense=dense, baselines={},
               packages={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'PyWavelets', 'torch', 'torchvision']})
    for key, config in MODELS.items():
        params = {k: v for k, v in config.items() if k != 'name'}
        estimator = make_pipeline(StandardScaler(), SVC(**params)) if key == 'svm' else RandomForestClassifier(**params)
        started = time.perf_counter()
        estimator.fit(train_x, train_y)  # Includes scaler fitting on TRAIN ONLY.
        fit_seconds = time.perf_counter() - started
        started = time.perf_counter()
        pred = estimator.predict(test_x)
        predict_seconds = time.perf_counter() - started
        joblib.dump(dict(estimator=estimator, dataset_id=metadata['dataset_id'], feature_spec=FEATURE_SPEC), fitted_dir / f'{key}.joblib')
        record = save_evaluation(key, config['name'], truth, pred, index, metadata['dataset_id'])
        records.append(record)
        run['baselines'][key] = dict(config=config, training_seconds=fit_seconds, inference_seconds=predict_seconds,
                                     all_estimator_parameters=estimator.get_params(deep=True))
        # Stringify estimator objects only in the metadata; not used to reconstruct parameters.
        print(json.dumps(record, ensure_ascii=False), flush=True)
    pd.DataFrame(records).to_csv(RESULT / 'model_comparison.csv', index=False)
    run['completed_utc'] = datetime.now(timezone.utc).isoformat()
    (RESULT / 'run_metadata.json').write_text(json.dumps(run, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('PASS: comparison outputs saved to results/comparison', flush=True)


if __name__ == '__main__':
    main()
