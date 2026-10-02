"""Audit published evidence without training models or requiring local checkpoints."""
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULT = ROOT / 'results/comparison'
WEEK2 = ROOT / 'semg_week2_RTX'


def main():
    manifest = pd.read_csv(WEEK2 / 'output/week2/trial_split.csv')
    assert len(manifest) == 250
    train = manifest[manifest.split == 'train']
    test = manifest[manifest.split == 'test']
    assert len(train) == 200 and len(test) == 50
    assert not set(train.trial_id) & set(test.trial_id)
    assert not set(train.sha256) & set(test.sha256)
    assert (train.groupby('subject').size() == 40).all()
    assert (test.groupby('subject').size() == 10).all()
    for row in manifest.itertuples():
        path = WEEK2 / row.path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256, f'Data modified: {path}'
        week1 = ROOT / 'semg_week1/data/data' / row.trial_id
        assert hashlib.sha256(week1.read_bytes()).hexdigest() == row.sha256
        with path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.reader(stream))
        assert rows[0] == ['Comp Ch 3', 'Comp Ch 4']
        values = np.array(rows[1:], dtype=float)
        assert values.shape == (3000, 2) and np.isfinite(values).all()
    expected = pd.read_csv(WEEK2 / 'output/week2/test_windows.csv')
    assert len(expected) == 950 and expected['index'].tolist() == list(range(950))
    assert set(expected.trial_id) == set(test.trial_id)
    assert (expected.groupby('trial_id').size() == 19).all()
    table = pd.read_csv(RESULT / 'model_comparison.csv')
    assert set(table.key) == {'densenet161', 'svm', 'random_forest'} and len(table) == 3
    run = json.loads((RESULT / 'run_metadata.json').read_text(encoding='utf-8'))
    assert set(table.dataset_id) == {run['dataset_id']}
    source = ROOT / run['dense']['source']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == run['dense']['source_sha256']
    for row in table.itertuples():
        prediction = pd.read_csv(RESULT / f'{row.key}_predictions.csv')
        assert len(prediction) == 950
        for column in expected.columns:
            np.testing.assert_array_equal(expected[column], prediction[column])
        truth = np.array([ord(s) - ord('A') for s in expected.subject])
        np.testing.assert_array_equal(truth, prediction.true_label)
        pred = prediction.predicted_label.to_numpy(dtype=int)
        assert set(pred).issubset(set(range(5)))
        matrix = np.zeros((5, 5), dtype=int)
        np.add.at(matrix, (truth, pred), 1)
        saved_matrix = pd.read_csv(RESULT / f'{row.key}_confusion_matrix.csv', index_col=0)
        assert saved_matrix.columns.tolist() == list('ABCDE') and saved_matrix.index.tolist() == list('ABCDE')
        np.testing.assert_array_equal(matrix, saved_matrix)
        tp = matrix.diagonal().astype(float)
        precision = np.divide(tp, matrix.sum(0), out=np.zeros(5), where=matrix.sum(0) != 0)
        recall = tp / matrix.sum(1)
        f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(5), where=(precision+recall) != 0)
        np.testing.assert_allclose([row.accuracy, row.precision_macro, row.recall_macro, row.f1_macro],
                                   [tp.sum()/950, precision.mean(), recall.mean(), f1.mean()], atol=1e-12, rtol=0)
        assert (RESULT / f'{row.key}_confusion_matrix.png').read_bytes().startswith(b'\x89PNG')
    history = pd.read_csv(WEEK2 / 'output/week2/training_log.csv')
    assert history.epoch.tolist() == [1,2,3,4,5] and (history.samples == 3800).all()
    reproduction_checked = False
    reproduction = ROOT / 'results/reproduction_check'
    if (reproduction / 'check.json').exists():
        rerun = pd.read_csv(reproduction / 'model_comparison.csv')
        recorded = json.loads((reproduction / 'check.json').read_text(encoding='utf-8'))
        assert recorded['status'] == 'passed' and recorded['preprocessing_arrays_identical']
        assert len(rerun) == 3 and set(rerun.key) == set(table.key)
        for row in rerun.itertuples():
            p = pd.read_csv(reproduction / f'{row.key}_predictions.csv')
            for column in expected.columns:
                np.testing.assert_array_equal(expected[column], p[column])
            truth = np.array([ord(s)-65 for s in expected.subject])
            np.testing.assert_array_equal(truth, p.true_label)
            matrix = np.zeros((5,5), dtype=int)
            pred = p.predicted_label.to_numpy(dtype=int)
            assert set(pred).issubset(set(range(5)))
            np.add.at(matrix, (truth, pred), 1)
            np.testing.assert_array_equal(matrix, pd.read_csv(reproduction / f'{row.key}_confusion_matrix.csv', index_col=0))
            f1 = (2*matrix.diagonal()/(matrix.sum(0)+matrix.sum(1))).mean()
            np.testing.assert_allclose([row.accuracy, row.f1_macro], [matrix.trace()/950, f1], atol=1e-12, rtol=0)
        reproduction_checked = True
    executed = {}
    for relative in ['semg_week1/week1_semg.ipynb', 'semg_week2_RTX/week2_semg.ipynb']:
        nb = json.loads((ROOT / relative).read_text(encoding='utf-8'))
        code = [c for c in nb['cells'] if c['cell_type'] == 'code']
        assert all(c['execution_count'] is not None for c in code)
        assert not any(o['output_type'] == 'error' for c in code for o in c['outputs'])
        executed[relative] = len(code)
    readme = (ROOT/'README.md').read_text(encoding='utf-8')
    links = re.findall(r'\]\(([^)]+)\)', readme)
    for target in links:
        if target.startswith(('https://', 'http://', '#')):
            continue
        if target == 'results/comparison/verification.json':
            continue  # Written only after all other checks pass.
        assert (ROOT / target.split('#')[0]).exists(), f'Broken README link: {target}'
    result = dict(status='passed', validated_original_csv_files=250, week1_week2_identical_files=250,
                  unique_signal_contents=manifest.sha256.nunique(), train_test_trial_overlap=0,
                  train_test_content_overlap=0, models=table.key.tolist(), test_windows_per_model=950,
                  metrics_recomputed_from_predictions=True, confusion_matrices_match_predictions=True,
                  dense_epochs=5, executed_notebook_cells=executed, readme_local_links_exist=True)
    result['additional_reproduction_metrics_verified'] = reproduction_checked
    (RESULT/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
