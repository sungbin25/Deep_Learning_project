"""Reproduce Week 1, Week 2 and the fixed three-model comparison."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
WEEK2 = ROOT / 'semg_week2_RTX'


def run_script(path, *args):
    path = ROOT / path
    print(f'\n>>> {path.name} {" ".join(args)}', flush=True)
    subprocess.run([sys.executable, '-u', '-X', 'utf8', str(path), *args],
                   cwd=path.parent, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['full', 'compare', 'report'], default='full',
                        help='full: reproduce all; compare: use saved DenseNet predictions; report: rebuild README only')
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    if args.mode != 'report':
        if args.mode == 'full':
            # Download only if CSV data are absent; the submitted repository contains them.
            run_script('semg_week1/download_data.py')
            run_script('semg_week2_RTX/download_data.py')
            run_script('semg_week1/run_notebook.py')
        run_script('semg_week2_RTX/week2_pipeline.py')
        run_script('semg_week2_RTX/test_week2.py')
        if args.mode == 'full':
            import torch
            checkpoint = WEEK2 / 'output/week2/checkpoints/last.pt'
            train_args = ['--epochs', '5', '--batch-size', '16', '--threads', '4']
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
                if saved['epoch'] > 5:
                    raise RuntimeError('This comparison fixes DenseNet at 5 epochs. Use a fresh clone for this experiment; existing >5 epoch checkpoints are preserved.')
                del saved
                train_args.append('--resume')
            if args.require_cuda:
                train_args.append('--require-cuda')
            run_script('semg_week2_RTX/train_week2.py', *train_args)
            eval_args = ['--batch-size', '16', '--threads', '4']
            if args.require_cuda:
                eval_args.append('--require-cuda')
            run_script('semg_week2_RTX/evaluate_week2.py', *eval_args)
            run_script('semg_week2_RTX/run_week2_notebook.py')
        run_script('compare_models.py')
    run_script('make_submission_readme.py')
    run_script('verify_submission.py')
    print('\nDONE: README.md and results/comparison/', flush=True)


if __name__ == '__main__':
    main()
