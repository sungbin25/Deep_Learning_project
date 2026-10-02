"""Week 1: inspect the released filtered signals without modifying them."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "data"
OUT = ROOT / "output"
FS = 1000
PHASES = ("Grasp", "Rotate", "Stop")


def load_trial(path):
    frame = pd.read_csv(path)
    if list(frame.columns) != ["Comp Ch 3", "Comp Ch 4"]:
        raise ValueError(f"Unexpected columns: {path}: {list(frame.columns)}")
    x = frame.to_numpy(dtype=float)
    if x.shape != (3000, 2) or not np.isfinite(x).all():
        raise ValueError(f"Invalid shape or nonfinite values: {path}: {x.shape}")
    return x


def inventory():
    files = sorted(DATA.glob("*/*.csv"), key=lambda p: (p.parent.name, int(re.search(r"\((\d+)\)", p.name)[1])))
    if not files:
        raise FileNotFoundError(f"No CSV files at {DATA}. Run download_data.py first.")
    rows, phases = [], []
    for f in files:
        x = load_trial(f)
        rows.append(dict(subject=f.parent.name, trial=int(re.search(r"\((\d+)\)", f.name)[1]),
                         file=f.relative_to(ROOT).as_posix(), samples=len(x), channels=x.shape[1],
                         duration_s=len(x)/FS, minimum=float(x.min()), maximum=float(x.max()),
                         sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
        for ch in range(2):
            for i, phase in enumerate(PHASES):
                z = x[i*FS:(i+1)*FS, ch]
                phases.append(dict(subject=f.parent.name, trial=rows[-1]["trial"], channel=ch+1,
                                   phase=phase, rms=float(np.sqrt(np.mean(z*z)))))
    return pd.DataFrame(rows), pd.DataFrame(phases)


def plot_trials(meta, subjects, shared=True):
    fig, axes = plt.subplots(2, len(subjects), figsize=(5*len(subjects), 6), sharex=True, sharey=shared, squeeze=False)
    selected = [meta[meta.subject == s].sort_values("trial").iloc[0] for s in subjects]
    limit = max(np.abs(load_trial(ROOT / r.file)).max() for r in selected) * 1.08
    for col, row in enumerate(selected):
        x = load_trial(ROOT / row.file)
        for ch in range(2):
            ax = axes[ch, col]
            ax.plot(np.arange(len(x))/FS, x[:, ch], lw=.55, color=("#176b99", "#b34b34")[ch])
            for i, label in enumerate(PHASES):
                ax.axvspan(i, i+1, color=("#cfe9fa", "#ffe2ad", "#dfeddb")[i], alpha=.35)
                ax.text(i+.5, .96, label, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=9)
            for t in (1, 2):
                ax.axvline(t, color="#aa3535", ls="--", lw=.8)
            bound = limit if shared else np.abs(x[:, ch]).max() * 1.15
            ax.set(xlim=(0,3), ylim=(-bound,bound), xlabel="Time (s)")
            ax.set_title(f"Subject {row.subject} / trial {row.trial} / CH{ch+1} ({('APB','ADM')[ch]})", fontsize=10)
            ax.set_ylabel("Amplitude (CSV units)")
            ax.grid(alpha=.2)
    scale_note = "shared amplitude scale" if shared else "individual panel scales - compare waveform shape only"
    fig.suptitle(f"Filtered palm sEMG | {scale_note}", fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95))
    return fig


def main():
    OUT.mkdir(exist_ok=True)
    meta, phases = inventory()
    assert len(meta) == 250 and set(meta.subject) == set("ABCDE")
    assert (meta.groupby("subject").size() == 50).all()
    summary = meta.groupby("subject").agg(trials=("file","size"), samples=("samples","first"),
        channels=("channels","first"), duration_s=("duration_s","first"), minimum=("minimum","min"), maximum=("maximum","max"))
    meta.to_csv(OUT / "file_manifest.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "data_summary.csv", encoding="utf-8-sig")
    phases.to_csv(OUT / "phase_rms.csv", index=False)
    for subjects, name in [(["A"], "signal_example"), (["A","B","C"], "signals_ABC")]:
        fig = plot_trials(meta, subjects)
        fig.savefig(OUT / f"{name}.png", dpi=160)
        plt.close(fig)
    fig = plot_trials(meta, ["A", "B", "C"], shared=False)
    fig.savefig(OUT / "signals_ABC_detail.png", dpi=160)
    plt.close(fig)
    versions = {m:importlib.metadata.version(m) for m in ["numpy","scipy","matplotlib","pandas","scikit-learn","PyWavelets","ipykernel","nbformat","nbclient"]}
    environment = {"python":sys.version, "executable":Path(sys.executable).name, "platform":platform.platform(), "packages":versions}
    (OUT / "environment.json").write_text(json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8")
    print("OK: 250 trials, 5 subjects, 3000 samples x 2 channels, all finite")
    print(summary.to_string())
    print("\nFirst-trial RMS:", phases[phases.trial == 1].pivot(index=["subject","channel"],columns="phase",values="rms").round(5).to_string())
    return meta, summary, phases, environment


if __name__ == "__main__":
    main()
