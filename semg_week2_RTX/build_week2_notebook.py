"""Build a readable Week 2 notebook; keeps missing training results explicit."""
from pathlib import Path
import nbformat as nbf
ROOT=Path(__file__).resolve().parent
md=nbf.v4.new_markdown_cell
code=nbf.v4.new_code_cell
nb=nbf.v4.new_notebook()
nb.cells=[
md('''# 손바닥 sEMG 사용자 식별 · 2주차
## 전처리 · DenseNet161 학습 · 테스트 평가

**기준**: `Step1_2주차_강의와실습.pdf` 실습 7단계 및 19쪽 제출 항목.

이 프로젝트는 **NVIDIA RTX Windows 환경에서 전체 데이터 5에폭 학습 및 테스트 평가**를 수행한다. 실행 순서는 `GPU_START_HERE.md`를 참고한다. 아래 환경·설정·학습·평가 결과는 저장된 실제 실행 파일에서 읽는다. 이 노트북은 1주차의 실제 공개 데이터를 이어서 사용한다. 데이터에 필터를 추가 적용하는 실습과 원 연구의 재현을 구분한다. 학습 로그는 실제 파일이 있을 때만 표시하며, 아직 생성되지 않은 결과를 임의로 채우지 않는다.

출처: [대상 논문](https://doi.org/10.1038/s41598-026-46294-3), [공개 데이터 및 CC BY 4.0](https://github.com/sea3551/palm-sEMG-doorknob-filtered), 제공 2주차 강의자료.

## 먼저 확인할 차이점

- 공개 CSV는 이미 필터링되어 있다. 아래 before는 원시 근전도가 아니라 저자가 제공한 신호이며 after는 강의 필터를 **추가 적용한 신호**다.
- 20~499 Hz 4차 Butterworth와 60 Hz/Q=30 노치를 사용한다. 시간축에 양방향 필터를 적용하고, 대역통과는 수치 안정성을 위해 SOS로 구현한다.
- 파일명이 다른 E의 38번·50번 CSV가 바이트 단위로 동일하다. 같은 파일뿐 아니라 같은 SHA-256을 가진 파일도 한 집합에 묶어서 분할한다.
- 전체 CWT를 디스크에 저장하면 약 547MB가 필요하므로 정규화 윈도우 약 11.4MB를 저장하고 배치마다 CWT를 계산한다. 샘플 수나 CNN 입력 크기는 줄이지 않는다.
'''),
code('''from pathlib import Path
import json, sys, shutil, subprocess, importlib.metadata
import numpy as np
import pandas as pd
from IPython.display import display, Image, Markdown
from week2_pipeline import ROOT, OUT, CACHE, prepare, preprocess, make_windows, minmax, to_cwt, split_trials
metadata = prepare()
print('Python:', sys.version)
print('Interpreter:', Path(sys.executable).name)
print('Disk free (GiB):', round(shutil.disk_usage(ROOT).free / 1024**3, 2))
print('Preprocessing cache:', CACHE.relative_to(ROOT))
from check_week2 import check
readiness = check()
for filename in ['gpu_environment.json', 'training_config.json']:
    result_path = OUT / filename
    if result_path.exists():
        print(filename)
        display(pd.Series(json.loads(result_path.read_text(encoding='utf-8'))))
'''),
md('''## 1~2. 필터 적용 및 검증
`filter_check.png`는 실데이터의 추가 필터 전후 PSD다. 원자료가 이미 필터링되어 있어 60 Hz의 큰 봉우리가 나타나지 않을 수 있다. 별도의 합성 신호(5, 60, 100 Hz 성분)로 필터 기능을 검증한다. 합성 신호의 감쇠 수치를 실제 피험자의 잡음 제거 성능으로 해석하지 않는다.
'''),
code('''import inspect
print(inspect.getsource(preprocess))
display(Image(filename=str(OUT/'filter_check.png')))
display(Image(filename=str(OUT/'filter_synthetic_check.png')))
filter_metrics=json.loads((OUT/'filter_validation.json').read_text())
display(pd.DataFrame([{'frequency_hz': float(f), 'amplitude_change_db': v} for f,v in filter_metrics['attenuation_db'].items()]))
'''),
md('''## 3~4. 윈도우와 정규화
3,000개 샘플을 300개씩 자르고 150개씩 이동하면 `(3000-300)/150+1 = 19`개 윈도우다. 정규화는 강의 코드와 같이 각 윈도우의 시간·채널 두 축을 함께 사용한다. 따라서 두 채널의 상대 진폭 비율은 남으며, 각 채널이 독립적으로 0~1이 되는 방식과 다르다.
'''),
code('''stages=np.load(CACHE/'example_stages.npz')
w=stages['windows']; wn=stages['normalized']
print('Released:', stages['released'].shape)
print('Windows:', w.shape)
print('Normalized range:', wn.min(), wn.max())
assert w.shape==(19,300,2)
assert np.isfinite(wn).all() and wn.min()>=0 and wn.max()<=1
print('Constant input test:', minmax(np.ones((1,300,2))).max())
'''),
md('''## 5. CWT → (3, 32, 300)
채널별 Morlet CWT 절댓값을 만든 뒤 세 번째 채널에 두 맵의 평균을 넣는다. 이는 강의자료의 구체적 구성으로, 논문의 모든 미공개 구현 세부사항을 확인한 것은 아니다.

스케일은 주파수(Hz)가 아니다. 아래 그림은 스케일 축으로 표시한다. `morl`의 scale 1 중심 주파수 환산값은 1,000 Hz 샘플링에서 812.5 Hz로 나이퀴스트를 넘으므로, 작은 스케일의 앨리어싱 가능성을 기록한다. 실습과의 일치를 위해 1~32를 유지했으며 후속 실험에서는 주파수 대역에 맞춘 스케일 선택을 비교할 수 있다. [PyWavelets 문서](https://pywavelets.readthedocs.io/en/latest/ref/cwt.html)
'''),
code('''tensor=to_cwt(wn[0])
assert tensor.shape==(3,32,300)
print('Tensor:', tensor.shape, tensor.dtype)
np.testing.assert_allclose(tensor[2],tensor[:2].mean(0),rtol=1e-6,atol=1e-7)
display(Image(filename=str(OUT/'cwt_example.png')))
display(pd.DataFrame({'scale':metadata['config']['scales'],'equivalent_frequency_hz':metadata['scale_frequencies_hz']}).head())
'''),
md('''## 6. 시행 단위 분할과 중복 방지
**아래 함수는 필터링·윈도우 생성 전에 호출된다.** 먼저 피험자별로 파일 SHA-256 그룹을 만들고 각 피험자의 테스트 파일 10개를 선택한다. 같은 내용의 파일을 분리하지 않으면서 학습 40개·테스트 10개를 유지한다. 따라서 단순 `train_test_split(random_state=42)`와는 다른 분할이며, 해시 그룹 단위 분할로 강화했다.

250개 파일의 고유 내용은 249개다. 중복 파일은 원본에서 삭제하지 않고 같은 집합에 유지하므로, 표의 시행 수는 독립 획득 횟수가 아니라 파일 수를 뜻한다.
'''),
code('''print(inspect.getsource(split_trials))
manifest=pd.read_csv(OUT/'trial_split.csv')
summary=pd.read_csv(OUT/'dataset_summary.csv')
display(summary)
display(pd.read_csv(OUT/'duplicate_trials.csv')[['subject','trial_id','split']])
train=manifest[manifest.split=='train']; test=manifest[manifest.split=='test']
assert not set(train.trial_id)&set(test.trial_id)
assert not set(train.sha256)&set(test.sha256)
print('Trial overlap: 0; exact-content overlap: 0')
print('Train windows:',summary[summary.split=='train'].windows.sum())
print('Test windows:',summary[summary.split=='test'].windows.sum())
print('Unique contents:', manifest.sha256.nunique())
result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'test_week2.py')],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',check=True)
print(result.stdout)
'''),
md('''## 7. DenseNet161 학습
사전학습 없이 DenseNet161 전체 파라미터를 학습하며, 출력은 5클래스다. Adam 학습률 0.001, CrossEntropyLoss를 사용한다. GPU 실행은 강의 설정인 배치 16을 사용하고, DenseNet의 `memory_efficient=True`로 중간 활성값을 역전파 때 재계산한다. 실제 배치·장치·패키지 버전은 첫 셀의 저장된 실행 설정에서 확인한다. 기본 stem과 dropout 0은 강의 코드 기준이며, 논문에 언급된 수정 stem·dropout의 정확한 세부 값을 재현했다고 주장하지 않는다.

제출 기준은 **완전한 학습셋을 순회한 5에폭 이상의 실제 로그**다. 미니배치 5회나 일부 데이터 5회로 대신하지 않는다. 데이터셋별 학습 로그·가중치·optimizer 상태를 저장하며 `--resume --epochs 45`는 누적 45에폭까지 이어서 학습한다. 학습 정확도는 일반화 성능이 아니며, 이번 단계에서는 테스트셋으로 모델 선택을 하지 않는다.

### 설치와 실행 (프로젝트 PowerShell)
```powershell
.\\.venv\\Scripts\\python.exe run_gpu_pipeline.py --epochs 5 --batch-size 16 --threads 4
# 완료 후 45에폭까지 이어서 실행
.\\.venv\\Scripts\\python.exe run_gpu_pipeline.py --epochs 45 --batch-size 16 --threads 4
```

현재 설치 상태와 여유 디스크 공간은 첫 셀의 readiness 결과에서 확인한다. 학습 스크립트는 체크포인트 교체를 위해 최소 1GiB 여유 공간을 검사한다. 전체 실행 명령은 마지막 완료 에폭에서 재개한 뒤 평가와 보고서 생성까지 수행한다.
'''),
code('''# 기본 Run All은 저장된 결과를 표시한다. 실제 학습은 CLI 또는 아래 스위치로 실행한다.
RUN_TRAINING = False
if RUN_TRAINING:
    from train_week2 import train as run_training
    run_training(epochs=5,batch_size=16,threads=4,resume=(OUT/'checkpoints/last.pt').exists(),require_cuda=True)
log_path=OUT/'training_log.csv'
if log_path.exists():
    history=pd.read_csv(log_path)
    display(history)
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,3.5))
    ax.plot(history.epoch,history.train_loss,marker='o')
    ax.set(xlabel='Completed epoch',ylabel='Training cross entropy',title='Full-dataset epoch loss')
    ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT/'training_loss.png',dpi=150)
    plt.close(fig); display(Image(filename=str(OUT/'training_loss.png')))
    enough = len(history)>=5 and (history.samples==3800).all()
    print('Submission training requirement:', 'PASS' if enough else 'NOT YET MET')
else:
    print('아직 완료된 에폭 로그가 없습니다. 학습 진행 상태는 output/week2/training_progress.json에서 확인하세요.')
'''),
md('''## 8. 분리된 테스트셋 평가
학습 종료 후 마지막 체크포인트로 테스트 윈도우 950개와 시행 50개를 평가한다. 시행 예측은 각 시행에 속한 19개 윈도우의 클래스 확률을 평균한 뒤 최대값을 선택한다. 테스트셋 결과를 이용해 최적 에폭이나 설정을 고르지 않는다.

이는 5에폭의 강의 실습 결과이며 논문의 45에폭 성능을 재현했다고 주장하지 않는다. 겹치는 윈도우는 독립 관측이 아니므로, 윈도우 정확도와 시행 정확도를 구분해서 본다.
'''),
code('''metrics_path=OUT/'test_metrics.json'
if metrics_path.exists():
    metrics=json.loads(metrics_path.read_text())
    assert metrics['dataset_id']==metadata['dataset_id']
    display(pd.DataFrame([metrics]))
    display(pd.read_csv(OUT/'classification_report.csv',index_col=0))
    display(Image(filename=str(OUT/'confusion_matrix.png')))
    print('Test labels: A, B, C, D, E; window metric and trial metric are distinct.')
else:
    print('테스트 평가 결과가 아직 없습니다.')
'''),
md('''## 설정 근거 메모

**윈도우 300ms — 두 문장**

1. 1,000 Hz 신호에서 300ms는 300개 샘플로, 한 동작 내의 짧은 근활성 변화를 학습 단위로 표현한다.
2. 강의·논문의 설정에 맞춰 150ms 홉을 사용하면 50% 중첩으로 시간적 연속성을 일부 유지하지만, 인접 조각이 독립 표본은 아니므로 시행 단위 분할을 먼저 수행한다.

**스케일 32 — 두 문장**

1. 32개 스케일은 시간–주파수 표현의 정보량과 계산량을 절충하려는 강의·논문의 입력 높이 설정이다.
2. 이번 실습은 Morlet 스케일 1~32를 사용하되 이 선택이 최적이라고 단정하지 않으며, 특히 나이퀴스트를 넘는 작은 스케일의 해석에는 주의한다.

## 제출 항목 점검

- 필터 검증 그림: 완료, 공개 신호 추가 필터와 합성 검증을 구별
- CWT 예시 그림: 완료
- 데이터셋 요약: 완료, 학습 3,800 / 테스트 950 윈도우
- 시행 단위 분할 코드: 표시, 중복 내용까지 격리
- 최소 5에폭 학습 로그: 위 셀의 실제 상태 참고, 로그가 없으면 제출 기준 미충족
- 설정 근거 메모: 완료
- 제출 주소: https://github.com/sungbin25/Deep_Learning_project (통합 README에 3개 모델 비교 포함)

참고 구현: [Torchvision DenseNet161](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet161.html), [SciPy SOS 양방향 필터](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfiltfilt.html).
''')]
nb.metadata={'kernelspec':{'display_name':'Python (.venv sEMG)','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.12'}}
nbf.write(nb,ROOT/'week2_semg.ipynb')
print('Created week2_semg.ipynb')
