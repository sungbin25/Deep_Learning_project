# 다른 NVIDIA RTX 게이밍 컴퓨터에서 실행하기

통합 GitHub 저장소에서는 루트의 [README](../README.md)와 `python main.py`를 우선 사용하세요. 아래는 2주차 폴더만 별도로 실행하는 기존 안내입니다.

현재 PC에는 `.venv` GPU 환경을 구성했습니다. 현재 실행 기록과 결과는 `output/week2/gpu_environment.json`, `gpu_run_status.json`, `training_log.csv`에서 확인하세요. 아래 설치·이동 안내는 다른 PC에 새 환경을 만드는 경우에도 사용할 수 있습니다.

## 옮길 파일

`semg_week2_RTX_windows.zip` 하나를 USB 또는 파일 전송으로 새 컴퓨터에 복사하세요. **ZIP 내부에서 바로 실행하지 말고 전체 압축을 해제**하세요. 예: `D:\Projects\semg_week2_RTX`.

원본 배포 ZIP에는 CSV 250개, 저자 라이선스, 전처리 캐시, 분할 목록, 노트북, 학습·평가 코드가 포함되어 있습니다. 배포 당시 `.venv`와 학습 가중치는 포함되지 않았습니다. 이후 생성된 체크포인트를 함께 옮기면 마지막 완료 에폭에서 재개할 수 있으며, 원본 ZIP만 풀면 1에폭부터 시작합니다.

## 새 컴퓨터 준비

- Windows 10/11 64비트, NVIDIA GeForce RTX
- NVIDIA 그래픽 드라이버 업데이트
- **Python 3.12 64비트** 설치 (Python Launcher 포함 권장)
- 초기 패키지 다운로드를 위한 인터넷
- 여유 디스크 15~20GB, 시스템 메모리 16GB 이상 권장
- 결과를 VS Code로 보려면 Microsoft Python 및 Jupyter 확장 설치

## 더블클릭으로 실행

1. **`01_SETUP_GPU.cmd`** 실행: 가상환경과 CUDA용 PyTorch를 설치하고 GPU에서 작은 행렬 연산을 확인합니다. 설치 완료 메시지를 확인하세요.
2. **`02_RUN_GPU.cmd`** 실행: 5에폭 학습 → 테스트 평가 → 결과 노트북·HTML 생성까지 자동으로 진행합니다. 실행 중인 명령창을 닫지 말고 PC가 절전 모드로 전환되지 않게 하세요.
3. 완료 후 **`week2_semg.ipynb`**를 VS Code에서 열거나 **`output/week2/week2_report.html`**을 브라우저에서 엽니다.

GPU 사용 확인은 `output/week2/gpu_environment.json`, 진행 단계는 `output/week2/gpu_run_status.json`, 전체 출력은 `output/week2/gpu_console.log`에 저장됩니다. 학습 설정의 device가 cuda인지 확인할 수 있습니다. 실제 학습 시간은 GPU·메모리·드라이버 환경에 따라 달라집니다.

## 고정한 설치 구성

- PyTorch **2.10.0** + Torchvision **0.25.0**, CUDA **12.8** wheel
- Python **3.12**
- 분석 패키지는 `requirements-gpu.txt`에 버전 고정

위 PyTorch·Torchvision·CUDA 조합은 [PyTorch 공식 설치 명령](https://pytorch.org/get-started/previous-versions/#v2100)을 기준으로 정했습니다. CUDA Toolkit을 별도로 설치하는 대신 CUDA 런타임이 포함된 PyTorch wheel을 사용합니다. 드라이버는 새 PC에 설치되어 있어야 합니다. 기존 CPU용 `requirements-week2-lock.txt`나 다른 PC의 `.venv`를 새 GPU 환경에 복사하지 마세요.

## 학습 설정과 재개

기본은 전체 학습 윈도우 3,800개, **5에폭·배치 16·Adam lr 0.001·DenseNet161**입니다. CPU 시험 실행의 배치 4와는 달리 강의 설정인 16을 사용합니다. 사전학습 가중치, 일부 데이터만 사용한 가짜 에폭, backbone 동결을 사용하지 않습니다.

각 에폭 완료 시 `output/week2/checkpoints/last.pt`에 모델과 Adam 상태를 함께 저장합니다. 중단 후 같은 `02_RUN_GPU.cmd`를 다시 실행하면 마지막 완료 에폭에서 재개합니다. 미완료 에폭의 진행분은 다시 계산합니다. 이미 5에폭이 완료되면 학습 추가 없이 저장된 체크포인트 평가와 보고서 생성을 실행합니다.

45에폭까지 더 학습하려면 **`03_CONTINUE_TO_45.cmd`**를 실행합니다. 테스트 정확도로 에폭 수를 선택하지 말고, 비교할 에폭 수를 미리 정하세요.

GPU 메모리 부족 시 프로젝트 PowerShell에서 다음처럼 배치를 줄일 수 있습니다.

```powershell
.\.venv\Scripts\python.exe run_gpu_pipeline.py --epochs 5 --batch-size 8 --threads 4
```

이미 완료 에폭의 체크포인트가 있으면 재개 시 배치 크기를 바꿀 수 없습니다. 설정을 바꾼 별도 실험은 ZIP을 새 폴더에 다시 풀어 시작하세요. 기존 체크포인트를 임의로 지우거나 덮어쓰지 않습니다.

## 자동으로 생기는 결과

| 파일 | 내용 |
|---|---|
| `output/week2/training_log.csv` | 에폭별 loss·학습 정확도·시간 |
| `output/week2/checkpoints/last.pt` | 마지막 완료 에폭의 모델·optimizer |
| `output/week2/test_metrics.json` | 테스트 윈도우 accuracy·macro F1·시행 accuracy |
| `output/week2/test_predictions.csv` | 950개 윈도우별 예측 |
| `output/week2/test_trial_predictions.csv` | 50개 시행별 평균 확률·예측 |
| `output/week2/confusion_matrix.png` | 혼동행렬 |
| `week2_semg.ipynb` | 실제 실행 결과를 포함한 제출용 노트북 |
| `output/week2/week2_report.html` | 브라우저용 보고서 |

결과를 이 컴퓨터로 다시 가져오려면 `week2_semg.ipynb`와 `output/week2` 폴더를 복사하세요. 학습 재개가 필요하면 `checkpoints/last.pt`도 포함하세요.

## 오류가 나면

- `py -3.12`를 찾지 못함: Python 3.12 64비트와 Launcher를 설치한 뒤 다시 실행하세요. 직접 경로를 줄 수도 있습니다: `powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_gpu.ps1 -PythonExe "C:\path\to\python.exe"`.
- NVIDIA 드라이버 또는 CUDA 확인 실패: `nvidia-smi` 출력과 `gpu_environment.json`을 확인하고 드라이버를 업데이트하세요. GPU 확인 실패 시 학습은 시작하지 않습니다.
- CUDA out of memory: 새 실험에서 배치를 8 또는 4로 줄이세요.
- 설치 스크립트 차단: `01_SETUP_GPU.cmd`는 해당 PowerShell 프로세스에만 실행 정책 예외를 적용하며 시스템 전체 정책을 변경하지 않습니다. 조직 정책으로 차단되면 관리자의 정책을 따르세요.
- 데이터가 손상됐다는 메시지: 통합 저장소 루트에서 `python verify_submission.py`로 원본 CSV 해시와 결과를 확인하세요. 원본 ZIP용 포장 매니페스트 검사는 통합 저장소에서 사용하지 않습니다.

## 데이터와 해석

동일 내용인 E의 38·50번 파일은 모두 학습 집합에 넣어 테스트 누수를 막았습니다. 학습 200파일/3,800윈도우, 테스트 50파일/950윈도우를 유지합니다. 파일 250개의 고유 내용은 249개입니다.

필터 추가 적용, Morlet 스케일 및 입력 채널 구성은 강의 실습 기준입니다. 논문 전체 구현을 정확히 재현했다고 주장하지 않으며, 실험 설정상의 차이는 노트북에 기록했습니다. 원본 데이터는 © 2025 Yeonjung Shin, CC BY 4.0이며 `data/README.md`와 `data/LICENSE`를 함께 제공합니다.
