# 2주차 실습 실행 안내

2026-09-22 현재 PC의 NVIDIA GeForce RTX 4060 Laptop GPU에서 **5에폭 학습·테스트 평가·노트북 실행을 완료**했습니다. 프로젝트 `.venv`에 Python 3.12.14, PyTorch 2.10.0+cu128, Torchvision 0.25.0+cu128을 설치했으며 원본 전처리 캐시를 재사용했습니다.

- 전체 3,800개 샘플 × 5에폭, 순수 학습 시간 274.49초
- 학습 loss: 1.0142 → 0.4665, 마지막 에폭 학습 정확도: 82.32%
- 테스트 윈도우 정확도: 81.47% (774/950), macro F1: 0.8109
- 시행별 평균 확률 정확도: 98.00% (49/50)
- 데이터 검증 통과, 노트북 코드 7개 셀 오류 없이 실행 완료

윈도우와 시행의 평가는 집계 단위가 다릅니다. 이 결과는 5에폭 실습 결과이며, 45에폭 학습 결과로 해석하지 않습니다. 실제 로그와 환경 목록은 `output/week2`에 저장했습니다.

## 실행과 재개

프로젝트 폴더의 PowerShell에서 실행합니다.

```powershell
.\.venv\Scripts\python.exe run_gpu_pipeline.py --epochs 5 --batch-size 16 --threads 4
```

이 명령은 GPU 확인 → 전처리 확인 → 데이터 검증 → 5에폭 학습 → 테스트 평가 → 노트북·HTML 생성을 실행합니다. `02_RUN_GPU.cmd`로도 같은 작업을 실행할 수 있습니다. 완료 에폭의 체크포인트가 있으면 자동 재개하며, 목표 에폭까지 완료되어 있으면 추가 학습 없이 평가와 보고서를 갱신합니다.

45에폭까지 연장하려면 다음 명령 또는 `03_CONTINUE_TO_45.cmd`를 사용합니다. 테스트 결과를 보고 에폭을 고르지 말고 실험 계획에 따라 실행하세요.

```powershell
.\.venv\Scripts\python.exe run_gpu_pipeline.py --epochs 45 --batch-size 16 --threads 4
```

DenseNet161 전체를 사전학습 없이 Adam(lr=0.001), CrossEntropyLoss, float32로 학습합니다. 배치는 16, CPU 스레드는 4이며 CUDA 사용을 필수로 검사합니다. 에폭마다 모델·optimizer 상태를 저장하고, 중단 시 마지막 완료 에폭 다음부터 재개합니다. 재개할 때 배치 크기를 유지하세요.

## 결과 확인

- 제출용 노트북: `week2_semg.ipynb`
- 브라우저 보고서: `output/week2/week2_report.html`
- 에폭별 loss·학습 정확도: `output/week2/training_log.csv`
- 테스트 성능: `output/week2/test_metrics.json`
- 혼동행렬: `output/week2/confusion_matrix.png`
- 모델 및 optimizer: `output/week2/checkpoints/last.pt`
- 실제 GPU 환경: `output/week2/gpu_environment.json`
- 전체 실행 로그: `output/week2/gpu_console.log`

VS Code에서 노트북을 열 때 커널은 `.venv/Scripts/python.exe`를 선택하세요. 기본 Run All은 저장된 결과 표시와 전처리 검증만 수행합니다. 학습을 다시 시작하지 않고 보고서만 갱신하려면 다음을 실행합니다.

```powershell
.\.venv\Scripts\python.exe run_week2_notebook.py
```

## 데이터 및 해석

| 구분 | 시행 파일 | 윈도우 | 피험자별 윈도우 |
|---|---:|---:|---:|
| 학습 | 200 | 3,800 | 760 |
| 테스트 | 50 | 950 | 190 |

윈도우는 300샘플 × 2채널, 홉은 150샘플입니다. 윈도우 내 시간·채널 축 전체를 min-max 정규화하고, CWT를 계산해 `(3, 32, 300)` float32 입력을 만듭니다.

원본 CSV 250개 중 고유 내용은 249개입니다. 동일 내용인 E의 38번·50번 파일은 같은 집합에 유지하며, 피험자별 40:10 시행 비율로 먼저 분할한 뒤 윈도우를 만듭니다. `test_week2.py`는 경계·정규화·CWT·레이블·시행 및 해시 분할을 검증합니다.

공개 CSV는 이미 필터링된 신호입니다. 강의 실습용으로 60Hz 노치와 20~499Hz 대역통과를 추가 적용했으며, 원시 신호를 복구한 것은 아닙니다. Morlet 스케일 1~32, 세 번째 채널의 평균 맵, 기본 DenseNet stem은 강의 구현을 따릅니다. 논문의 전체 구현을 정밀 재현한 결과로 해석하지 않습니다.

강의 19쪽의 6가지 항목은 노트북에 포함합니다. 학습 로그 요건은 실제 3,800개 샘플 전체를 순회한 최소 5에폭으로 확인합니다. 통합 제출과 3개 모델 성능 비교는 루트의 [README](../README.md)를 참고하세요. 제출 주소는 https://github.com/sungbin25/Deep_Learning_project 입니다.

## 환경 메모

이 PC의 가상환경은 Codex에 포함된 Python 3.12.14를 기반으로 생성했습니다. 다른 PC에는 `.venv`를 복사하지 말고 통합 README의 설치 절차를 사용하세요. 현재 저장소는 루트의 `verify_submission.py`로 데이터와 결과를 검증합니다. 원본 ZIP 포장 매니페스트와 대용량 가중치는 Git에 포함하지 않습니다.
