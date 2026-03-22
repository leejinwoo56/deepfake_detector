# DeepfakeDetector - GenD-CLIP 기반 딥페이크 탐지

CLIP ViT-L/14를 기반으로 한 딥페이크 탐지 학습 프로젝트입니다.  
`pre-projection CLS(1024)` 임베딩을 활용하고, `LayerNorm + 분류 헤드(+옵션 주파수 브랜치)`만 학습하는 경량 파인튜닝 전략을 적용했습니다.

## 1. 프로젝트 요약

- 목표: 다양한 딥페이크 데이터셋에서 일반화 가능한 탐지 모델 학습
- 백본: OpenAI CLIP `ViT-L/14`
- 핵심 아이디어: CLIP 시각 인코더의 표현력은 유지하고, 최소 파라미터만 학습해 안정적인 성능 확보
- 평가 지표: `AUC (ROC-AUC)`

## 2. 기술 스택

- Python, PyTorch, torchvision
- OpenAI CLIP
- facenet-pytorch
- scikit-learn (AUC)
- OpenCV, PIL
- Google Colab + Google Drive

## 3. 모델/학습 핵심 구현

- `ln_post` forward hook으로 CLIP visual의 `pre-proj CLS(1024)` 추출
- 학습 가능 파라미터 제한
- CLIP visual의 `LayerNorm` 파라미터만 학습
- 분류기(`Linear 1024 -> 2`) 학습
- 옵션으로 주파수(Frequency) 브랜치 결합 가능
- 손실 함수 구성
- `CrossEntropy`
- `Alignment Loss`
- `Uniformity Loss`
- 최종 손실: `CE + alpha*Alignment + beta*Uniformity`
- 스케줄러
- `Cosine Cyclic + Warmup` 스케줄 적용
- 입력 해상도 처리
- ViT-L/14 patch(14) 배수 강제 스냅
- Positional Embedding 보간으로 고해상도 입력 대응

## 4. 데이터 구성 전략

- 학습(Train)
- FF++ 페어 데이터
- KODF2 페어 데이터 (participant split 기반)
- `fake_grok_processed`를 KODF2 real과 PID 매칭해 추가 사용 가능
- 검증(Validation)
- DFDC 비디오 기반 검증 (프레임 예산/밸런싱 적용)
- sample_test 혼합 검증 (이미지 + 비디오, 비디오는 프레임 평균)

## 5. 데이터 누수 방지/샘플링 정책

- participant 단위 분할로 train/val 누수 최소화
- real-fake 프레임 인덱스 숫자 기반 정렬/매칭 (`1,10,100` 정렬 이슈 방지)
- 비디오는 균등 프레임 샘플링으로 추론 시간과 안정성 균형
- KODF2의 다양한 fake source(method/grok)를 설정값으로 제어

## 6. 증강(Augmentation) 정책

- Paired Augmentation 중심 설계
- Flip, Color Jitter, Affine, JPEG Compression
- Skin Smoothing(뷰티앱 유사) 증강
- `blur augmentation은 0으로 고정`하여 비활성화

## 7. 실행 방법 (Colab 기준)

### 7.1 환경 준비

```bash
pip install git+https://github.com/openai/CLIP.git
pip install facenet-pytorch
pip install opencv-python scikit-learn pillow
```

### 7.2 Drive 마운트

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 7.3 스크립트 실행

- Config 경로(`FFPP_ROOT`, `KODF2_ROOT`, `DFDC_ROOT`, `SAMPLE_TEST_ROOT`)를 Drive 구조에 맞게 설정
- 학습 실행 후 `BEST_CKPT_PATH`, `LAST_CKPT_PATH`에 체크포인트 저장
- `RESUME_WEIGHTS=True` 설정 시 이전 체크포인트에서 재학습 가능

## 8. 주요 설정 포인트

- Backbone: `CLIP ViT-L/14`
- Input size: `IMAGE_SIZE=384` (patch 배수 자동 보정)
- Batch: `BATCH_SIZE=16` (pair batch)
- Loss 계수: `ALPHA_ALIGN=0.1`, `BETA_UNIFORM=0.5`
- 스케줄: `CYCLE_EPOCHS=10`, `NUM_CYCLES=4`, `WARMUP_EPOCHS=1`
- 주파수 브랜치: `USE_FREQ`로 on/off 제어
- 현재 기본값: `USE_FREQ=False`

## 9. 배운점

- CLIP 기반 딥페이크 탐지 모델에서 학습 파라미터를 제한(LayerNorm + Head)해 효율적인 파인튜닝 파이프라인을 구현했습니다.
- FF++/KODF2/DFDC를 아우르는 멀티 데이터셋 학습·검증 체계를 구성하고 participant split으로 데이터 누수를 방지했습니다.
- CE+Alignment+Uniformity 복합 손실과 cosine cyclic warmup 스케줄을 적용해 표현 공간 정렬과 일반화 성능을 함께 개선했습니다.
- 이미지·비디오 혼합 검증(sample_test)과 프레임 예산 기반 추론 설계를 통해 실제 운영 시나리오에 가까운 평가 체계를 구현했습니다.
- 데이터셋별 도메인 적응(DA) 또는 TTA 전략 추가
