# 🌲 Gotjawal-Monitoring-LSTM

> **ConvLSTM을 활용한 제주 환상숲 곶자왈 식생 밀도 변화 예측 및 시공간 시계열 분석**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![GEE](https://img.shields.io/badge/Google%20Earth%20Engine-Sentinel--2-34A853?style=flat-square&logo=google&logoColor=white)](https://earthengine.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## 📌 연구 개요

제주도 곶자왈은 '제주의 허파'라 불리는 생태적 요충지로, 독특한 화산 지형 위에 형성된 보존 가치가 매우 높은 숲입니다. 그러나 면적이 넓어 전 구역을 직접 조사하기에는 물리적·시간적 한계가 따릅니다.

본 연구는 **Google Earth Engine(GEE) 위성 데이터**와 **기상청 API 기상 데이터**를 결합하고, **ConvLSTM 딥러닝 모델**을 통해 곶자왈의 식생 밀도(NDVI) 변화를 **시공간적으로 예측**합니다. 단순 수치 예측을 넘어 공간적 패턴과 시간적 흐름을 동시에 학습함으로써, 비정상적 식생 감소를 조기에 감지하고 선제적 보전 대응을 가능하게 합니다.

---

## 🔍 핵심 연구 질문

| # | 질문 | 접근 방식 |
|---|------|-----------|
| 1 | **공간적 변화** — 곶자왈 경계면은 개발·자연 천이에 의해 어떻게 변화하는가? | ConvLSTM 기반 공간 예측 + 에러 맵 분석 |
| 2 | **시계열 예측** — 과거 6년(2019~2025) 데이터로 미래 NDVI 변화를 예측할 수 있는가? | Many-to-One LSTM 구조, 12개월 입력 → 다음 달 예측 |
| 3 | **현장 검증** — 위성 기반 NDVI와 실제 현장 식생 구조는 얼마나 일치하는가? | 2026.05.07 현장 조사 및 GPS 태깅 사진 대조 |

---

## 🛠 기술 스택

| 분야 | 기술 |
|------|------|
| **언어** | Python 3.10+ |
| **딥러닝** | PyTorch (CUDA 12.1 가속 지원) |
| **위성 데이터** | Google Earth Engine — `COPERNICUS/S2_SR_HARMONIZED` (Sentinel-2) |
| **기상 데이터** | 기상청 ASOS API (관측소 stn_id: 184, 제주 지점) |
| **GIS / 전처리** | `rasterio`, `affine`, `shapely`, `geopandas`, `opencv-python` |
| **분석 / 시각화** | `numpy`, `pandas`, `matplotlib` |

---

## 📂 프로젝트 구조

```
mathtrip_research/
│
├── 📄 .env                      # 환경변수 (GEE Project ID, API Key 등)
│
├── 📊 데이터 수집 · 처리
│   ├── get_image.py             # GEE에서 월별 NDVI 이미지 수집
│   ├── get_weather_data.py      # 기상청 API로 기온·강수·습도 등 5종 수집
│   ├── get_now.py               # 최신(당월) 위성 데이터 수집
│   ├── get_202601.py            # 2026년 1월 실측 데이터 수집
│   ├── discernBoundary.py       # 곶자왈 경계 추출 및 정제
│   ├── GeeConnectionTest.py     # GEE 연결 상태 테스트
│   └── check_data.py            # 수집 데이터 무결성 검사
│
├── 🧹 전처리 · 입력 생성
│   ├── preprocess.py            # 노이즈 제거, 이상치 보정, 정규화
│   └── make_input_data.py       # ConvLSTM 입력 시퀀스 (X, y) 생성
│
├── 🤖 모델 학습 · 예측
│   ├── model.py                 # GotjawalConvLSTM 아키텍처 정의
│   ├── train.py                 # 학습 루프, Early Stopping, 결과 저장
│   └── make_error_map.py        # 예측값 vs 실측값 오차 행렬 생성
│
├── 📈 시각화
│   └── visualize_ndvi_trend.py  # NDVI 시계열 트렌드 시각화
│
├── 📁 data/                     # 월별 NDVI 배열, 기상 데이터 (*.npy, *.csv)
├── 📁 models/                   # 학습된 모델 가중치 (*.pth)
├── 📁 plots/                    # 학습 곡선, 예측 맵, 에러 맵
├── 📁 selected_coords/          # 현장 검증 좌표
│
├── gotjawal_roi.csv             # 곶자왈 ROI 다각형 (WKT 형식)
├── gotjawal_ndvi_timeseries.csv # 월별 평균 NDVI 시계열 기록
└── requirements.txt
```

---

## 🔬 연구 방법론

### 1. 데이터 수집 (`get_image.py`, `get_weather_data.py`)

- **기간**: 2019년 1월 ~ 2025년 12월 (월 단위, 총 84개월)
- **위성**: Sentinel-2 SR Harmonized — 구름 비율이 가장 낮은 데이터 선별
- **NDVI 계산**: `(B8 - B4) / (B8 + B4)`
- **기상 변수**: 기온, 강수량, 습도, 전운량, 일조시간 (5종)

### 2. 전처리 (`preprocess.py`)

1. **하한선 클리핑** — 각 월의 하위 20% 픽셀(그림자·안개 노이즈) 제거
2. **시간축 이상치 보정** — 5개월 Rolling Median 대비 0.15 초과 이탈값을 주변 평균으로 대체
3. **정규화** — 전체 값을 0~1 범위로 스케일링 (MinMax)
4. **결측치 처리** — 7~8개월분 누락 데이터 선형 보간(Linear Interpolation)

### 3. 모델 아키텍처 (`model.py`)

```
입력: (Batch, 12, 7, 21, 28)  ← 12개월 × 7채널(NDVI+기상5종+마스크) × 21×28픽셀
  │
  ▼
ConvLSTMCell (kernel_size=3, hidden_channels=16)
  ├─ 시간 흐름: LSTM 게이트 (Input / Forget / Output / Gate)
  └─ 공간 패턴: Conv2D (3×3 커널)
  │
  ▼  [12 스텝 순환]
  │
Conv2D (1×1) → Sigmoid
  │
  ▼
출력: (Batch, 1, 21, 28)  ← 예측 NDVI 맵 (다음 달)
```

- **구조**: Many-to-One (이전 12개월 입력 → 다음 달 NDVI 예측)
- **손실 함수**: MSE Loss
- **옵티마이저**: Adam (lr=0.0005)
- **정규화**: L2 Weight Decay, Batch Normalization, Dropout 30%
- **과적합 방지**: Early Stopping (patience=10)

### 4. 에러 맵 분석 (`make_error_map.py`)

학습된 모델의 예측값(2026년 1월)과 GEE에서 수집한 실측값을 비교하여 픽셀별 오차를 시각화합니다.

- **Red** (양수 오차): 모델이 실제보다 높게 예측한 영역 (과대 추정)
- **Blue** (음수 오차): 모델이 실제보다 낮게 예측한 영역 (과소 추정)
- **RMSE**가 결과 맵에 자동 표기됩니다.

---

## 🚀 시작하기

### 1. 환경 설정

```bash
git clone https://github.com/JHCHO98/gotjawal-monitoring-lstm.git
cd gotjawal-monitoring-lstm
pip install -r requirements.txt
```

### 2. 환경 변수 설정 (`.env`)

```env
GEE_PROJECT_ID=your_project_id_here
WEATHER_API_KEY=your_api_key_here
START_YEAR=2019
START_MONTH=1
END_YEAR=2025
END_MONTH=12
stn_id=184
```

> GEE Project ID는 [Google Earth Engine](https://earthengine.google.com/)에서 프로젝트를 생성 후 확인할 수 있습니다.

### 3. 실행 순서

```bash
# Step 1: GEE 연결 확인
python GeeConnectionTest.py

# Step 2: 위성 데이터 수집 (월별 NDVI)
python get_image.py

# Step 3: 기상 데이터 수집
python get_weather_data.py

# Step 4: 전처리
python preprocess.py

# Step 5: 모델 학습
python train.py

# Step 6: 에러 맵 생성 (2026년 1월 예측 vs 실측 비교)
python make_error_map.py

# Step 7: NDVI 트렌드 시각화
python visualize_ndvi_trend.py
```

---

## 📊 주요 결과

### 학습 성능
- MSE 손실이 낮은 수치로 안정적으로 수렴하며 양호한 학습 상태 확인
- Early Stopping으로 과적합 없이 최적 가중치 저장

### 2026년 1월 에러 맵 분석
- **곶자왈 중심부**: 예측값과 실측값이 대체로 일치 (낮은 오차)
- **경계부**: NDVI를 실제보다 낮게 예측하는 경향(과소 추정) 발견 → 경계 전이 지역의 식생 패턴 학습 어려움으로 판단

### 현장 검증 (2026.05.07)
- **측정 위치**: 제주도 환상숲 곶자왈 및 주요 경계 지점
- **방법**: GPS 태깅 사진 촬영 및 픽셀값-현장 식생 밀도 대조
- **주요 발견**: 육안으로 식생이 풍부해 보이는 지점에서도 NDVI가 **0.0494**로 기록된 사례 발견 → 위성 데이터와 실제 식생량 간 불일치 확인

---

## 💡 결론 및 한계점

### 성과
- ConvLSTM을 통해 곶자왈 식생 변화의 시공간적 패턴을 성공적으로 학습
- 기상 요인(기온·강수·습도 등)과 NDVI 상관관계 분석 기반 마련
- 에러 맵을 통한 예측 신뢰도 구역별 정량 평가 시스템 구축

### 한계 및 향후 과제
| 한계 | 향후 방향 |
|------|-----------|
| Sentinel-2 해상도(10m×10m)의 미세 식생 탐지 한계 | 고해상도 위성 데이터(예: Planet, WorldView) 활용 탐색 |
| 위성 데이터와 육안 관찰 사이의 불일치 원인 불명 | 수관 밀폐도, 임상(林床) 반사율 등 추가 변수 분석 |
| 단일 ConvLSTM 레이어 구조 | Stacked ConvLSTM, Attention 메커니즘 도입 |
| 단기 예측(1개월) | 계절 단위(3~6개월) 중장기 예측 모델 확장 |

---

## 📝 라이선스

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

**경기북과학고등학교 2학년 수학여행 현장연구 프로젝트**  
*2026.05.07 — 환상숲 곶자왈 현장 조사 완료*

</div>
