# 🌲 Gotjawal-monitoring-LSTM
**ConvLSTM을 활용한 제주 곶자왈 식생 밀도 변화 예측 및 시공간 시계열 분석**

## 📌 Project Overview

본 프로젝트는 '제주의 허파'라 불리는 곶자왈의 생태적 가치를 보존하기 위해, 위성 영상 데이터와 딥러닝 기술을 결합하여 식생 변화를 예측합니다. 단순 수치 예측을 넘어 ConvLSTM(Convolutional LSTM) 모델을 통해 공간적 패턴과 시간적 흐름을 동시에 학습하여 곶자왈의 식생 밀도(NDVI) 변화를 시뮬레이션합니다.

## 🔍 Key Research Questions

* **공간적 변화**: 곶자왈의 경계면은 인위적 개발이나 자연적 천이에 의해 어떻게 확장 또는 축소되는가?

* **시계열 예측**: 과거 6년(2019~2025)의 데이터를 바탕으로 향후 식생 밀도 변화를 예측할 수 있는가?

* **현장 검증**: 위성 데이터 기반의 NDVI 수치가 실제 곶자왈 현장의 식생 구조와 얼마나 일치하는가?

## 🛠 Tech Stack
* **Language**: Python 3.10+
* **Deep Learning**: PyTorch (CUDA 12.1 가속 지원)
* **Data Source**: Google Earth Engine (GEE), Sentinel-2 Satellite Imagery
* **Weather Data**: 기상청 ASOS (stn_id: 184 - 제주 지점)
* **GIS/Preprocessing**: `rasterio`, `affine`, `shapely`, `geopandas`
* **Analysis**: `numpy`, `pandas`, `matplotlib`

## 🚀 Getting Started

### 1. Environment Variables
본 프로젝트는 보안을 위해 GEE Project ID를 환경변수로 관리합니다. .env 파일을 생성하세요.

```env
GEE_PROJECT_ID=gotjawal-monitoring-lstm
WEATHER_API_KEY=your_api_key_here
START_YEAR=2019
START_MONTH=1
END_YEAR=2025
END_MONTH=12
stn_id=184
```

### 2. Data Collection
Google Earth Engine을 활용하여 곶자왈 다각형 영역의 Sentinel-2 영상을 수집합니다. 구름 노이즈 제거 및 누락된 데이터(7~8개월분)에 대해서는 Linear Interpolation을 적용하였습니다.

### 3. Model: ConvLSTM
식생의 '성장'과 '이동'을 학습하기 위해 Convolutional Layer가 결합된 LSTM을 사용합니다.

Input: 12-month NDVI sequence images

Output: Predicted NDVI image for the next month

## 📍 Field Verification (Ground Truth)

Location: 제주도 환상숲 곶자왈 및 주요 경계 지점

Methods: GPS 태깅 사진 촬영 및 픽셀 값-현장 밀도 대조 분석

## 📝 License

This project is licensed under the MIT License.
