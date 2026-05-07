import ee
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import GotjawalConvLSTM
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from shapely.wkt import loads
from dotenv import load_dotenv
from make_input_data import create_precise_mask,load_data,create_sequences
# 1. 초기화
load_dotenv()
PROJECT_ID = os.getenv('GEE_PROJECT_ID')
try:
    ee.Initialize(project=PROJECT_ID)
except:
    ee.Authenticate()
    ee.Initialize(project=PROJECT_ID)


def get_resized_actual_jan_2026(wkt_path):
    # 2. ROI 및 Mask 준비
    df = pd.read_csv(wkt_path)
    poly = loads(df['wkt'].iloc[0])
    coords = list(poly.exterior.coords)
    if coords[0][0] < 100: coords = [[c[1], c[0]] for c in coords]
    roi_geometry = ee.Geometry.Polygon([coords])
    buffer_roi = roi_geometry.buffer(100).bounds()
    
    # 21x28 마스크 생성
    roi_mask = create_precise_mask(wkt_path, h=21, w=28)

    # 3. 2026년 1월 실측 데이터 수집 (SR)
    collection = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
                  .filterBounds(buffer_roi)
                  .filterDate('2026-01-01', '2026-02-01')
                  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))

    composite = collection.median()
    ndvi_img = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # 데이터 추출 및 21x28 리사이징
    pixel_data = ndvi_img.reproject(crs='EPSG:4326', scale=20).sampleRectangle(region=buffer_roi)
    actual_ndvi = np.array(pixel_data.get('NDVI').getInfo())
    actual_resized = cv2.resize(actual_ndvi.astype(np.float32), (28, 21))
    
    # 4. 🚨 ROI 마스킹 적용 (마스크 밖은 NaN 처리하여 투명하게 만듦)
    return actual_resized
# 실행
actual_resized=get_resized_actual_jan_2026('gotjawal_roi.csv')

data, roi_mask = load_data()
X_tensor, y_tensor = create_sequences(data)

# 시계열 분리
X_train, X_val = X_tensor[:-12], X_tensor[-12:]
y_train, y_val = y_tensor[:-12], y_tensor[-12:]

# 2. 모델 객체 생성 및 가중치 로드
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GotjawalConvLSTM().to(device)

# 가중치 파일 경로 확인 (.pth)
model.load_state_dict(torch.load("./models/gotjawal_torch_model.pth", map_location=device))
model.eval()

# 3. 예측 수행 (2026년 1월)
with torch.no_grad():
    # X_val[-1:] 데이터의 차원 확인: (1, 12, 7, 21, 28) 형태여야 함
    test_input = X_val[-1:].to(device) 
    
    # 모델 통과 (출력: 0~1 사이의 Sigmoid 값)
    prediction = model(test_input)
    
    # 넘파이 변환 및 차원 축소 (21, 28)
    pred_np = prediction.cpu().numpy().squeeze()
    
    # NDVI 값 복원 (0~1 -> -1~1)
    pred_restored = (pred_np * 2) - 1

# 4. 🚨 오차 행렬(Error Map) 시각화
# actual_resized: GEE에서 가져온 2026년 1월 실측 행렬
error_map = pred_restored - actual_resized

plt.figure(figsize=(10, 7))
# 배경 마스킹 처리
masked_error = np.where(roi_mask > 0, error_map, np.nan)

# 오차 시각화 (RdBu: Red(과대예측), Blue(과소예측))
im = plt.imshow(masked_error, cmap='RdBu', vmin=-0.5, vmax=0.5)
plt.colorbar(im, label='NDVI Difference (Pred - Actual)')
plt.title("2026-01 NDVI Prediction Error Map (Torch Model)", fontsize=14)

# 성능 지표 추가
rmse = np.sqrt(np.nanmean(error_map**2))
plt.annotate(f'RMSE: {rmse:.4f}', xy=(0.05, 0.05), xycoords='axes fraction', 
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.8))
plt.savefig('./plots/error_map.png', dpi=150, bbox_inches='tight')
plt.show()