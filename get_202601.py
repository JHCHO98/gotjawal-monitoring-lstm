import ee
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from shapely.wkt import loads
from rasterio import features
from affine import Affine
from dotenv import load_dotenv

# 1. 초기화
load_dotenv()
PROJECT_ID = os.getenv('GEE_PROJECT_ID')
ee.Initialize(project=PROJECT_ID)

# --- 정밀 마스크 생성 함수 (학습 코드와 동일) ---
def create_precise_mask(wkt_path, h=21, w=28):
    df = pd.read_csv(wkt_path)
    poly = loads(df['wkt'].iloc[0])
    minx, miny, maxx, maxy = poly.bounds
    res_x, res_y = (maxx - minx) / w, (maxy - miny) / h
    transform = Affine.translation(minx, maxy) * Affine.scale(res_x, -res_y)
    mask = features.geometry_mask([poly], out_shape=(h, w), transform=transform, invert=True)
    return mask.astype(np.float32)

def visualize_masked_actual_jan_2026(wkt_path):
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
    masked_actual = np.where(roi_mask > 0, actual_resized, np.nan)
    
    # 5. 시각화
    plt.figure(figsize=(8, 6))
    # 배경색을 살짝 어둡게 설정하면 마스킹된 구역이 더 잘 보입니다.
    plt.gca().set_facecolor('#eeeeee') 
    
    im = plt.imshow(masked_actual, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.title("Masked Actual NDVI (January 2026)", fontsize=13, fontweight='bold')
    plt.colorbar(im, label='NDVI Value')
    
    # 그리드 제거 및 깔끔한 출력
    plt.grid(False)
    plt.show()

# 실행
visualize_masked_actual_jan_2026('gotjawal_roi.csv')