import ee
import os
import numpy as np
import pandas as pd
from shapely.wkt import loads
import cv2
from dotenv import load_dotenv

load_dotenv()
project_id = os.getenv('GEE_PROJECT_ID')
start_year = int(os.getenv('START_YEAR', 2019)) # 2019년부터 추천
start_month = int(os.getenv('START_MONTH', 1))
end_year = int(os.getenv('END_YEAR', 2025))
end_month = int(os.getenv('END_MONTH', 12))

# GEE 초기화
ee.Initialize(project=project_id)

def get_ndvi_sequence(file_path):
    # 1. ROI 로드
    df = pd.read_csv(file_path)
    poly = loads(df['wkt'].iloc[0])
    coords = list(poly.exterior.coords)
    if coords[0][0] < 100: coords = [[c[1], c[0]] for c in coords]
    
    roi = ee.Geometry.Polygon([coords])
    buffer_roi = roi.buffer(100).bounds()

    # 🚨 Harmonized SR 데이터셋 사용
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

    all_months_data = []

    for year in range(start_year, end_year + 1):
        for month in range(start_month, 13) if year == start_year else range(1, 13):
            if year == end_year and month > end_month:
                break
            
            start_date = ee.Date.fromYMD(year, month, 1)
            end_date = start_date.advance(1, 'month')

            # 2. 해당 월의 영상 필터링 및 구름 제거
            monthly_col = (collection.filterBounds(buffer_roi)
                           .filterDate(start_date, end_date)
                           .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))) # 구름 30% 미만만
            
            try:
                if monthly_col.size().getInfo() > 0:
                    # 🚨 Mosaic 대신 Median(중간값) 사용: 구름 제거 및 안정적인 NDVI 추출
                    composite = monthly_col.median() 
                    
                    # NDVI 계산 ((B8 - B4) / (B8 + B4))
                    ndvi_img = composite.normalizedDifference(['B8', 'B4']).rename('NDVI')
                    
                    # 데이터 샘플링 (20m 해상도)
                    pixel_data = ndvi_img.reproject(crs='EPSG:4326', scale=20).sampleRectangle(region=buffer_roi)
                    ndvi_array = np.array(pixel_data.get('NDVI').getInfo())
                    
                    all_months_data.append(ndvi_array)
                    print(f"✅ {year}-{month:02d} | 성공 (Mean NDVI: {np.mean(ndvi_array):.4f})")
                else:
                    print(f"⚠️ {year}-{month:02d} | 영상 없음 (보간 대상)")
                    all_months_data.append(None)
            except Exception as e:
                all_months_data.append(None)
                print(f"❌ {year}-{month:02d} | 에러 발생: {e}")

    # 3. 보간 및 크기 조정
    print("\n🔄 데이터 보간 및 전처리 중...")
    processed_data = []
    
    for i in range(len(all_months_data)):
        target = all_months_data[i]
        if target is None:
            prev_val = next((all_months_data[j] for j in range(i-1, -1, -1) if all_months_data[j] is not None), None)
            next_val = next((all_months_data[j] for j in range(i+1, len(all_months_data)) if all_months_data[j] is not None), None)
            
            if prev_val is not None and next_val is not None:
                target = (prev_val + next_val) / 2
            elif prev_val is not None: target = prev_val
            elif next_val is not None: target = next_val
            else: target = np.full((21, 28), 0.5)

        # ConvLSTM 크기 (28, 21)로 통일
        resized = cv2.resize(np.array(target, dtype=np.float32), (28, 21))
        
        # 🚨 데이터 정규화: NDVI(-1~1)를 학습용(0~1)으로 변환
        # (val + 1) / 2
        normalized = (resized + 1) / 2
        processed_data.append(normalized)

    final_x = np.array(processed_data)
    os.makedirs('./data/processed', exist_ok=True)
    np.save('./data/processed/X_train_harmonized.npy', final_x)
    print(f"🚀 수집 완료! 파일 저장됨: ./data/processed/X_train_harmonized.npy ({final_x.shape})")

get_ndvi_sequence('gotjawal_roi.csv')