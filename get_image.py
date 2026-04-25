import ee
import os
import numpy as np
import pandas as pd
from shapely.wkt import loads
import cv2
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경 변수 로드
project_id = os.getenv('GEE_PROJECT_ID')

# GEE 초기화
ee.Initialize(project=project_id)

def get_greenest_ndvi_sequence(file_path):
    # 1. ROI 로드 및 버퍼 설정
    df = pd.read_csv(file_path)
    poly = loads(df['wkt'].iloc[0])
    coords = list(poly.exterior.coords)
    if coords[0][0] < 100: coords = [[c[1], c[0]] for c in coords]
    
    roi = ee.Geometry.Polygon([coords])
    # 🚨 픽셀 유실 방지를 위해 100m 버퍼 추가
    buffer_roi = roi.buffer(100).bounds()

    collection = ee.ImageCollection("COPERNICUS/S2")

    def addNDVI(img):
        return img.addBands(img.normalizedDifference(['B8', 'B4']).rename('NDVI'))

    all_months_data = []

    for year in range(2019, 2026):
        for month in range(1, 13):
            if year == 2026: break
            start_date = ee.Date.fromYMD(year, month, 1)
            end_date = start_date.advance(1, 'month')

            # 🚨 Quality Mosaic: 가장 초록색인 픽셀만 합성
            monthly_col = collection.filterBounds(buffer_roi).filterDate(start_date, end_date).map(addNDVI)
            
            try:
                if monthly_col.size().getInfo() > 0:
                    # NDVI가 최대인 픽셀들로 모자이크 생성
                    greenest_img = monthly_col.qualityMosaic('NDVI').select('NDVI')
                    # 🚨 해상도를 20m로 조절하여 안정성 확보
                    pixel_data = greenest_img.reproject(crs='EPSG:4326', scale=20).sampleRectangle(region=buffer_roi)
                    
                    ndvi_array = np.array(pixel_data.get('NDVI').getInfo())
                    
                    # 모든 데이터가 0이거나 음수면 실패로 간주
                    if np.max(ndvi_array) <= 0: raise ValueError
                    
                    all_months_data.append(ndvi_array)
                    print(f"✅ {year}-{month:02d} | 성공 (Max: {np.max(ndvi_array):.4f})")
                else:
                    raise ValueError
            except:
                all_months_data.append(None)
                print(f"❌ {year}-{month:02d} | 데이터 없음")

    # --- 보간 및 크기 조정 ---
    print("\n🔄 데이터 보간 및 전처리를 시작합니다...")
    processed_data = []
    
    # 임시로 None을 앞뒤 값으로 채우는 로직
    for i in range(len(all_months_data)):
        target = all_months_data[i]
        if target is None:
            # 주변 데이터 탐색
            prev_val = next((all_months_data[j] for j in range(i-1, -1, -1) if all_months_data[j] is not None), None)
            next_val = next((all_months_data[j] for j in range(i+1, len(all_months_data)) if all_months_data[j] is not None), None)
            
            if prev_val is not None and next_val is not None:
                target = (prev_val + next_val) / 2
            elif prev_val is not None: target = prev_val
            elif next_val is not None: target = next_val
            else: target = np.full((21, 28), 0.5) # 최후의 수단: 중간값

        # 🚨 ConvLSTM 입력을 위해 크기를 (21, 28)로 통일
        resized = cv2.resize(np.array(target), (28, 21))
        processed_data.append(resized)

    final_x = np.array(processed_data)
    np.save('./data/processed/X_train.npy', final_x)
    print("🚀 수집 및 저장 완료!")

get_greenest_ndvi_sequence('my_gotjawal_roi.csv')