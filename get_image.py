import ee
import os
import numpy as np
import pandas as pd
from shapely.wkt import loads
from dotenv import load_dotenv

# 1. 환경변수 로드 및 GEE 초기화
load_dotenv()
PROJECT_ID = os.getenv('GEE_PROJECT_ID')

ee.Authenticate()
ee.Initialize(project=PROJECT_ID)

# 2. ROI 로드 함수
def load_roi_from_csv(file_path):
    df = pd.read_csv(file_path)
    wkt_str = df['wkt'].iloc[0]
    poly = loads(wkt_str)
    if poly.geom_type == 'Polygon':
        coords = [list(poly.exterior.coords)]
    else:
        coords = [list(p.exterior.coords) for p in poly.geoms]
    return ee.Geometry.Polygon(coords)

# 3. NDVI 추출 및 배열 변환 함수
def get_monthly_ndvi_array(roi, start_year, end_year):
    # 구름 필터를 50%로 완화
    collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterBounds(roi) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 50))
    
    all_months_data = []
    
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            if year == 2026: break # 현재 날짜 기준
            
            start_date = ee.Date.fromYMD(year, month, 1)
            end_date = start_date.advance(1, 'month')
            
            # Median 대신 해당 월의 가장 구름 적은 영상 한 장만 써보기
            img = collection.filterDate(start_date, end_date).sort('CLOUDY_PIXEL_PERCENTAGE').first()
            
            # 데이터가 존재하는지 확인
            if img.getInfo() is None:
                all_months_data.append(None)
                print(f"⚠️ {year}-{month:02d} | 해당 기간에 위성 영상 자체가 없음")
                continue

            ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI').clip(roi)
            
            try:
                # scale=10(미터) 명시, defaultValue 설정
                pixel_data = ndvi.sampleRectangle(region=roi, defaultValue=0)
                ndvi_array = np.array(pixel_data.get('NDVI').getInfo())
                all_months_data.append(ndvi_array)
                print(f"✅ {year}-{month:02d} 수집 성공! (Shape: {ndvi_array.shape})")
            except Exception as e:
                all_months_data.append(None)
                print(f"❌ {year}-{month:02d} 변환 실패: {e}")
                
    return all_months_data

# 4. 실행 및 저장
roi = load_roi_from_csv('my_gotjawal_roi.csv')
print(f"프로젝트 [{PROJECT_ID}]에서 데이터 수집을 시작합니다...")

ndvi_sequences = get_monthly_ndvi_array(roi, 2019, 2025)

# 데이터 저장 폴더 생성
output_dir = './data/processed'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(roi.centroid().coordinates().getInfo())
# 리스트를 넘파이 오브젝트 배열로 저장 (크기가 제각각일 수 있으므로)
save_path = os.path.join(output_dir, 'gotjawal_ndvi_sequence.npy')
np.save(save_path, np.array(ndvi_sequences, dtype=object))

print(f"\n🚀 모든 작업 완료! 데이터가 '{save_path}'에 저장되었습니다.")