import pandas as pd
import os
import ee
from dotenv import load_dotenv

# 1. .env 파일의 환경변수를 읽어옴
load_dotenv()

# 2. 환경변수에서 프로젝트 ID 가져오기
project_id = os.getenv('GEE_PROJECT_ID')

# 1. GEE 인증 및 초기화
try:
    ee.Initialize(project=project_id)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=project_id)

# 2. 분석 설정
# 좌표: 환상숲 곶자왈 공원 중심부
target_point = ee.Geometry.Point([126.2743, 33.3210])
# 주변 500m 반경 설정 (곶자왈 면적에 따라 조절 가능)
roi = target_point.buffer(500)

# 분석 기간 설정
start_year = 2019
end_year = 2025

# 3. 데이터 컬렉션 및 전처리 함수
def get_ndvi_collection():
    # 최신 Harmonized 컬렉션 사용
    s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(roi) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
    
    def add_ndvi(image):
        ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return image.addBands(ndvi)
    
    return s2.map(add_ndvi)

collection = get_ndvi_collection()

# 4. 월별 데이터 추출 로직
print("데이터 추출 중... 잠시만 기다려주세요.")
monthly_data = []

for year in range(start_year, end_year + 1):
    for month in range(1, 13):
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, 'month')
        
        # 1. 해당 월 이미지 필터링
        monthly_col = collection.filterDate(start_date, end_date)
        
        # 2. 이미지가 존재하는지 확인 (핵심!)
        if monthly_col.size().getInfo() > 0:
            monthly_img = monthly_col.median()
            
            # NDVI 밴드가 실제로 생성되었는지 한 번 더 체크
            band_names = monthly_img.bandNames().getInfo()
            if 'NDVI' in band_names:
                stats = monthly_img.select('NDVI').reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=roi,
                    scale=10,
                    maxPixels=1e9
                ).getInfo()
                
                ndvi_val = stats.get('NDVI')
                if ndvi_val is not None:
                    monthly_data.append({
                        'date': f"{year}-{month:02d}",
                        'ndvi': ndvi_val
                    })
            else:
                print(f"Skipping {year}-{month:02d}: No NDVI band generated.")
        else:
            print(f"Skipping {year}-{month:02d}: No cloud-free images found.")
            
# 5. Pandas 데이터프레임 변환 및 파일 저장
df = pd.DataFrame(monthly_data)

# 파일명 설정
file_name = 'gotjawal_ndvi_timeseries.csv'
df.to_csv(file_name, index=False, encoding='utf-8-sig')

print(f"저장 완료: {os.path.abspath(file_name)}")
print(df.head()) # 상위 5개 데이터 확인