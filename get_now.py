import ee
import os
import pandas as pd
from dotenv import load_dotenv

# GEE 초기화
load_dotenv()
PROJECT_ID = os.getenv('GEE_PROJECT_ID')

try:
    ee.Initialize(project=PROJECT_ID)
except Exception as e:
    print(f"❌ GEE 초기화 실패: {e}")

def dms_to_dd(d, m, s):
    return d + (m / 60.0) + (s / 3600.0)

def get_verified_ndvi(locations, date_start='2026-01-01'):
    results = []
    
    for i, (lat_dms, lon_dms) in enumerate(locations):
        lat = dms_to_dd(*lat_dms)
        lon = dms_to_dd(*lon_dms)
        point = ee.Geometry.Point([lon, lat])
        
        # Sentinel-2 영상 검색
        s2_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                         .filterBounds(point)
                         .filterDate(date_start, '2026-12-31')
                         # 처음엔 구름 제한을 풀고 가져온 뒤, 개별 영상의 전운량을 체크합니다.
                         .sort('system:time_start', False))

        latest_image = s2_collection.first()
        
        if latest_image.getInfo() is None:
            results.append({"ID": i+1, "NDVI": "N/A", "전운량(%)": "N/A", "상태": "영상 없음"})
            continue

        # 해당 영상 전체의 전운량 메타데이터 가져오기
        cloud_pixel_percentage = latest_image.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
        
        # NDVI 계산
        ndvi = latest_image.normalizedDifference(['B8', 'B4'])
        val = ndvi.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10
        ).get('nd').getInfo()
        
        date = ee.Date(latest_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
        
        # 전운량에 따른 데이터 신뢰도 판단
        if cloud_pixel_percentage < 5:
            status = "✅ 매우 맑음 (신뢰도 높음)"
        elif cloud_pixel_percentage < 20:
            status = "⚠️ 약간 흐림 (참고용)"
        else:
            status = "❌ 구름 많음 (데이터 왜곡 위험)"

        results.append({
            "ID": i+1,
            "변환_위도": round(lat, 6),
            "변환_경도": round(lon, 6),
            "NDVI": round(val, 4) if val is not None else "N/A",
            "전운량(%)": round(cloud_pixel_percentage, 2),
            "상태": status,
            "촬영날짜": date
        })
        
    return pd.DataFrame(results)

field_locations = [
    ((33, 19, 8.6016), (126, 16, 6.01789)), # 1
    ((33, 19, 6.8196), (126, 16, 4.79784)), # 2
    ((33, 19, 14.08188), (126, 15, 56.25648)), # 3
    ((33, 19, 19.83684), (126, 15, 48.93480)), # 4
    ((33, 19, 21.28080), (126, 15, 48.54780)), # 5
    ((33, 19, 23.10564), (126, 15, 49.22748))  # 6
]

# 실행 및 결과 출력
df_results = get_verified_ndvi(field_locations)

print("\n🌿 현장 지점별 최신 NDVI 분석 결과")
print("=" * 80)
print(df_results.to_string(index=False))
print("=" * 80)

# 엑셀이나 CSV로 저장하고 싶다면:
# df_results.to_csv('field_check_results.csv', index=False, encoding='utf-8-sig')