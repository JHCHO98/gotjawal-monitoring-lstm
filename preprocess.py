import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

start_year = int(os.getenv('START_YEAR', 2015))
start_month = int(os.getenv('START_MONTH', 1))
end_year = int(os.getenv('END_YEAR', 2025))
end_month = int(os.getenv('END_MONTH', 12))

def preprocess_advanced(data_path, save_path):
    # 1. 데이터 로드 (84, 21, 28)
    data = np.load(data_path)
    num_months = data.shape[0]
    
    # --- 단계 1: 최댓값 기반 필터링 (Top 10% 픽셀 무시, 하위 노이즈 제거) ---
    # 평균 대신 각 달의 상위 90%~95% 지점의 값을 기준으로 스케일링을 보정합니다.
    refined_data = data.copy()
    
    for i in range(num_months):
        month_slice = data[i]
        valid_pixels = month_slice[month_slice > 0]
        
        if len(valid_pixels) > 0:
            # 너무 낮은 값(그림자/안개)은 해당 달의 평균으로 하한선 설정
            lower_bound = np.percentile(valid_pixels, 20) 
            refined_data[i] = np.where(refined_data[i] < lower_bound, lower_bound, refined_data[i])

    # --- 단계 2: 시간축 이상치 제거 (Smoothing & Outlier Removal) ---
    # 2025년 초처럼 갑자기 툭 떨어진 값은 주변 달의 평균으로 교체합니다.
    temporal_means = np.mean(refined_data, axis=(1, 2))
    df_temp = pd.DataFrame({'ndvi': temporal_means})
    
    # 이동 평균(Rolling Mean)을 사용하여 너무 튀는 값 찾기
    df_temp['smooth'] = df_temp['ndvi'].rolling(window=5, center=True, min_periods=1).median()
    diff = np.abs(df_temp['ndvi'] - df_temp['smooth'])
    
    # 차이가 큰 지점(이상치) 인덱스 추출
    outlier_idx = diff > 0.15 # 0.15 이상 튀면 이상치로 판단
    
    for idx in np.where(outlier_idx)[0]:
        print(f"⚠️ Index {idx} ({start_year+idx//12}년 {idx%12+1}월) 이상치 감지 -> 보정 실시")
        # 주변 2개월의 평균 이미지로 대체
        start = max(0, idx-1)
        end = min(num_months, idx+2)
        refined_data[idx] = np.mean(refined_data[start:end], axis=0)

    # --- 단계 3: 연도별/월별 상대 정규화 (Normalization) ---
    # 2022년 이후 Baseline 변경으로 낮아진 수치를 전체 기간의 평균 수준으로 끌어올립니다.
    final_data = refined_data.copy()
    global_max = 0.8  # 곶자왈 최성기 목표 NDVI
    
    for i in range(num_months):
        current_max = np.max(final_data[i])
        if current_max < 0.5: # 너무 낮게 측정된 달은 강제로 타겟 농도에 맞춰 스케일업
            scale_factor = 0.6 / (current_max + 1e-6)
            final_data[i] *= scale_factor
            
    # 최종 데이터 범위를 0~1로 클리핑
    final_data = np.clip(final_data, 0, 1)
    
    np.save(save_path, final_data)
    print(f"🚀 전처리 완료! 저장 경로: {save_path}")



# 실행
preprocess_advanced('./data/X_train2.npy', './data/processed/X_train_final.npy')
