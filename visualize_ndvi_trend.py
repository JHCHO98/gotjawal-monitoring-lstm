import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

start_year = int(os.getenv('START_YEAR', 2015))
start_month = int(os.getenv('START_MONTH', 1))
end_year = int(os.getenv('END_YEAR', 2025))
end_month = int(os.getenv('END_MONTH', 12))


def visualize_ndvi_trend(data_path):
    # 1. 데이터 로드 (84, 21, 28)
    data = np.load(data_path)
    
    # 2. 월별 평균값 계산
    # 픽셀 중 0인 값(마스킹된 부분)을 제외하고 계산하면 더 정확합니다.
    monthly_means = []
    for i in range(len(data)):
        month_data = data[i]
        # 0보다 큰 값(실제 식생)만 추출하여 평균 계산
        valid_pixels = month_data[month_data > 0]
        if len(valid_pixels) > 0:
            monthly_means.append(np.mean(valid_pixels))
        else:
            monthly_means.append(np.nan)

    # 3. 시간축 생성 (2009-01 ~ 2025-12)
    dates = pd.date_range(start=f'{start_year}-{start_month:02d}-01', periods=84, freq='MS')
    
    # 4. 그래프 그리기
    plt.figure(figsize=(15, 6))
    plt.plot(dates, monthly_means, marker='o', linestyle='-', color='#2ca02c', linewidth=2, markersize=4)
    
    # 배경에 계절 표시 (여름: 노란색, 겨울: 하늘색 등 - 선택사항)
    for i in range(start_year, end_year + 1):
        plt.axvspan(pd.Timestamp(f'{i}-06-01'), pd.Timestamp(f'{i}-08-31'), color='yellow', alpha=0.1, label='Summer' if i==start_year else "")
        plt.axvspan(pd.Timestamp(f'{i}-12-01'), pd.Timestamp(f'{i+1}-02-28' if i<end_year else f'{end_year}-{end_month:02d}-31'), color='blue', alpha=0.05, label='Winter' if i==start_year else "")

    # 그래프 꾸미기
    plt.title('2019-2025 Jeju Gotjawal Average NDVI Trend', fontsize=16, pad=20)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Average NDVI', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim(0, 1.0) # NDVI는 0~1 사이
    
    # 연도별 구분을 위한 수직선
    for year in range(2020, 2026):
        plt.axvline(pd.Timestamp(f'{year}-01-01'), color='red', linestyle=':', alpha=0.3)

    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# 실행
visualize_ndvi_trend('./data/processed/X_train_final.npy')