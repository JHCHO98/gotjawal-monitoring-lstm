import torch
import numpy as np
import pandas as pd
from shapely.wkt import loads
from rasterio import features
from affine import Affine
from sklearn.preprocessing import MinMaxScaler

def create_precise_mask(wkt_path, h=21, w=28):
    df = pd.read_csv(wkt_path)
    poly = loads(df['wkt'].iloc[0])
    minx, miny, maxx, maxy = poly.bounds
    res_x, res_y = (maxx - minx) / w, (maxy - miny) / h
    transform = Affine.translation(minx, maxy) * Affine.scale(res_x, -res_y)
    mask = features.geometry_mask([poly], out_shape=(h, w), transform=transform, invert=True)
    return mask.astype(np.float32)

def load_data():
    ndvi = np.load('./data/processed/X_train_final.npy')  # (84, 21, 28)
    weather_df = pd.read_csv('data/weather_data.csv')
    mask = create_precise_mask('gotjawal_roi.csv')

    scaler = MinMaxScaler()
    weather = scaler.fit_transform(weather_df[['평균기온(°C)', '강수량합계(mm)', '평균습도(%)', '평균전운량(10분위)', '합계일조시간(hr)']])

    # 토치 차원: (Time, Channel, H, W) -> (84, 7, 21, 28)
    combined = np.zeros((84, 7, 21, 28), dtype=np.float32)
    for i in range(84):
        combined[i, 0, :, :] = ndvi[i] * mask
        for j in range(5):
            combined[i, j+1, :, :] = weather[i, j] * mask
        combined[i, 6, :, :] = mask  # 마스크 채널
    return combined, mask

# 시퀀스 생성 (12개월 -> 1개월 예측)
def create_sequences(data, seq_len=12):
    X_list, y_list = [], []
    for i in range(len(data) - seq_len):
        X_list.append(data[i:i+seq_len])          # (12, 7, 21, 28)
        y_list.append(data[i+seq_len, 0:1, :, :]) # (1, 21, 28)
    return torch.tensor(np.array(X_list)), torch.tensor(np.array(y_list))
