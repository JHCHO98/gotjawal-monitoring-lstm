import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import MinMaxScaler
from shapely.wkt import loads
import cv2

# 1. WKT로부터 바이너리 마스크 생성 (21x28)
def create_mask_from_wkt(wkt_path, shape=(21, 28)):
    roi_df = pd.read_csv(wkt_path)
    # 첫 번째 행의 WKT를 가져와서 다각형 객체로 변환
    polygon = loads(roi_df['wkt'].iloc[0])
    
    # GEE에서 가져온 픽셀 좌표계와 일치시킨다고 가정 (0~27, 0~20)
    # 만약 좌표가 경위도라면 픽셀 인덱스로 변환하는 과정이 필요할 수 있습니다.
    mask = np.zeros(shape, dtype=np.float32)
    
    # 다각형의 외곽선 좌표 추출
    if polygon.geom_type == 'Polygon':
        coords = np.array(polygon.exterior.coords, dtype=np.int32)
        cv2.fillPoly(mask, [coords], 1.0)
    return mask

# 2. 데이터 통합 및 채널 확장 (NDVI + 기상 5종 + 마스크 1종 = 7채널)
def prepare_masked_data(ndvi_path, weather_csv, wkt_path):
    ndvi = np.load(ndvi_path)
    weather_df = pd.read_csv(weather_csv)
    mask = create_mask_from_wkt(wkt_path)
    
    features = ['평균기온(°C)', '강수량합계(mm)', '평균습도(%)', '평균전운량(10분위)', '합계일조시간(hr)']
    scaler = MinMaxScaler()
    weather_scaled = scaler.fit_transform(weather_df[features])
    
    # (84, 21, 28, 7) 데이터 생성
    combined = np.zeros((84, 21, 28, 7), dtype='float32')
    for i in range(84):
        combined[i, :, :, 0] = ndvi[i] * mask # NDVI에 마스크 적용
        for j in range(5):
            combined[i, :, :, j+1] = weather_scaled[i, j] * mask # 기상 데이터도 영역 내만 표시
        combined[i, :, :, 6] = mask # 7번째 채널로 마스크 정보 전달
        
    return combined, mask

# 3. 모델 정의 (CUDA 최적화 + 7채널 입력)
def build_masked_convlstm(input_shape):
    # Mixed Precision 활성화
    from tensorflow.keras import mixed_precision
    mixed_precision.set_global_policy('mixed_float16')

    inputs = layers.Input(shape=input_shape)
    x = layers.ConvLSTM2D(
        filters=16, kernel_size=(3, 3), padding='same',
        return_sequences=False, kernel_regularizer=regularizers.l2(1e-4)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # 0~1 사이 NDVI 출력을 위한 Sigmoid
    outputs = layers.Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid', padding='same', dtype='float32')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mse', metrics=['mae'])
    return model

# --- 실행 프로세스 ---

# 데이터 준비
combined_data, roi_mask = prepare_masked_data('./data/processed/X_train_final.npy', 'data/weather_data.csv', 'gotjawal_roi.csv')

# 시퀀스 생성 (과거 12개월 -> 다음 1개월)
def create_sequences(data, seq_length=12):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, :, :, 0:1])
    return np.array(X, dtype='float32'), np.array(y, dtype='float32')

X_all, y_all = create_sequences(combined_data)
X_train, X_val = X_all[:-12], X_all[-12:]
y_train, y_val = y_all[:-12], y_all[-12:]

# 모델 빌드 및 학습
model = build_masked_convlstm((12, 21, 28, 7))
early_stop = tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150, batch_size=4,
    callbacks=[early_stop]
)

# 4. 결과 저장 및 시각화
model.save('./models/gotjawal_masked_model.h5')

def plot_masked_results(model, X_val, y_val, mask):
    preds = model.predict(X_val) * mask[np.newaxis, ..., np.newaxis] # 예측값도 마스킹
    
    plt.figure(figsize=(12, 6))
    for i in range(2):
        plt.subplot(2, 2, i+1); plt.imshow(y_val[i*6].squeeze(), cmap='RdYlGn', vmin=0, vmax=1); plt.title("Actual")
        plt.subplot(2, 2, i+3); plt.imshow(preds[i*6].squeeze(), cmap='RdYlGn', vmin=0, vmax=1); plt.title("Predicted")
    plt.tight_layout()
    plt.savefig('./plots/masked_prediction_result.png')
    plt.show()

plot_masked_results(model, X_val, y_val, roi_mask)