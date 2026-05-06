import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import MinMaxScaler

# 1. GPU 장치 확인 및 메모리 성장 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 프로그램이 필요한 만큼만 GPU 메모리를 점유하도록 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ CUDA GPU 사용 가능: {len(gpus)}개 장치 발견")
        
        # 혼합 정밀도 학습 활성화 (학습 속도 대폭 향상)
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ GPU를 찾을 수 없습니다. CPU로 진행합니다.")

# 2. 데이터 로드 및 전처리 함수
def prepare_data(ndvi_path, weather_csv):
    ndvi = np.load(ndvi_path)
    weather_df = pd.read_csv(weather_csv)
    features = ['평균기온(°C)', '강수량합계(mm)', '평균습도(%)', '평균전운량(10분위)', '합계일조시간(hr)']
    
    scaler = MinMaxScaler()
    weather_scaled = scaler.fit_transform(weather_df[features])
    
    # (84, 21, 28, 6) 결합
    combined = np.zeros((84, 21, 28, 6), dtype='float32')
    for i in range(84):
        combined[i, :, :, 0] = ndvi[i]
        for j in range(5):
            combined[i, :, :, j+1] = weather_scaled[i, j]
    return combined

def create_sequences(data, seq_length=12):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, :, :, 0:1])
    return np.array(X, dtype='float32'), np.array(y, dtype='float32')

# 데이터 준비
combined_data = prepare_data('./data/processed/X_train_final.npy', 'data/weather_data.csv')
X_all, y_all = create_sequences(combined_data)

# 학습/검증 분리
X_train, X_val = X_all[:-12], X_all[-12:]
y_train, y_val = y_all[:-12], y_all[-12:]

# 3. tf.data 파이프라인 구축 (GPU 데이터 전송 최적화)
batch_size = 4
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(len(X_train)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# 4. 경량화된 ConvLSTM 모델 정의
def build_cuda_optimized_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # ConvLSTM2D (과적합 방지를 위해 필터 수 최소화)
    x = layers.ConvLSTM2D(
        filters=16, kernel_size=(3, 3), padding='same',
        return_sequences=False,
        kernel_regularizer=regularizers.l2(1e-4)
    )(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # 출력층 (float32로 강제하여 수치 안정성 확보)
    outputs = layers.Conv2D(
        filters=1, kernel_size=(1, 1), activation='sigmoid', 
        padding='same', dtype='float32'
    )(x)
    
    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='mse',
        metrics=['mae']
    )
    return model

model = build_cuda_optimized_model((12, 21, 28, 6))

# 5. 학습 (TensorBoard 및 EarlyStopping 적용)
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=20) # 학습 정체 시 LR 감소
]

history = model.fit(
    train_ds,
    epochs=200, # GPU니까 에포크를 넉넉히 잡고 EarlyStopping에 맡깁니다
    validation_data=val_ds,
    callbacks=callbacks
)

import os

# 1. 저장 경로 설정
os.makedirs('./models', exist_ok=True)
os.makedirs('./plots', exist_ok=True)

# 2. 모델 저장 (.h5 또는 SavedModel 형식)
model_save_path = './models/gotjawal_convlstm_multimodal.h5'
model.save(model_save_path)
print(f"✅ 모델이 성공적으로 저장되었습니다: {model_save_path}")

# 3. 학습 곡선(Loss & MAE) 시각화 및 저장
def save_training_plots(history, save_path):
    plt.figure(figsize=(12, 5))
    
    # Loss (MSE) 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss (MSE)')
    plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
    plt.title('Model Loss (MSE) Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # MAE 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title('Model MAE Trend')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"✅ 학습 곡선 그래프가 저장되었습니다: {save_path}")

# 그래프 저장 실행
save_training_plots(history, './plots/training_history.png')

# 6. 결과 확인
def visualize_prediction(model, X_val, y_val):
    preds = model.predict(X_val)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    for i in range(3):
        # 실제값
        axes[0, i].imshow(y_val[i].squeeze(), cmap='RdYlGn', vmin=0, vmax=1)
        axes[0, i].set_title(f"Target (Month {i+1})")
        # 예측값
        axes[1, i].imshow(preds[i].squeeze(), cmap='RdYlGn', vmin=0, vmax=1)
        axes[1, i].set_title(f"Pred (Month {i+1})")
    
    plt.tight_layout()
    plt.show()

visualize_prediction(model, X_val, y_val)