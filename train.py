import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from shapely.wkt import loads
from rasterio import features
from affine import Affine
from sklearn.preprocessing import MinMaxScaler
import os

# 1. 장치 설정 (CUDA 가속 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU를 찾지 못했습니다. CPU로 진행합니다.")

# 2. 데이터 및 마스크 준비 (Affine 연동)
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

# 3. ConvLSTM 모델 정의
class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=kernel_size//2)

    def forward(self, x, hidden):
        h_cur, c_cur = hidden
        combined = torch.cat([x, h_cur], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.split(gates, self.hidden_channels, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class GotjawalConvLSTM(nn.Module):
    def __init__(self, in_channels=7, hidden_channels=16):
        super().__init__()
        self.conv_lstm = ConvLSTMCell(in_channels, hidden_channels, kernel_size=3)
        self.decoder = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: (Batch, Time, Channel, H, W)
        b, t, _, h, w = x.size()
        h_t = torch.zeros(b, 16, h, w).to(x.device)
        c_t = torch.zeros(b, 16, h, w).to(x.device)

        for i in range(t):
            h_t, c_t = self.conv_lstm(x[:, i], (h_t, c_t))

        return torch.sigmoid(self.decoder(h_t))

# 4. 메인 실행
data, roi_mask = load_data()
X_tensor, y_tensor = create_sequences(data)

# 시계열 분리
X_train, X_val = X_tensor[:-12], X_tensor[-12:]
y_train, y_val = y_tensor[:-12], y_tensor[-12:]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=4, shuffle=True)
val_loader   = DataLoader(TensorDataset(X_val,   y_val),   batch_size=4, shuffle=False)

model = GotjawalConvLSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

os.makedirs('./models', exist_ok=True)
os.makedirs('./plots',  exist_ok=True)

# Early Stopping 설정
best_val_loss    = float('inf')
patience         = 10
patience_counter = 0

train_losses = []
val_losses   = []

# 5. 학습
for epoch in range(200):
    # --- Train ---
    model.train()
    train_loss = 0
    for bx, by in train_loader:
        bx, by = bx.to(device), by.to(device)
        optimizer.zero_grad()
        output = model(bx)
        loss = criterion(output, by)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # --- Validation ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for bx, by in val_loader:
            bx, by = bx.to(device), by.to(device)
            val_loss += criterion(model(bx), by).item()

    epoch_train = train_loss / len(train_loader)
    epoch_val   = val_loss   / len(val_loader)
    train_losses.append(epoch_train)
    val_losses.append(epoch_val)

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1:>3} | Train Loss: {epoch_train:.6f} | Val Loss: {epoch_val:.6f}")

    # --- Early Stopping ---
    if epoch_val < best_val_loss:
        best_val_loss    = epoch_val
        patience_counter = 0
        torch.save(model.state_dict(), "./models/gotjawal_torch_model.pth")  # 최적 가중치 저장
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"⏹ Early Stopping at epoch {epoch+1} (best val loss: {best_val_loss:.6f})")
            break

# 최적 가중치 복원
model.load_state_dict(torch.load("./models/gotjawal_torch_model.pth"))
print("✅ 최적 가중치 복원 완료")

# 6. Loss 곡선 저장
plt.figure(figsize=(8, 4))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses,   label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig('./plots/training_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Loss plot 저장 완료: ./plots/training_history.png")

# 7. NDVI 예측 시각화 저장
model.eval()
with torch.no_grad():
    test_in = X_val[-1:].to(device)
    pred = model(test_in).cpu().numpy().squeeze()
    pred_restored = (pred * 2) - 1  # NDVI 복원 (-1 ~ 1)

plt.figure(figsize=(6, 5))
plt.imshow(np.where(roi_mask, pred_restored, np.nan), cmap='RdYlGn', vmin=-1, vmax=1)
plt.title("PyTorch Predicted NDVI(2026/01)")
plt.colorbar()
plt.tight_layout()
plt.savefig('./plots/prediction.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ NDVI plot 저장 완료: ./plots/prediction.png")