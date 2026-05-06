import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # <--- 이 줄이 없으면 'plt is not defined'가 뜹니다!
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
    ndvi = np.load('./data/processed/X_train_final.npy') # (84, 21, 28)
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
        combined[i, 6, :, :] = mask # 마스크 채널
    return combined, mask

# 시퀀스 생성 (12개월 -> 1개월 예측)
def create_sequences(data, seq_len=12):
    X_list, y_list = [], []
    for i in range(len(data) - seq_len):
        X_list.append(data[i:i+seq_len]) # (12, 7, 21, 28)
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
model = GotjawalConvLSTM().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.MSELoss()

# 학습
os.makedirs('./models', exist_ok=True)
for epoch in range(100):
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
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1} | Loss: {train_loss/len(train_loader):.6f}")

torch.save(model.state_dict(), "./models/gotjawal_torch_model.pth")

# 5. 시각화 (테스트 샘플 1개)
model.eval()
with torch.no_grad():
    test_in = X_val[0:1].to(device)
    pred = model(test_in).cpu().numpy().squeeze()
    # NDVI 복원 (-1 ~ 1)
    pred_restored = (pred * 2) - 1
    
    plt.imshow(np.where(roi_mask, pred_restored, np.nan), cmap='RdYlGn', vmin=-1, vmax=1)
    plt.title("PyTorch Predicted NDVI")
    plt.colorbar()
    plt.show()