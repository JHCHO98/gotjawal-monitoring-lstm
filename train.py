import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from model import GotjawalConvLSTM
from make_input_data import load_data, create_sequences

# 1. 장치 설정 (CUDA 가속 확인)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
else:
    print("⚠️ GPU를 찾지 못했습니다. CPU로 진행합니다.")

if __name__ == "__main__":
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