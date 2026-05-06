import numpy as np
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.wkt import loads
import pandas as pd
from rasterio import features
from affine import Affine
from matplotlib.widgets import Slider

def interactive_gotjawal_viewer(data_path, roi_csv_path):
    # 1. 데이터 로드
    data = np.load(data_path)  # (84, 21, 28)
    df = pd.read_csv(roi_csv_path)
    poly = loads(df['wkt'].iloc[0])
    minx, miny, maxx, maxy = poly.bounds

    # 2. 마스크 생성 (한 번만 계산하면 됩니다)
    h, w = data[0].shape
    res_x = (maxx - minx) / w
    res_y = (maxy - miny) / h
    transform = Affine.translation(minx, maxy) * Affine.scale(res_x, -res_y)
    mask = features.geometry_mask([poly], out_shape=(h, w), transform=transform, invert=True)

    # 3. 그래프 초기 설정
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.2)  # 하단 슬라이더 공간 확보

    # 초기 화면 (0번 인덱스)
    initial_img = np.where(mask, data[0], np.nan)
    display_img = (initial_img * 2) - 1 # NDVI 복원

    im = ax.imshow(display_img, cmap='RdYlGn', vmin=-1, vmax=1,
                   extent=[minx, maxx, miny, maxy], 
                   alpha=0.8, zorder=2, interpolation='nearest')
    
    # 다각형 경계선
    x, y = poly.exterior.xy
    ax.plot(x, y, color='black', linewidth=1.5, zorder=3)

    # 배경 지도
    margin = 0.005
    ax.set_xlim(minx - margin, maxx + margin)
    ax.set_ylim(miny - margin, maxy + margin)
    ctx.add_basemap(ax, crs='EPSG:4326', source=ctx.providers.OpenStreetMap.Mapnik, zoom=16)

    plt.colorbar(im, label='NDVI Index')
    title = ax.set_title(f"Month: 1 (Index 0)", fontsize=15)

    # 4. 슬라이더 추가 (화살표 대신 드래그로 빠르게 넘기기 위해)
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03]) # [left, bottom, width, height]
    slider = Slider(ax_slider, 'Month Index', 0, len(data)-1, valinit=0, valfmt='%d')

    # 5. 업데이트 함수 정의
    def update(val):
        idx = int(slider.val)
        new_data = np.where(mask, data[idx], np.nan)
        new_display = (new_data * 2) - 1
        im.set_data(new_display)
        title.set_text(f"Month Index: {idx} ({2019 + idx//12}/{idx%12 + 1})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    
    print("💡 팁: 슬라이더의 양 끝 화살표를 누르거나 드래그하여 월을 변경하세요.")
    plt.show()

# 실행
interactive_gotjawal_viewer('./data/processed/X_train_final.npy', 'gotjawal_roi.csv')