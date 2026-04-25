import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.widgets import PolygonSelector
import contextily as ctx
import pandas as pd
from shapely.geometry import Polygon
import os

# 1. 설정 (제주 환상숲 부근 좌표로 초기 세팅)
# 중심점: 저지리 환상숲 부근 (위경도)
center_lat, center_lon = 33.326048, 126.264015
delta = 0.005 # 보여줄 범위 (약 500m)

class GotjawalDrawer:
    def __init__(self):
        self.points = []
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        # 배경 지도 띄우기 (3857 좌표계 사용)
        # 범위를 설정해서 환상숲 근처만 크게 띄웁니다.
        temp_gdf = gpd.GeoDataFrame(geometry=[Polygon([
            (center_lon-delta, center_lat-delta), (center_lon+delta, center_lat-delta),
            (center_lon+delta, center_lat+delta), (center_lon-delta, center_lat+delta)
        ])], crs="EPSG:4326").to_crs(epsg=3857)
        
        temp_gdf.plot(ax=self.ax, alpha=0) # 투명하게 범위만 잡기
        ctx.add_basemap(self.ax, source=ctx.providers.OpenStreetMap.Mapnik)
        
        self.ax.set_title("Click to draw Gotjawal boundary\n(Close window when finished)")
        
        # 마우스 선택기 연결
        self.poly = PolygonSelector(self.ax, self.onselect)
        plt.show()

    def onselect(self, verts):
        self.points = verts
        print(f"현재 선택된 점 개수: {len(verts)}")

    def save_to_csv(self, filename="manual_gotjawal.csv"):
        if len(self.points) < 3:
            print("다각형을 만들려면 최소 3개의 점이 필요합니다!")
            return

        # 선택된 좌표는 3857이므로 다시 4326(위경도)으로 변환
        poly_3857 = Polygon(self.points)
        gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:3857", geometry=[poly_3857])
        gdf_4326 = gdf.to_crs(epsg=4326)
        
        # CSV 저장
        df_save = pd.DataFrame({'wkt': [gdf_4326.geometry.iloc[0].wkt]})
        df_save.to_csv(filename, index=False)
        print(f"✅ 저장 완료: {filename}")
        print(f"WKT: {gdf_4326.geometry.iloc[0].wkt}")

# 실행
drawer = GotjawalDrawer()
drawer.save_to_csv("my_gotjawal_roi.csv")