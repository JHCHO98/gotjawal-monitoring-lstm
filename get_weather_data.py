import os
from dotenv import load_dotenv
import requests
import time
import csv
from datetime import date, timedelta
from collections import defaultdict

load_dotenv()

API_KEY = os.getenv('WEATHER_API_KEY')   # 공공데이터포털 인증키 (URL 인코딩 전 원문 키)
STN_ID  = 104                   # 지점번호: 108=서울, 133=대전, 143=대구, 156=광주, 159=부산, 184=제주
OUTPUT_CSV = "data/monthly_weather.csv"

 
BASE_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"

# 수집할 컬럼 (API 응답 필드명 → 표시 이름)
FIELDS = {
    "avgTa":    "평균기온(°C)",
    "sumRn":    "강수량합계(mm)",
    "avgRhm":   "평균습도(%)",
    "avgTca":   "평균전운량(10분위)",
    "sumSsHr":  "합계일조시간(hr)",
}
 
 
def date_range_by_month(start_year, start_month, end_year, end_month):
    """월별 (startDt, endDt) 쌍을 생성"""
    y, m = start_year, start_month
    while (y, m) <= (end_year, end_month):
        first = date(y, m, 1)
        # 해당 월의 마지막 날
        if m == 12:
            last = date(y + 1, 1, 1) - timedelta(days=1)
        else:
            last = date(y, m + 1, 1) - timedelta(days=1)
        yield y, m, first.strftime("%Y%m%d"), last.strftime("%Y%m%d")
        m += 1
        if m > 12:
            m = 1
            y += 1
 
 
def fetch_month(service_key, stn_id, start_dt, end_dt, max_rows=50):
    """한 달치 일자료를 모두 가져옴 (페이지네이션 처리)"""
    query_params = {
        "numOfRows": max_rows,
        "pageNo":    1,
        "dataType":  "JSON",
        "dataCd":    "ASOS",
        "dateCd":    "DAY",
        "startDt":   start_dt,
        "endDt":     end_dt,
        "stnIds":    stn_id,
    }
 
    all_items = []
    while True:
        try:
            url = f"{BASE_URL}?serviceKey={service_key}"
            resp = requests.get(url, params=query_params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"  [오류] 요청 실패: {e}")
            break
 
        data = resp.json()
 
        # 응답 구조: response > body > items > item
        try:
            body = data["response"]["body"]
            items = body["items"]["item"]
            total = int(body["totalCount"])
        except (KeyError, TypeError):
            # NODATA_ERROR 등 정상 빈 응답 처리
            header = data.get("response", {}).get("header", {})
            code = header.get("resultCode", "?")
            if code == "03":
                print(f"  → 해당 기간 데이터 없음")
            else:
                print(f"  [응답 오류] {header}")
            break
 
        if isinstance(items, dict):
            items = [items]   # 단건일 때 dict로 오는 경우 대비
 
        all_items.extend(items)
 
        # 페이지 끝 확인
        if len(all_items) >= total:
            break
        query_params["pageNo"] += 1
        time.sleep(0.3)   # API 과호출 방지
 
    return all_items
 
 
def monthly_average(items):
    """일자료 목록 → 월 통계 계산"""
    sums   = defaultdict(float)
    counts = defaultdict(int)
 
    for item in items:
        for field in FIELDS:
            val = item.get(field)
            if val is not None and val != "":
                try:
                    sums[field]   += float(val)
                    counts[field] += 1
                except ValueError:
                    pass
 
    result = {}
    for field, label in FIELDS.items():
        n = counts[field]
        if n == 0:
            result[label] = ""
        elif field == "sumRn":
            # 강수량은 합계 그대로 (월 합산)
            result[label] = round(sums[field], 1)
        elif field == "sumSsHr":
            # 일조시간도 월 합산
            result[label] = round(sums[field], 1)
        else:
            # 나머지는 평균
            result[label] = round(sums[field] / n, 2)
    return result
 
 
def main():
    print(f"기상청 ASOS API 월평균 데이터 수집")
    print(f"지점: {STN_ID} | 기간: 2019-01 ~ 2025-12\n")
 
    rows = []
 
    for year, month, start_dt, end_dt in date_range_by_month(2019, 1, 2025, 12):
        label = f"{year}-{month:02d}"
        print(f"[{label}] {start_dt} ~ {end_dt} 조회 중...", end=" ", flush=True)
 
        items = fetch_month(API_KEY, STN_ID, start_dt, end_dt)
 
        if not items:
            print(f"  → 데이터 없음, 건너뜀")
            continue
 
        stats = monthly_average(items)
        row = {"연월": label, "일수": len(items)}
        row.update(stats)
        rows.append(row)
 
        avg_ta   = stats.get("평균기온(°C)", "-")
        sum_rn   = stats.get("강수량합계(mm)", "-")
        avg_rhm  = stats.get("평균습도(%)", "-")
        avg_tca  = stats.get("평균전운량(10분위)", "-")
        sum_sshr = stats.get("합계일조시간(hr)", "-")
        print(f"기온={avg_ta}°C, 강수={sum_rn}mm, 습도={avg_rhm}%, 구름={avg_tca}, 일조={sum_sshr}hr")
 
        time.sleep(0.5)   # API 호출 간격
 
    # CSV 저장
    if rows:
        fieldnames = ["연월", "일수"] + list(FIELDS.values())
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n✅ 완료! '{OUTPUT_CSV}' 저장됨 ({len(rows)}개월)")
    else:
        print("\n⚠️  저장할 데이터가 없습니다.")

if __name__ == "__main__":
    main()