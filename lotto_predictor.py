import streamlit as st
import requests
import pandas as pd
import random
import time
import re
import matplotlib.pyplot as plt

# =========================
# 1️⃣ 로또 용지 배열 & 좌표
# =========================
grid = [
    [1,2,3,4,5,6,7],
    [8,9,10,11,12,13,14],
    [15,16,17,18,19,20,21],
    [22,23,24,25,26,27,28],
    [29,30,31,32,33,34,35],
    [36,37,38,39,40,41,42],
    [43,44,45]
]

num_to_coord = {}
for r,row in enumerate(grid):
    for c,num in enumerate(row):
        num_to_coord[num] = (c, -r)

# =========================
# 2️⃣ 최신 회차 번호 가져오기
# =========================
def get_latest_draw_no():
    url = "https://www.dhlottery.co.kr/common.do?method=main"
    try:
        resp = requests.get(url)
        html = resp.text
        m = re.search(r'id="lottoDrwNo">(\d+)<\/strong>', html)
        if m:
            return int(m.group(1))
    except:
        return None
    return None

# =========================
# 3️⃣ 특정 회차 번호 가져오기
# =========================
def fetch_lotto(draw_no):
    url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}"
    try:
        data = requests.get(url).json()
        if data.get("returnValue") != "success":
            return None
        nums = [data[f"drwtNo{i}"] for i in range(1,7)]
        bonus = data["bnusNo"]
        return {"draw_no": draw_no, "numbers": nums, "bonus": bonus}
    except:
        return None

# =========================
# 4️⃣ 최근 N회차 불러오기
# =========================
def fetch_recent(draw_count=50):
    latest = get_latest_draw_no()
    if not latest:
        st.error("최신 회차를 가져올 수 없습니다.")
        return pd.DataFrame()
    start = max(1, latest - draw_count + 1)
    records = []
    for drw in range(start, latest + 1):
        rec = fetch_lotto(drw)
        if rec:
            records.append(rec)
        time.sleep(0.05)
    return pd.DataFrame(records)

# =========================
# 5️⃣ 구간별 평균 계산
# =========================
def get_group_stats(df):
    groups = {i: [] for i in range(1,6)}
    for nums in df['numbers']:
        counts = [0]*5
        for n in nums:
            if 1<=n<=9: counts[0]+=1
            elif 10<=n<=19: counts[1]+=1
            elif 20<=n<=29: counts[2]+=1
            elif 30<=n<=39: counts[3]+=1
            elif 40<=n<=45: counts[4]+=1
        for i in range(5):
            groups[i+1].append(counts[i])
    avg_counts = {i: round(sum(lst)/len(lst),2) for i,lst in groups.items()}
    return avg_counts

def get_group_weights(avg_counts):
    weights = []
    for i,num_range in enumerate([(1,9),(10,19),(20,29),(30,39),(40,45)],1):
        start,end = num_range
        w = avg_counts[i]/6
        for n in range(start,end+1):
            weights.append((n,w))
    nums,ws = zip(*weights)
    ws = pd.Series(ws,index=nums)
    ws = ws/ws.sum()
    return ws

# =========================
# 6️⃣ 연속 번호
# =========================
def count_consecutive(nums):
    nums = sorted(nums)
    max_seq = 1
    cur_seq = 1
    for i in range(1,len(nums)):
        if nums[i]==nums[i-1]+1:
            cur_seq+=1
            max_seq=max(max_seq,cur_seq)
        else:
            cur_seq=1
    return max_seq

def get_consecutive_stats(df):
    seq_counts=[]
    for nums in df['numbers']:
        seq_counts.append(count_consecutive(nums))
    series = pd.Series(seq_counts)
    probs = series.value_counts(normalize=True).sort_index()
    return probs

# =========================
# 7️⃣ 좌표 기반 선 패턴 (세로 1자 패널티 적용)
# =========================
def coord_distance(n1,n2):
    x1,y1=num_to_coord[n1]
    x2,y2=num_to_coord[n2]
    return abs(x1-x2)+abs(y1-y2)

def adjacent_score(combo):
    score=0
    xs=[num_to_coord[n][0] for n in combo]
    col_counts = pd.Series(xs).value_counts()
    # 세로 집중 패널티
    for cnt in col_counts:
        if cnt>=3: score -= 2*(cnt-2)
    # 거리 기반 점수
    for i in range(len(combo)-1):
        dist=coord_distance(combo[i],combo[i+1])
        if dist==1: score+=2
        elif dist==2: score+=1
    return score

# =========================
# 8️⃣ 번호 생성 (연번 최대 2개, 연번 그룹 최대 1개)
# =========================
def generate_numbers(ws, conseq_probs, past_combos, sum_range=(100,170), even_range=(2,4)):
    while True:
        selected = random.choices(ws.index.tolist(), weights=ws.tolist(), k=6)
        selected = sorted(set(selected))
        while len(selected) < 6:
            candidate = random.choices(ws.index.tolist(), weights=ws.tolist(), k=1)[0]
            if candidate not in selected:
                selected.append(candidate)
        selected.sort()

        # 과거 당첨번호와 겹치면 다시 선택
        if any(set(selected) == set(past) for past in past_combos):
            continue

        even = sum(1 for n in selected if n % 2 == 0)
        total_sum = sum(selected)
        pattern_score = adjacent_score(selected)

        # 🔹 연번 그룹 개수 계산
        seq_count = 0
        nums_sorted = sorted(selected)
        i = 0
        while i < len(nums_sorted) - 1:
            if nums_sorted[i+1] == nums_sorted[i] + 1:
                seq_count += 1
                # 그룹 건너뛰기
                while i+1 < len(nums_sorted) and nums_sorted[i+1] == nums_sorted[i] + 1:
                    i += 1
            i += 1

        max_seq = count_consecutive(selected)

        # 조건: 연번 최대 2개, 연번 그룹 최대 1개
        if (even_range[0] <= even <= even_range[1] and
            sum_range[0] <= total_sum <= sum_range[1] and
            max_seq <= 2 and
            seq_count <= 1):
            return tuple(selected), pattern_score

# =========================
# 9️⃣ 분석 & 예측
# =========================
def analyze_and_predict(df50, df100, num_combinations=10):
    if df50.empty or df100.empty:
        return []

    avg50=get_group_stats(df50)
    avg100=get_group_stats(df100)
    combined_avg={k: avg50[k]*0.7+avg100[k]*0.3 for k in avg50}
    ws=get_group_weights(combined_avg)

    conseq50=get_consecutive_stats(df50)
    conseq100=get_consecutive_stats(df100)
    conseq_probs=(conseq50*0.7).add(conseq100*0.3,fill_value=0)
    conseq_probs=conseq_probs/conseq_probs.sum()

    past_combos=df100["numbers"].tolist()
    candidate_list=[]
    for _ in range(1000):
        combo, pat_score=generate_numbers(ws, conseq_probs, past_combos)
        candidate_list.append((combo, pat_score))
    candidate_list.sort(key=lambda x:x[1], reverse=True)
    final_combos=[x[0] for x in candidate_list[:num_combinations]]

    return final_combos

# =========================
# 🔟 시각화
# =========================
def plot_combo(combo):
    plt.figure(figsize=(7,7))
    for r,row in enumerate(grid):
        for c,num in enumerate(row):
            plt.text(c,-r,str(num),fontsize=12,ha='center',va='center',color='gray')
    xs=[num_to_coord[n][0] for n in combo]
    ys=[num_to_coord[n][1] for n in combo]
    plt.plot(xs,ys,marker='o',color='red',linewidth=2)
    for n,x,y in zip(combo,xs,ys):
        plt.text(x,y+0.1,str(n),fontsize=12,color='blue',ha='center')
    plt.axis('off')
    st.pyplot(plt)

# =========================
# 1️⃣1️⃣ Streamlit UI
# =========================
st.title("🎯 로또 예측기")
st.caption("50회 최신 + 100회 장기 패턴 혼합")

if st.button("예측 번호 10개 생성 및 시각화"):
    st.info("🔄 최근 50회 + 100회 로또 데이터 수집 중...")
    df50=fetch_recent(50)
    df100=fetch_recent(100)

    if not df50.empty and not df100.empty:
        st.success("✅ 데이터 수집 완료! 번호 생성 중...")
        final_combos=analyze_and_predict(df50, df100)
        st.subheader("생성된 10개 조합")
        for idx, combo in enumerate(final_combos,1):
            st.write(f"{idx}: {combo}")
            plot_combo(combo)
    else:
        st.error("데이터를 불러오지 못했습니다.")
