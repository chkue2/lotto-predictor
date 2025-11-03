import streamlit as st
import requests, re, time, itertools, random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1️⃣ 대한민국 로또 7x7 배열
# =========================
lotto_grid = [
    [1,2,3,4,5,6,7],
    [8,9,10,11,12,13,14],
    [15,16,17,18,19,20,21],
    [22,23,24,25,26,27,28],
    [29,30,31,32,33,34,35],
    [36,37,38,39,40,41,42],
    [43,44,45]
]

def number_to_coord(num):
    for r, row in enumerate(lotto_grid):
        if num in row:
            return (r, row.index(num))
    return None

# =========================
# 2️⃣ 데이터 가져오기
# =========================
def get_latest_draw_no():
    url = "https://www.dhlottery.co.kr/common.do?method=main"
    try:
        resp = requests.get(url)
        m = re.search(r'id="lottoDrwNo">(\d+)<\/strong>', resp.text)
        if m: return int(m.group(1))
    except: return None
    return None

def fetch_lotto(draw_no):
    url = f"https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={draw_no}"
    try:
        data = requests.get(url).json()
        if data.get("returnValue") != "success": return None
        nums = [data[f"drwtNo{i}"] for i in range(1,7)]
        bonus = data["bnusNo"]
        return {"draw_no": draw_no, "numbers": nums, "bonus": bonus}
    except: return None

@st.cache_data
def fetch_recent(draw_count=100):
    latest = get_latest_draw_no()
    if not latest: return pd.DataFrame()
    start = max(1, latest - draw_count + 1)
    records = []
    for drw in range(start, latest + 1):
        rec = fetch_lotto(drw)
        if rec: records.append(rec)
        time.sleep(0.05)
    return pd.DataFrame(records)

# =========================
# 3️⃣ 마르코프 전이 행렬
# =========================
def build_transition_matrix(numbers):
    n_numbers = 45
    transition_count = np.zeros((n_numbers, n_numbers), dtype=int)
    for i in range(len(numbers)-1):
        current = numbers[i] - 1
        next_ = numbers[i+1] - 1
        for c in current:
            for n in next_:
                transition_count[c][n] += 1
    transition_prob = transition_count / transition_count.sum(axis=1, keepdims=True)
    transition_prob = np.nan_to_num(transition_prob)
    return transition_prob

# =========================
# 4️⃣ 몬테카를로 시뮬레이션
# =========================
def monte_carlo_a(last_draw, transition_prob, n_sim=2000):
    counts = np.zeros(45)
    for _ in range(n_sim):
        selected = set()
        for c in last_draw - 1:
            nums = np.random.choice(np.arange(45), 6, replace=False, p=transition_prob[c])
            selected.update(nums)
        for num in selected: counts[num] += 1
    return counts / n_sim

def monte_carlo_b(last_draw, transition_prob, n_sim=2000):
    counts = np.zeros(45)
    for _ in range(n_sim):
        selected = set()
        while len(selected) < 6:
            c = random.choice(last_draw) - 1
            num = np.random.choice(np.arange(45), 1, p=transition_prob[c])[0]
            selected.add(num)
        for num in selected: counts[num] += 1
    return counts / n_sim

def recommend_set(last_draw, transition_prob):
    probs_a = monte_carlo_a(last_draw, transition_prob)
    probs_b = monte_carlo_b(last_draw, transition_prob)
    avg_probs = (probs_a + probs_b)/2
    recommended = np.argsort(avg_probs)[-6:] + 1
    return sorted(recommended), avg_probs

def recommend_10_sets(df):
    numbers = np.array(df['numbers'].tolist())
    last_draw = numbers[-1]
    transition_prob = build_transition_matrix(numbers)
    sets = []
    for _ in range(10):
        rec_set, _ = recommend_set(last_draw, transition_prob)
        sets.append(rec_set)
    total_counts = np.zeros(45)
    for s in sets:
        for num in s: total_counts[num-1] += 1
    total_probs = total_counts / 10
    return sets, total_probs

# =========================
# 5️⃣ 그룹 분류 및 조합 생성
# =========================
def split_groups(probs):
    sorted_idx = np.argsort(probs)[::-1]
    group_a = sorted_idx[:15]+1
    group_b = sorted_idx[15:30]+1
    group_c = sorted_idx[30:]+1
    return group_a.tolist(), group_b.tolist(), group_c.tolist()

def generate_group_combinations(group_a, group_b, group_c):
    combs = []
    for n_a in range(1,6):
        for n_b in range(1,6-n_a):
            n_c = 6 - n_a - n_b
            for ca in itertools.combinations(group_a, n_a):
                for cb in itertools.combinations(group_b, n_b):
                    for cc in itertools.combinations(group_c, n_c):
                        combs.append(sorted(list(ca)+list(cb)+list(cc)))
    return combs

def calculate_efficiency(comb_list, probs):
    eff_list = []
    for comb in comb_list:
        eff = sum([probs[num-1] for num in comb])
        eff_list.append((comb, eff))
    eff_list.sort(key=lambda x: x[1], reverse=True)
    return eff_list

def select_best_combinations(eff_list, top_n=50):
    selected = []
    for comb, eff in eff_list:
        if comb not in selected: selected.append(comb)
        if len(selected)>=top_n: break
    return selected

# =========================
# 6️⃣ 다차원 지아넬라 패턴 분석
# =========================
def gianella_pattern(numbers):
    coords = [number_to_coord(n) for n in numbers]
    row_counts = [0]*7
    col_counts = [0]*7
    for r,c in coords:
        row_counts[r]+=1
        col_counts[c]+=1
    diag1 = sum([1 for r,c in coords if r==c])
    diag2 = sum([1 for r,c in coords if r==6-c])
    score = sum([x**2 for x in row_counts])+sum([x**2 for x in col_counts])+diag1+diag2
    return {
        "coords": coords, "row_counts": row_counts, "col_counts": col_counts,
        "diag1_count": diag1, "diag2_count": diag2, "pattern_score": score
    }

def plot_lotto_grid(numbers):
    grid_map = np.zeros((7,7))
    for n in numbers:
        coord = number_to_coord(n)
        if coord: r,c = coord; grid_map[r,c]=1
    fig, ax = plt.subplots()
    ax.imshow(grid_map, cmap='Greens', origin='upper')
    for r in range(7):
        for c in range(7):
            val = lotto_grid[r][c] if c<len(lotto_grid[r]) else ''
            ax.text(c, r, val, ha='center', va='center', color='black')
    ax.set_xticks(range(7)); ax.set_yticks(range(7))
    ax.set_xticklabels([]); ax.set_yticklabels([])
    return fig

# =========================
# 7️⃣ 최종 추천 필터링 (효율 + 패턴)
# =========================
def filter_by_pattern(best_combs, probs, pattern_threshold=8):
    filtered = []
    for comb in best_combs:
        pat = gianella_pattern(comb)
        if pat['pattern_score'] >= pattern_threshold:
            filtered.append((comb, pat))
    # 효율순 정렬
    filtered.sort(key=lambda x: sum([probs[num-1] for num in x[0]]), reverse=True)
    return filtered[:10]

# =========================
# 8️⃣ Streamlit UI (수정)
# =========================
st.title("로또 추천기")

if st.button("10세트 추천 & 최종 추천 조합 생성"):
    with st.spinner("번호 생성 중..."):
        df = fetch_recent(100)
        if df.empty: 
            st.error("최근 데이터를 가져올 수 없습니다.")
        else:
            # 10세트 추천과 각 번호 출현 확률 계산 (화면에는 표시하지 않음)
            sets, probs = recommend_10_sets(df)

            # 그룹 분류 및 모든 조합 생성
            group_a, group_b, group_c = split_groups(probs)
            all_combs = generate_group_combinations(group_a, group_b, group_c)
            eff_list = calculate_efficiency(all_combs, probs)
            best_combs = select_best_combinations(eff_list, top_n=50)

            # 최종 추천 10개 조합 (화면에 표시)
            final_combs = filter_by_pattern(best_combs, probs, pattern_threshold=8)

            st.subheader("🔹 최종 추천 조합 (상위 10개)")
            for i, (comb, pat) in enumerate(final_combs):
                st.write(f"조합 {i+1}: {comb}")
                st.write(f"패턴 점수: {pat['pattern_score']}, 행: {pat['row_counts']}, 열: {pat['col_counts']}, 대각선1: {pat['diag1_count']}, 대각선2: {pat['diag2_count']}")
                fig = plot_lotto_grid(comb)
                st.pyplot(fig)