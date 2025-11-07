import streamlit as st
import numpy as np
import pandas as pd
import itertools
import random
import time
import os

# =========================
# 1️⃣ 페이지 설정
# =========================
st.set_page_config(page_title="통합 로또 추천기 V8", layout="centered")
st.title("🎯 통합 로또 추천기 V8")

# =========================
# 2️⃣ 데이터 불러오기 (CSV 변경 감지)
# =========================
CSV_FILE = "lotto_data.csv"

@st.cache_data(show_spinner=False)
def load_lotto_data(file_path, file_mtime):
    df = pd.read_csv(file_path)
    df['numbers'] = df[[f"번호{i}" for i in range(1,7)]].values.tolist()
    return df

def get_file_mtime(file_path):
    return os.path.getmtime(file_path)

csv_mtime = get_file_mtime(CSV_FILE)
df = load_lotto_data(CSV_FILE, csv_mtime)
numbers_arr = np.array(df['numbers'].tolist())

# =========================
# 3️⃣ 마르코프 전이 확률
# =========================
def build_transition_matrix(numbers):
    n = 45
    m = np.zeros((n,n))
    for i in range(len(numbers)-1):
        for a in numbers[i]:
            for b in numbers[i+1]:
                m[a-1, b-1] += 1
    p = m / m.sum(1, keepdims=True)
    return np.nan_to_num(p)

# =========================
# 4️⃣ Monte Carlo 시뮬레이션
# =========================
def monte_carlo_vectorized(trans_matrix, last_draw, trials=3000):
    probs_base = trans_matrix[[n-1 for n in last_draw]].sum(0)
    probs_base = np.maximum(probs_base, 0.01)
    probs_base /= probs_base.sum()
    draws = np.random.choice(np.arange(1,46), size=(trials,6), p=probs_base)
    counts = np.bincount(draws.flatten()-1, minlength=45)
    return counts / counts.sum()

# =========================
# 5️⃣ 그룹 기반 후보 생성
# =========================
def divide_into_groups(probabilities):
    sorted_idx = np.argsort(-probabilities)
    g1 = sorted_idx[:15]+1
    g2 = sorted_idx[15:30]+1
    g3 = sorted_idx[30:]+1
    return g1.tolist(), g2.tolist(), g3.tolist()

def check_consecutive_rule(comb):
    comb = sorted(comb)
    groups = []
    cur = [comb[0]]
    for i in range(1,len(comb)):
        if comb[i] == comb[i-1]+1:
            cur.append(comb[i])
        else:
            if len(cur)>1: groups.append(cur)
            cur = [comb[i]]
    if len(cur)>1: groups.append(cur)
    if len(groups)>1 or any(len(g)>2 for g in groups): return False
    return True

def generate_group_combinations(groups):
    combs = []
    for c1 in itertools.combinations(groups[0],2):
        for c2 in itertools.combinations(groups[1],2):
            for c3 in itertools.combinations(groups[2],2):
                comb = sorted(set(c1+c2+c3))
                if len(comb)==6 and check_consecutive_rule(comb):
                    combs.append(comb)
    return combs

# =========================
# 6️⃣ Gianella 패턴 (V7 Grid 기반)
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

def gianella_pattern_v7(numbers):
    coords = [(r,c) for r,row in enumerate(lotto_grid) for c,v in enumerate(row) if v in numbers]
    rows = [0]*7
    cols = [0]*7
    for r,c in coords:
        rows[r] += 1
        cols[c] += 1
    row_penalty = sum(max(0, x-2)**2 for x in rows)
    col_penalty = sum(max(0, x-2)**2 for x in cols)
    balance_score = 50 - (row_penalty + col_penalty)
    diag1 = sum(r==c and r<len(lotto_grid) and c<len(lotto_grid[r]) for r,c in coords)
    diag2 = sum(c==6-r and r<len(lotto_grid) and c<len(lotto_grid[r]) for r,c in coords)
    diag_score = diag1 + diag2
    total_score = balance_score + diag_score
    return total_score

# =========================
# 7️⃣ 원형(Gianella) 패턴 - 다양성 강화
# =========================
def gianella_pattern_circular(numbers):
    zones = {
        1: range(1,8), 2: range(8,15), 3: range(15,22),
        4: range(22,29), 5: range(29,36), 6: range(36,43), 7: range(43,46)
    }
    counts = {z: len([n for n in numbers if n in rng]) for z, rng in zones.items()}
    diversity_bonus = len([v for v in counts.values() if v == 1])
    overlap_penalty = sum(max(0, v-2) for v in counts.values())
    score = 40 + (diversity_bonus * 2.5) - overlap_penalty
    return max(0, min(score, 70))

# =========================
# 8️⃣ 피트니스 함수 (통합형)
# =========================
def fitness_func(comb, probabilities):
    eff = sum(probabilities[i-1] for i in comb)
    pat_v7 = gianella_pattern_v7(comb)
    pat_circ = gianella_pattern_circular(comb)
    combined_pattern = (pat_v7 * 0.5 + pat_circ * 0.5)
    total_score = 0.7 * eff + 0.3 * (combined_pattern / 50)
    return eff, pat_v7, pat_circ, combined_pattern, total_score

# =========================
# 9️⃣ 조합 간 유사도 계산
# =========================
def combination_similarity(a, b):
    return len(set(a) & set(b))

# =========================
# 🔟 최종 조합 생성
# =========================
def generate_final_combinations(n_sets=10):
    trans = build_transition_matrix(numbers_arr)
    last_draw = numbers_arr[-1]
    mc1 = monte_carlo_vectorized(trans, last_draw)
    mc2 = monte_carlo_vectorized(trans, last_draw)
    probs = (mc1 + mc2)/2

    groups = divide_into_groups(probs)
    candidates = generate_group_combinations(groups)
    candidates = [sorted(c) for c in candidates]

    unique_candidates = []
    seen = set()
    for c in candidates:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    candidates = unique_candidates

    final_results = []
    displayed = st.empty()

    for i in range(n_sets):
        scored = []
        for c in candidates:
            eff, pat_v7, pat_circ, pat_comb, total = fitness_func(c, probs)
            diversity_penalty = sum(combination_similarity(c, prev[0]) for prev in final_results) * 0.01
            final_score = total - diversity_penalty
            scored.append((c, eff, pat_v7, pat_circ, pat_comb, final_score))

        scored.sort(key=lambda x: x[5], reverse=True)
        best = scored[0]
        final_results.append(best)
        candidates.remove(best[0])
        displayed.text(f"{i+1}번째 조합 생성 중...")
        time.sleep(0.05)

    displayed.text("✅ 모든 조합 생성 완료!")
    return final_results

# =========================
# 11️⃣ UI 버튼
# =========================
if st.button("추천 번호 생성"):
    with st.spinner("계산 중... 잠시만 기다려주세요."):
        results = generate_final_combinations(10)
        st.success("🎯 추천 번호 생성 완료!")
        for i,(comb, eff, pat_v7, pat_circ, pat_comb, score) in enumerate(results,1):
            st.write(f"{comb} | 확률: {eff:.4f} | V7패턴: {pat_v7:.1f} | 원형패턴: {pat_circ:.1f} | 통합패턴: {pat_comb:.1f} | 종합점수: {score:.4f}")
