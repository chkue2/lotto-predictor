import streamlit as st
import numpy as np
import pandas as pd
import itertools
import random
import time
import os
import matplotlib.pyplot as plt

from matplotlib import font_manager, rc
import platform

# === 한글 폰트 설정 ===
if platform.system() == 'Darwin':  # macOS
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # Windows
    rc('font', family='Malgun Gothic')
else:  # Linux
    rc('font', family='NanumGothic')

plt.rcParams['axes.unicode_minus'] = False

# =========================
# 1️⃣ 페이지 설정
# =========================
st.set_page_config(page_title="통합 로또 추천기 V8 Dual", layout="centered")
st.title("🎯 통합 로또 추천기 V8 Dual")

# =========================
# 2️⃣ 데이터 불러오기
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
def monte_carlo_vectorized(trans_matrix, last_draw, trials=3000, focus_mode=False):
    probs_base = trans_matrix[[n-1 for n in last_draw]].sum(0)
    probs_base = np.maximum(probs_base, 0.01)
    if focus_mode:
        probs_base = probs_base ** 2
    probs_base /= probs_base.sum()
    draws = np.random.choice(np.arange(1,46), size=(trials,6), p=probs_base)
    counts = np.bincount(draws.flatten()-1, minlength=45)
    return counts / counts.sum()

# =========================
# 5️⃣ 그룹 기반 후보 생성
# =========================
def divide_into_groups(probabilities, focus_mode=False):
    sorted_idx = np.argsort(-probabilities)
    if focus_mode:
        g1 = sorted_idx[:10]+1
        g2 = sorted_idx[10:25]+1
        g3 = sorted_idx[25:]+1
    else:
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
# 6️⃣ Gianella 패턴 (V7 Grid)
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
# 7️⃣ 원형 패턴 점수
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
# 8️⃣ 피트니스 함수
# =========================
def fitness_func(comb, probabilities, focus_mode=False):
    eff = sum(probabilities[i-1] for i in comb)
    pat_v7 = gianella_pattern_v7(comb)
    pat_circ = gianella_pattern_circular(comb)
    combined_pattern = (pat_v7 * 0.5 + pat_circ * 0.5)
    if focus_mode:
        total_score = 0.85 * eff + 0.15 * (combined_pattern / 50)
    else:
        total_score = 0.7 * eff + 0.3 * (combined_pattern / 50)
    return eff, pat_v7, pat_circ, combined_pattern, total_score

# =========================
# 9️⃣ 조합 간 유사도
# =========================
def combination_similarity(a, b):
    return len(set(a) & set(b))

# =========================
# 🔟 조합 생성 (진행바 포함)
# =========================
def generate_final_combinations(n_sets=10, focus_mode=False):
    trans = build_transition_matrix(numbers_arr)
    last_draw = numbers_arr[-1]
    mc1 = monte_carlo_vectorized(trans, last_draw, focus_mode=focus_mode)
    mc2 = monte_carlo_vectorized(trans, last_draw, focus_mode=focus_mode)
    probs = (mc1 + mc2)/2

    groups = divide_into_groups(probs, focus_mode=focus_mode)
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

    # Streamlit 진행바
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(n_sets):
        scored = []
        for c in candidates:
            eff, pat_v7, pat_circ, pat_comb, total = fitness_func(c, probs, focus_mode=focus_mode)
            diversity_penalty = sum(combination_similarity(c, prev[0]) for prev in final_results) * 0.01
            final_score = total - diversity_penalty
            scored.append((c, eff, pat_v7, pat_circ, pat_comb, final_score))
        if not scored:
            break
        scored.sort(key=lambda x: x[5], reverse=True)
        best = scored[0]
        final_results.append(best)
        try:
            candidates.remove(best[0])
        except ValueError:
            pass
        # 진행바 업데이트
        progress_bar.progress((i+1)/n_sets)
        status_text.text(f"{i+1}/{n_sets}번째 조합 생성 중... ({'집중형' if focus_mode else '균형형'})")
        time.sleep(0.05)

    status_text.text(f"✅ {'집중형' if focus_mode else '균형형'} 조합 생성 완료!")
    progress_bar.empty()
    return final_results, probs

# =========================
# 리포트 유틸 함수
# =========================
def compute_historic_freq(numbers_array):
    flat = np.array(numbers_array).flatten()
    counts = np.bincount(flat-1, minlength=45)
    probs = counts / counts.sum()
    return counts, probs

def cooccurrence_matrix(numbers_array):
    mat = np.zeros((45,45), dtype=int)
    for draw in numbers_array:
        for a,b in itertools.combinations(draw,2):
            mat[a-1,b-1] += 1
            mat[b-1,a-1] += 1
    return mat

def combos_to_df(results_list, start_index=1, label="균형형"):
    rows = []
    for idx, (comb, eff, pat_v7, pat_circ, pat_comb, score) in enumerate(results_list, start=start_index):
        rows.append({
            "rank": idx,
            "type": label,
            "combo": comb,
            "eff": eff,
            "v7": pat_v7,
            "circ": pat_circ,
            "pat": pat_comb,
            "score": score
        })
    return pd.DataFrame(rows)

# =========================
# 11️⃣ UI 버튼 및 실행
# =========================
if st.button("추천 번호 생성 & 분석 리포트"):
    with st.spinner("계산 중... 잠시만 기다려주세요."):
        # 균형형
        results_balanced, probs_balanced = generate_final_combinations(10, focus_mode=False)
        df_bal = combos_to_df(results_balanced, start_index=1, label="균형형")

        # 집중형
        results_focused, probs_focused = generate_final_combinations(10, focus_mode=True)
        df_focus = combos_to_df(results_focused, start_index=11, label="집중형")

        # 통합 DataFrame
        result_df = pd.concat([df_bal, df_focus]).reset_index(drop=True)

        st.success("🎯 추천 번호 생성 완료!")

        st.subheader("✅ 균형형 추천 10조합")
        for _, row in df_bal.iterrows():
            st.write(f"[{int(row['rank']):02}] {row['combo']} | 확률: {row['eff']:.4f} | V7: {row['v7']:.1f} | 원형: {row['circ']:.1f} | 통합: {row['pat']:.1f} | 점수: {row['score']:.4f}")

        st.subheader("🔥 확률 집중형 추천 10조합")
        for _, row in df_focus.iterrows():
            st.write(f"[{int(row['rank']):02}] {row['combo']} | 확률: {row['eff']:.4f} | V7: {row['v7']:.1f} | 원형: {row['circ']:.1f} | 통합: {row['pat']:.1f} | 점수: {row['score']:.4f}")

        # =========================
        # 분석 리포트
        # =========================
        st.markdown("---")
        st.subheader("📊 강화된 분석 리포트")

        # 과거 데이터 핫/콜드
        hist_counts, hist_probs = compute_historic_freq(numbers_arr)
        hot_idx = np.argsort(-hist_counts)[:10] + 1
        cold_idx = np.argsort(hist_counts)[:10] + 1
        st.write("**과거 데이터(전체) — 핫 10 / 콜드 10**")
        st.write(f"Hot: {hot_idx.tolist()}, Cold: {cold_idx.tolist()}")

        # 과거 데이터 히스토그램 + 누적확률
        fig1, ax1 = plt.subplots(figsize=(9,3))
        idxs = np.arange(1,46)
        ax1.bar(idxs, hist_counts, color='skyblue', label='출현 횟수')
        ax2 = ax1.twinx()
        ax2.plot(idxs, np.cumsum(hist_probs), color='red', marker='o', label='누적확률')
        ax1.set_xlabel("번호"); ax1.set_ylabel("등장 횟수"); ax2.set_ylabel("누적확률")
        ax1.set_title("과거 데이터 번호 등장 횟수 및 누적확률")
        ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
        st.pyplot(fig1)

        # 공출현 히트맵 + 상위 10쌍
        mat = cooccurrence_matrix(numbers_arr)
        fig2, ax2 = plt.subplots(figsize=(7,6))
        im = ax2.imshow(mat, interpolation='nearest', cmap='YlOrRd')
        ax2.set_title("과거 데이터 공출현 행렬")
        ax2.set_xlabel("번호"); ax2.set_ylabel("번호")
        ax2.set_xticks(range(45)); ax2.set_xticklabels(range(1,46))
        ax2.set_yticks(range(45)); ax2.set_yticklabels(range(1,46))
        fig2.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        st.pyplot(fig2)

        co_pairs = [((i+1,j+1), mat[i,j]) for i in range(45) for j in range(i+1,45)]
        co_pairs.sort(key=lambda x: -x[1])
        st.write("**상위 10 공출현 번호 쌍:**", [p[0] for p in co_pairs[:10]])

        # 생성된 20조합 분석
        all_generated = [c for c in result_df['combo']]
        flat_generated = np.array(all_generated).flatten()
        gen_counts = np.bincount(flat_generated-1, minlength=45)
        gen_order = np.argsort(-gen_counts) + 1
        st.write("Generated 번호 빈도 상위 10:", gen_order[:10].tolist())

        # 패턴 점수 히스토그램
        fig3, ax3 = plt.subplots(figsize=(9,3))
        ax3.bar(result_df['rank'], result_df['v7'], alpha=0.7, label='V7 패턴')
        ax3.bar(result_df['rank'], result_df['circ'], alpha=0.5, label='원형 패턴')
        ax3.set_xlabel("조합 순위"); ax3.set_ylabel("패턴 점수"); ax3.set_title("20조합 패턴 점수 비교")
        ax3.legend(); st.pyplot(fig3)

        # 번호 그룹별 등장 횟수
        group1 = flat_generated[(flat_generated >=1) & (flat_generated <=15)]
        group2 = flat_generated[(flat_generated >=16) & (flat_generated <=30)]
        group3 = flat_generated[(flat_generated >=31) & (flat_generated <=45)]
        st.write("**번호 그룹별 등장 횟수** 1~15:{}, 16~30:{}, 31~45:{}".format(len(group1), len(group2), len(group3)))

        # 조합 간 중복 통계
        overlaps = [len(set(a) & set(b)) for a,b in itertools.combinations(all_generated,2)]
        overlaps = np.array(overlaps)
        st.write(f"조합 간 평균 중복: {overlaps.mean():.3f}, 최대: {overlaps.max()}, 최소: {overlaps.min()}")

        # 균형형 vs 집중형 교집합
        bal_combos = [c for c in df_bal['combo']]
        foc_combos = [c for c in df_focus['combo']]
        inter_counts = [len(set(a) & set(b)) for a in bal_combos for b in foc_combos]
        inter_counts = np.array(inter_counts)
        st.write(f"균형형 vs 집중형 평균 교집합: {inter_counts.mean():.3f}, 최대: {inter_counts.max()}")
