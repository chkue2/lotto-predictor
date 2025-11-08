import streamlit as st
import numpy as np
import pandas as pd
import itertools
import time
import os
import matplotlib.pyplot as plt
from matplotlib import rc
import platform
from concurrent.futures import ThreadPoolExecutor

# =========================
# 한글 폰트 설정
# =========================
if platform.system() == 'Darwin':  # macOS
    rc('font', family='AppleGothic')
elif platform.system() == 'Windows':  # Windows
    rc('font', family='Malgun Gothic')
else:  # Linux
    rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# =========================
# 설정 및 데이터 불러오기
# =========================
st.set_page_config(page_title="통합 로또 추천기 V14", layout="centered")
st.title("🎯 통합 로또 추천기 V14")

CSV_FILE = "lotto_data.csv"

def load_lotto_data(file_path):
    df = pd.read_csv(file_path)
    df['numbers'] = df[[f"번호{i}" for i in range(1,7)]].values.tolist()
    return df

def get_file_mtime(file_path):
    return os.path.getmtime(file_path)

csv_mtime = get_file_mtime(CSV_FILE)
df = load_lotto_data(CSV_FILE)
numbers_arr = np.array(df['numbers'].tolist())

# =========================
# 1️⃣ 전이행렬 (마르코프)
# =========================
def build_transition_matrix(numbers):
    n = 45
    m = np.zeros((n,n), dtype=float)
    for i in range(len(numbers)-1):
        for a in numbers[i]:
            for b in numbers[i+1]:
                m[a-1,b-1] += 1
    row_sums = m.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = m / row_sums
    return np.nan_to_num(p)

# =========================
# 2️⃣ Monte Carlo 시뮬레이션
# =========================
def apply_recent_draw_penalty(probs_base, last_draw, penalty_factor=0.2):
    """
    최근 회차에 나온 번호들의 확률을 낮춤.
    penalty_factor: 0 ~ 1, 작을수록 확률 더 낮아짐
    """
    probs = probs_base.copy()
    for n in last_draw:
        probs[n-1] *= penalty_factor  # 최근 번호 확률 감소
    probs /= probs.sum()  # 확률 정규화
    return probs

def monte_carlo_vectorized(trans_matrix, last_draw, trials=3000, focus_mode=False, random_perturb=0.02, recent_penalty=True):
    probs_base = trans_matrix[[n-1 for n in last_draw]].sum(0)
    probs_base = np.maximum(probs_base, 0.01)
    
    if focus_mode:
        probs_base = probs_base ** 2
    
    if recent_penalty:
        probs_base = apply_recent_draw_penalty(probs_base, last_draw, penalty_factor=0.2)
    
    perturb = np.random.uniform(-random_perturb, random_perturb, size=probs_base.shape)
    probs_base += perturb
    probs_base = np.clip(probs_base, 0.001, None)
    probs_base /= probs_base.sum()
    
    draws = np.array([np.random.choice(np.arange(1,46), size=6, replace=False, p=probs_base) for _ in range(trials)])
    return draws


# =========================
# 3️⃣ 그룹 분할 및 후보 생성
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

lotto_grid = [
    [1,2,3,4,5,6,7],
    [8,9,10,11,12,13,14],
    [15,16,17,18,19,20,21],
    [22,23,24,25,26,27,28],
    [29,30,31,32,33,34,35],
    [36,37,38,39,40,41,42],
    [43,44,45]
]
GRID_POS = {v:(r,c) for r,row in enumerate(lotto_grid) for c,v in enumerate(row)}

def is_strict_diagonal(comb):
    coords = [GRID_POS[n] for n in comb]
    coords_sorted = sorted(coords, key=lambda x: (x[0], x[1]))
    drs = [coords_sorted[i+1][0]-coords_sorted[i][0] for i in range(len(coords_sorted)-1)]
    dcs = [coords_sorted[i+1][1]-coords_sorted[i][1] for i in range(len(coords_sorted)-1)]
    return len(set(drs))==1 and len(set(dcs))==1

def diagonal_penalty_score(comb):
    coords = [GRID_POS[n] for n in comb]
    diffs = [(coords[i+1][0]-coords[i][0], coords[i+1][1]-coords[i][1]) for i in range(len(coords)-1)]
    penalty = 0
    for dr, dc in diffs:
        if abs(dr)==abs(dc) and abs(dr)>=1:
            penalty += 1
    return max(0, 20-penalty*5)

# -------------------------
# 그룹 조합 생성 (샘플링 기반)
# -------------------------
def generate_group_combinations(groups, n_samples=5000):
    """조합 폭발 방지를 위해 랜덤 샘플링 기반으로 후보 조합 생성"""
    g0, g1, g2 = groups
    candidates = []
    g_all = g0 + g1 + g2
    g_all = list(set(g_all))
    while len(candidates) < n_samples:
        comb = np.random.choice(g_all, size=6, replace=False).tolist()
        if not check_consecutive_rule(comb):
            continue
        if is_strict_diagonal(comb) and np.random.rand() >= 0.1:
            continue
        if morphological_pattern_score(comb) == 0:
            continue
        candidates.append(comb)
    # 중복 제거
    candidates = list({tuple(c): c for c in candidates}.values())
    return candidates

# =========================
# 4️⃣ 패턴 점수
# =========================
def gianella_pattern_v7(numbers):
    coords = [GRID_POS[n] for n in numbers]
    rows = [0]*7; cols = [0]*7
    for r,c in coords:
        rows[r] += 1; cols[c] += 1
    row_penalty = sum(max(0,x-2)**2 for x in rows)
    col_penalty = sum(max(0,x-2)**2 for x in cols)
    balance_score = 50 - (row_penalty + col_penalty)
    diag1 = sum(r==c for r,c in coords)
    diag2 = sum(c==6-r for r,c in coords)
    return balance_score + diag1 + diag2

def gianella_pattern_circular(numbers):
    zones = {1: range(1,8),2: range(8,15),3: range(15,22),
             4: range(22,29),5: range(29,36),6: range(36,43),7: range(43,46)}
    counts = {z:0 for z in zones}
    for n in numbers:
        for z,rng in zones.items():
            if n in rng:
                counts[z]+=1
                break
    diversity_bonus = sum(1 for v in counts.values() if v==1)
    overlap_penalty = sum(max(0,v-2) for v in counts.values())
    return max(0,min(70,40 + diversity_bonus*2.5 - overlap_penalty))

def morphological_pattern_score(numbers):
    pos = [GRID_POS[n] for n in numbers]
    for dr, dc in [(1,1),(1,-1)]:
        for r,c in pos:
            chain=1
            nr,nc=r+dr,c+dc
            while (nr,nc) in GRID_POS.values() and (nr,nc) in pos:
                chain+=1
                nr+=dr; nc+=dc
            if chain>=4: return 0
    return 20

# =========================
# 5️⃣ 병렬화 평가
# =========================
def evaluate_patterns_batch(candidates):
    v7_vals=[]; circ_vals=[]; morph_vals=[]; diag_vals=[]
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda c: (
            gianella_pattern_v7(c),
            gianella_pattern_circular(c),
            morphological_pattern_score(c),
            diagonal_penalty_score(c)
        ), candidates))
    for v7,circ,morph,diag in results:
        v7_vals.append(v7)
        circ_vals.append(circ)
        morph_vals.append(morph)
        diag_vals.append(diag)
    return np.array(v7_vals), np.array(circ_vals), np.array(morph_vals), np.array(diag_vals)

# =========================
# 5️⃣ 최근 번호 패널티
# =========================
def recent_number_penalty_dual(candidates, numbers_arr, short_n=20, long_n=50, max_penalty_drop=0.35):
    penalties = []
    short_recent = numbers_arr[-short_n-1:-1]
    long_recent = numbers_arr[-long_n-1:-1]
    short_flat = short_recent.flatten()
    long_flat = long_recent.flatten()
    short_unique_ratio = len(set(short_flat)) / len(short_flat)
    long_unique_ratio = len(set(long_flat)) / len(long_flat)
    short_factor = 0.8 + 0.6 * short_unique_ratio
    long_factor = 0.8 + 0.6 * long_unique_ratio
    combined_factor = (short_factor * 0.6 + long_factor * 0.4)
    recent_set = set(long_flat)
    for comb in candidates:
        overlap_count = len(set(comb) & recent_set)
        penalty = 1 - min(overlap_count * 0.05 * combined_factor, max_penalty_drop)
        penalties.append(penalty)
    return np.array(penalties)

# =========================
# 6️⃣ 최종 조합 생성 (Fast, 번호대 균형 제거)
# =========================
def generate_final_combinations_fast(n_sets=10, focus_mode=False, ignore_group_balance=True):
    trans = build_transition_matrix(numbers_arr)
    last_draw = numbers_arr[-1]
    candidates_raw = monte_carlo_vectorized(trans, last_draw, trials=3500, focus_mode=focus_mode)
    counts = np.bincount(candidates_raw.flatten()-1, minlength=45)
    probs = counts / counts.sum()
    groups = divide_into_groups(probs, focus_mode)
    candidates = generate_group_combinations(groups, n_samples=5000)

    top_hot = np.argsort(-counts)[:5] + 1
    filtered = [c for c in candidates if len(set(c) & set(top_hot)) <= 2]
    candidates = filtered or candidates

    cand_arr = np.array(candidates)
    eff_vals = probs[cand_arr-1].sum(axis=1)
    v7_vals, circ_vals, morph_vals, diag_vals = evaluate_patterns_batch(candidates)
    combined_pattern = (v7_vals*0.45 + circ_vals*0.45 + diag_vals*0.1)
    rand_factor = np.random.uniform(0.95,1.05,len(eff_vals))
    recent_pen = recent_number_penalty_dual(candidates, numbers_arr, short_n=20, long_n=50)

    if not focus_mode:
        total_scores = (0.65*eff_vals + 0.35*(combined_pattern/50)) * rand_factor * recent_pen
    else:
        total_scores = (0.8*eff_vals + 0.2*(combined_pattern/50)) * rand_factor * recent_pen

    top_idx = np.argsort(-total_scores)[:n_sets]
    final_results=[]
    for idx in top_idx[:n_sets]:
        c = sorted(candidates[idx])
        final_results.append((c, float(eff_vals[idx]), float(v7_vals[idx]), float(circ_vals[idx]),
                              float(morph_vals[idx]), float(combined_pattern[idx]), float(total_scores[idx])))
    return final_results, probs

# =========================
# 6️⃣ 혼합형 생성
# =========================
def generate_final_combinations_mixed(n_sets=10, focus_mode=False, free_mode_ratio=0.4):
    trans = build_transition_matrix(numbers_arr)
    last_draw = numbers_arr[-1]
    candidates_raw = monte_carlo_vectorized(trans, last_draw, trials=3500, focus_mode=focus_mode)
    counts = np.bincount(candidates_raw.flatten()-1, minlength=45)
    probs = counts / counts.sum()
    groups = divide_into_groups(probs, focus_mode)
    
    candidates_bal = generate_group_combinations(groups, n_samples=5000)
    candidates_bal = [c for c in candidates_bal if morphological_pattern_score(c)!=0]

    n_free = int(len(candidates_bal)*free_mode_ratio)
    candidates_free = []
    while len(candidates_free) < n_free:
        comb = np.random.choice(np.arange(1,46), size=6, replace=False).tolist()
        if morphological_pattern_score(comb)!=0 and check_consecutive_rule(comb):
            candidates_free.append(comb)

    candidates = list({tuple(c): c for c in candidates_bal + candidates_free}.values())

    top_hot = np.argsort(-counts)[:5] + 1
    filtered = [c for c in candidates if len(set(c) & set(top_hot)) <= 2]
    candidates = filtered or candidates

    cand_arr = np.array(candidates)
    eff_vals = probs[cand_arr-1].sum(axis=1)
    v7_vals, circ_vals, morph_vals, diag_vals = evaluate_patterns_batch(candidates)
    combined_pattern = (v7_vals*0.45 + circ_vals*0.45 + diag_vals*0.1)
    rand_factor = np.random.uniform(0.95,1.05,len(eff_vals))
    recent_pen = recent_number_penalty_dual(candidates, numbers_arr, short_n=20, long_n=50)

    if not focus_mode:
        total_scores = (0.65*eff_vals + 0.35*(combined_pattern/50)) * rand_factor * recent_pen
    else:
        total_scores = (0.8*eff_vals + 0.2*(combined_pattern/50)) * rand_factor * recent_pen

    top_idx = np.argsort(-total_scores)[:n_sets]
    final_results=[]
    for idx in top_idx[:n_sets]:
        c = sorted(candidates[idx])
        final_results.append((c, float(eff_vals[idx]), float(v7_vals[idx]), float(circ_vals[idx]),
                              float(morph_vals[idx]), float(combined_pattern[idx]), float(total_scores[idx])))
    return final_results, probs

# =========================
# 7️⃣ 리포트 유틸
# =========================
def compute_historic_freq(numbers_array):
    flat=np.array(numbers_array).flatten()
    counts=np.bincount(flat-1, minlength=45)
    probs=counts/counts.sum()
    return counts, probs

def cooccurrence_matrix(numbers_array):
    mat=np.zeros((45,45),dtype=int)
    for draw in numbers_array:
        for a,b in itertools.combinations(draw,2):
            mat[a-1,b-1]+=1
            mat[b-1,a-1]+=1
    return mat

def combos_to_df(results_list,start_index=1,label="균형형"):
    rows=[]
    for idx,(comb,eff,v7,circ,morph,pat_comb,score) in enumerate(results_list,start=start_index):
        rows.append({
            "rank":idx,"type":label,"combo":comb,"eff":eff,
            "v7":v7,"circ":circ,"morph":morph,"pat":pat_comb,"score":score
        })
    return pd.DataFrame(rows)

# =========================
# 8️⃣ Streamlit UI
# =========================
if st.button("추천 번호 생성 & 분석 리포트"):
    with st.spinner("계산 중..."):
        t0 = time.time()
        res_mixed,_ = generate_final_combinations_mixed(10, focus_mode=False, free_mode_ratio=0.3)
        res_focus,_ = generate_final_combinations_fast(10, focus_mode=True)
        res_ignore_balance,_ = generate_final_combinations_fast(10, focus_mode=False, ignore_group_balance=True)
        t1 = time.time()

        # ----------------------------
        # 추천 결과 표시
        # ----------------------------
        st.subheader("✅ 혼합형 추천 10조합 (균형형+자유형)")
        for _,(comb,eff,v7,circ,morph,pat_comb,score) in enumerate(res_mixed,1):
            st.write(f"{comb} | 효율:{eff:.4f} | V7:{v7:.1f} | 원형:{circ:.1f} | 형태학:{morph:.1f} | 통합:{pat_comb:.1f} | 점수:{score:.4f}")

        st.subheader("🔥 집중형 추천 10조합")
        for _,(comb,eff,v7,circ,morph,pat_comb,score) in enumerate(res_focus,1):
            st.write(f"{comb} | 효율:{eff:.4f} | V7:{v7:.1f} | 원형:{circ:.1f} | 형태학:{morph:.1f} | 통합:{pat_comb:.1f} | 점수:{score:.4f}")

        st.subheader("🌟 번호군 균형 제외 추천 10조합")
        for _,(comb,eff,v7,circ,morph,pat_comb,score) in enumerate(res_ignore_balance,1):
            st.write(f"{comb} | 효율:{eff:.4f} | V7:{v7:.1f} | 원형:{circ:.1f} | 형태학:{morph:.1f} | 통합:{pat_comb:.1f} | 점수:{score:.4f}")

        st.write(f"계산 소요 시간: {t1-t0:.2f}초")

        # ----------------------------
        # 추천 결과 → DataFrame
        # ----------------------------
        df_mixed = combos_to_df(res_mixed, label="혼합형")
        df_focus = combos_to_df(res_focus, label="집중형")
        df_ignore = combos_to_df(res_ignore_balance, label="번호군 균형 제외")
        df_all = pd.concat([df_mixed, df_focus, df_ignore], ignore_index=True)
        st.subheader("📋 추천 결과 테이블")
        st.dataframe(df_all)

        # ----------------------------
        # 역대 데이터 통계
        # ----------------------------
        st.subheader("📊 번호별 출현 빈도")
        counts, probs = compute_historic_freq(numbers_arr)
        fig, ax = plt.subplots(figsize=(10,4))
        ax.bar(np.arange(1,46), counts, color='skyblue')
        ax.set_xlabel("번호")
        ax.set_ylabel("출현 횟수")
        ax.set_title("역대 로또 번호 출현 빈도")
        st.pyplot(fig)

        st.subheader("📊 공출현 히트맵 (최근 로또 번호 기반)")

        co_mat = cooccurrence_matrix(numbers_arr)
        fig, ax = plt.subplots(figsize=(14,12))
        cax = ax.matshow(co_mat, cmap='Reds')

        ax.set_xticks(np.arange(45))
        ax.set_yticks(np.arange(45))
        ax.set_xticklabels(np.arange(1,46))
        ax.set_yticklabels(np.arange(1,46))
        plt.setp(ax.get_xticklabels(), rotation=90)  # X축 라벨 회전

        plt.colorbar(cax)
        st.pyplot(fig)

