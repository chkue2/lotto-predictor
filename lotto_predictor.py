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
# 고정 파라미터 (슬라이더 제거 후 고정값)
# =========================
TRIALS = 10000                # Monte Carlo 샘플 수 (안정성↑, 속도 고려)
FOCUS_MODE_UI = False         # UI 전역 집중 모드 (기본 False)
RANDOM_PERTURB = 0.015        # 확률 퍼트버이션(과도한 흔들림 방지)
RECENT_PENALTY_FACTOR = 0.18  # 최근번호 패널티 강도(기본 0.2보다 살짝 완화)
INCLUDE_LAST = False          # 최근 1회까지 패널티에 포함 여부(False 권장)
FREE_MODE_RATIO = 0.30        # 혼합형에서 자유형 비율
HOT_K = 5                     # 상위 핫넘버 K
HOT_CAP = 2                   # 한 조합 내 허용 핫넘버 최대 개수

# =========================
# 설정 및 데이터 불러오기
# =========================
st.set_page_config(page_title="통합 로또 추천기 V14", layout="centered")
st.title("🎯 통합 로또 추천기 V14")

CSV_FILE = "lotto_data.csv"

@st.cache_data
def load_lotto_data_cached(file_path):
    df = pd.read_csv(file_path)
    df['numbers'] = df[[f"번호{i}" for i in range(1,7)]].values.tolist()
    return df

@st.cache_data
def build_transition_matrix_cached(numbers_arr, mtime):
    n = 45
    m = np.zeros((n, n), dtype=float)
    for i in range(len(numbers_arr) - 1):
        for a in numbers_arr[i]:
            for b in numbers_arr[i + 1]:
                m[a - 1, b - 1] += 1
    row_sums = m.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = m / row_sums
    return np.nan_to_num(p)

@st.cache_data
def historic_stats_cached(numbers_arr, mtime):
    flat = np.array(numbers_arr).flatten()
    counts = np.bincount(flat - 1, minlength=45)
    probs = counts / counts.sum()

    mat = np.zeros((45, 45), dtype=int)
    for draw in numbers_arr:
        for a, b in itertools.combinations(draw, 2):
            mat[a - 1, b - 1] += 1
            mat[b - 1, a - 1] += 1
    return (counts, probs), mat

def get_file_mtime(file_path):
    return os.path.getmtime(file_path)

# CSV 로드(예외 처리)
try:
    csv_mtime = get_file_mtime(CSV_FILE)
    df = load_lotto_data_cached(CSV_FILE)
except FileNotFoundError:
    st.error(f"CSV 파일을 찾을 수 없습니다: {CSV_FILE}")
    st.stop()
except KeyError as e:
    st.error(f"CSV 컬럼 확인 필요: {e}")
    st.stop()

numbers_arr = np.array(df['numbers'].tolist())

# =========================
# 1️⃣ 전이행렬 (레거시 함수: 유지만)
# =========================
def build_transition_matrix(numbers):
    n = 45
    m = np.zeros((n, n), dtype=float)
    for i in range(len(numbers) - 1):
        for a in numbers[i]:
            for b in numbers[i + 1]:
                m[a - 1, b - 1] += 1
    row_sums = m.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = m / row_sums
    return np.nan_to_num(p)

# =========================
# 2️⃣ Monte Carlo 시뮬레이션
# =========================
def apply_recent_draw_penalty(probs_base, last_draw, penalty_factor=0.2):
    probs = probs_base.copy()
    for n in last_draw:
        probs[n - 1] *= penalty_factor
    probs /= probs.sum()
    return probs

def monte_carlo_vectorized(
    trans_matrix,
    last_draw,
    trials=3000,
    focus_mode=False,
    random_perturb=0.02,
    recent_penalty=True,
    recent_penalty_factor=0.2
):
    probs_base = trans_matrix[[n - 1 for n in last_draw]].sum(0)
    probs_base = np.maximum(probs_base, 0.01)

    if focus_mode:
        probs_base = probs_base ** 2

    if recent_penalty:
        probs_base = apply_recent_draw_penalty(probs_base, last_draw, penalty_factor=recent_penalty_factor)

    perturb = np.random.uniform(-random_perturb, random_perturb, size=probs_base.shape)
    probs_base += perturb
    probs_base = np.clip(probs_base, 0.001, None)
    probs_base /= probs_base.sum()

    draws = np.array([
        np.random.choice(np.arange(1, 46), size=6, replace=False, p=probs_base)
        for _ in range(trials)
    ])
    return draws

# =========================
# 3️⃣ 그룹 분할 및 후보 생성
# =========================
def divide_into_groups(probabilities, focus_mode=False):
    sorted_idx = np.argsort(-probabilities)
    if focus_mode:
        g1 = sorted_idx[:10] + 1
        g2 = sorted_idx[10:25] + 1
        g3 = sorted_idx[25:] + 1
    else:
        g1 = sorted_idx[:15] + 1
        g2 = sorted_idx[15:30] + 1
        g3 = sorted_idx[30:] + 1
    return g1.tolist(), g2.tolist(), g3.tolist()

def check_consecutive_rule(comb):
    comb = sorted(comb)
    groups = []
    cur = [comb[0]]
    for i in range(1, len(comb)):
        if comb[i] == comb[i - 1] + 1:
            cur.append(comb[i])
        else:
            if len(cur) > 1:
                groups.append(cur)
            cur = [comb[i]]
    if len(cur) > 1:
        groups.append(cur)
    if len(groups) > 1 or any(len(g) > 2 for g in groups):
        return False
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
GRID_POS = {v: (r, c) for r, row in enumerate(lotto_grid) for c, v in enumerate(row)}

def is_strict_diagonal(comb):
    coords = [GRID_POS[n] for n in comb]
    coords_sorted = sorted(coords, key=lambda x: (x[0], x[1]))
    drs = [coords_sorted[i + 1][0] - coords_sorted[i][0] for i in range(len(coords_sorted) - 1)]
    dcs = [coords_sorted[i + 1][1] - coords_sorted[i][1] for i in range(len(coords_sorted) - 1)]
    return len(set(drs)) == 1 and len(set(dcs)) == 1

def diagonal_penalty_score(comb):
    coords = [GRID_POS[n] for n in comb]
    diffs = [(coords[i + 1][0] - coords[i][0], coords[i + 1][1] - coords[i][1]) for i in range(len(coords) - 1)]
    penalty = 0
    for dr, dc in diffs:
        if abs(dr) == abs(dc) and abs(dr) >= 1:
            penalty += 1
    return max(0, 20 - penalty * 5)

def sample_with_quotas(g0, g1, g2, quotas):
    a, b, c = quotas
    if len(g0) < a or len(g1) < b or len(g2) < c:
        return None
    pick = []
    pick += np.random.choice(g0, size=a, replace=False).tolist()
    pick += np.random.choice(g1, size=b, replace=False).tolist()
    pick += np.random.choice(g2, size=c, replace=False).tolist()
    return sorted(pick)

def generate_group_combinations(groups, n_samples=10000, use_balance=True, quota_patterns=None):
    g0, g1, g2 = groups
    candidates = []
    if use_balance:
        if not quota_patterns:
            quota_patterns = [(1,2,3), (1,3,2), (2,1,3), (2,2,2), (2,3,1), (3,2,1), (3,1,2)]
        attempts = 0
        max_attempts = n_samples * 20
        while len(candidates) < n_samples and attempts < max_attempts:
            attempts += 1
            quotas = quota_patterns[np.random.randint(len(quota_patterns))]
            comb = sample_with_quotas(g0, g1, g2, quotas)
            if not comb:
                continue
            if not check_consecutive_rule(comb):
                continue
            if is_strict_diagonal(comb) and np.random.rand() >= 0.1:
                continue
            if morphological_pattern_score(comb) == 0:
                continue
            candidates.append(comb)
    else:
        g_all = list(set(g0 + g1 + g2))
        attempts = 0
        max_attempts = n_samples * 20
        while len(candidates) < n_samples and attempts < max_attempts:
            attempts += 1
            comb = np.random.choice(g_all, size=6, replace=False).tolist()
            if not check_consecutive_rule(comb):
                continue
            if is_strict_diagonal(comb) and np.random.rand() >= 0.1:
                continue
            if morphological_pattern_score(comb) == 0:
                continue
            candidates.append(sorted(comb))
    candidates = list({tuple(c): c for c in candidates}.values())
    return candidates

# =========================
# 4️⃣ 패턴 점수
# =========================
def gianella_pattern_v7(numbers):
    coords = [GRID_POS[n] for n in numbers]
    rows = [0]*7; cols = [0]*7
    for r, c in coords:
        rows[r] += 1; cols[c] += 1
    row_penalty = sum(max(0, x - 2)**2 for x in rows)
    col_penalty = sum(max(0, x - 2)**2 for x in cols)
    balance_score = 50 - (row_penalty + col_penalty)
    diag1 = sum(r == c for r, c in coords)
    diag2 = sum(c == 6 - r for r, c in coords)
    return balance_score + diag1 + diag2

def gianella_pattern_circular(numbers):
    zones = {1: range(1,8), 2: range(8,15), 3: range(15,22),
             4: range(22,29), 5: range(29,36), 6: range(36,43), 7: range(43,46)}
    counts = {z: 0 for z in zones}
    for n in numbers:
        for z, rng in zones.items():
            if n in rng:
                counts[z] += 1
                break
    diversity_bonus = sum(1 for v in counts.values() if v == 1)
    overlap_penalty = sum(max(0, v - 2) for v in counts.values())
    return max(0, min(70, 40 + diversity_bonus * 2.5 - overlap_penalty))

def morphological_pattern_score(numbers):
    pos = [GRID_POS[n] for n in numbers]
    pos_set = set(pos)
    for dr, dc in [(1,1), (1,-1)]:
        for r, c in pos:
            chain = 1
            nr, nc = r + dr, c + dc
            while (nr, nc) in pos_set:
                chain += 1
                nr += dr; nc += dc
            if chain >= 4:
                return 0
    return 20

# =========================
# 5️⃣ 병렬화 평가
# =========================
def evaluate_patterns_batch(candidates):
    v7_vals = []; circ_vals = []; morph_vals = []; diag_vals = []
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda c: (
            gianella_pattern_v7(c),
            gianella_pattern_circular(c),
            morphological_pattern_score(c),
            diagonal_penalty_score(c)
        ), candidates))
    for v7, circ, morph, diag in results:
        v7_vals.append(v7)
        circ_vals.append(circ)
        morph_vals.append(morph)
        diag_vals.append(diag)
    return np.array(v7_vals), np.array(circ_vals), np.array(morph_vals), np.array(diag_vals)

# =========================
# 5️⃣ 최근 번호 패널티
# =========================
def recent_number_penalty_dual(
    candidates,
    numbers_arr,
    short_n=20,
    long_n=50,
    include_last_draw=False,
    max_penalty_drop=0.35
):
    end = None if include_last_draw else -1
    short_recent = numbers_arr[-short_n-1:end]
    long_recent = numbers_arr[-long_n-1:end]
    short_flat = short_recent.flatten()
    long_flat = long_recent.flatten()
    short_unique_ratio = len(set(short_flat)) / len(short_flat) if len(short_flat) else 1.0
    long_unique_ratio = len(set(long_flat)) / len(long_flat) if len(long_flat) else 1.0
    short_factor = 0.8 + 0.6 * short_unique_ratio
    long_factor = 0.8 + 0.6 * long_unique_ratio
    combined_factor = (short_factor * 0.6 + long_factor * 0.4)
    recent_set = set(long_flat)
    penalties = []
    for comb in candidates:
        overlap_count = len(set(comb) & recent_set)
        penalty = 1 - min(overlap_count * 0.05 * combined_factor, max_penalty_drop)
        penalties.append(penalty)
    return np.array(penalties)

# =========================
# 6️⃣ 최종 조합 생성 (Fast, 번호군 균형 옵션 반영)
# =========================
def generate_final_combinations_fast(n_sets=10, focus_mode=False, ignore_group_balance=True):
    trans = build_transition_matrix_cached(numbers_arr, csv_mtime)
    last_draw = numbers_arr[-1]
    candidates_raw = monte_carlo_vectorized(
        trans, last_draw,
        trials=TRIALS,
        focus_mode=focus_mode or FOCUS_MODE_UI,
        random_perturb=RANDOM_PERTURB,
        recent_penalty=True,
        recent_penalty_factor=RECENT_PENALTY_FACTOR
    )
    counts = np.bincount(candidates_raw.flatten() - 1, minlength=45)
    probs = counts / counts.sum()
    groups = divide_into_groups(probs, focus_mode or FOCUS_MODE_UI)

    candidates = generate_group_combinations(
        groups,
        use_balance=(not ignore_group_balance),
        quota_patterns=[(2,2,2), (3,2,1), (2,3,1)]
    )

    top_hot = np.argsort(-counts)[:HOT_K] + 1
    filtered = [c for c in candidates if len(set(c) & set(top_hot)) <= HOT_CAP]
    candidates = filtered or candidates

    cand_arr = np.array(candidates)
    eff_vals = probs[cand_arr - 1].sum(axis=1)
    v7_vals, circ_vals, morph_vals, diag_vals = evaluate_patterns_batch(candidates)
    combined_pattern = (v7_vals * 0.45 + circ_vals * 0.45 + diag_vals * 0.1)
    rand_factor = np.random.uniform(0.95, 1.05, len(eff_vals))
    recent_pen = recent_number_penalty_dual(
        candidates, numbers_arr,
        short_n=20, long_n=50,
        include_last_draw=INCLUDE_LAST
    )

    if not (focus_mode or FOCUS_MODE_UI):
        total_scores = (0.65 * eff_vals + 0.35 * (combined_pattern / 50)) * rand_factor * recent_pen
    else:
        total_scores = (0.8 * eff_vals + 0.2 * (combined_pattern / 50)) * rand_factor * recent_pen

    top_idx = np.argsort(-total_scores)[:n_sets]
    final_results = []
    for idx in top_idx[:n_sets]:
        c = sorted(candidates[idx])
        final_results.append((
            c,
            float(eff_vals[idx]),
            float(v7_vals[idx]),
            float(circ_vals[idx]),
            float(morph_vals[idx]),
            float(combined_pattern[idx]),
            float(total_scores[idx])
        ))
    return final_results, probs

# =========================
# 6️⃣ 혼합형 생성 (균형형 + 비균형형 혼합)
# =========================
def generate_final_combinations_mixed(n_sets=10, focus_mode=False, free_mode_ratio=None):
    if free_mode_ratio is None:
        free_mode_ratio = FREE_MODE_RATIO

    trans = build_transition_matrix_cached(numbers_arr, csv_mtime)
    last_draw = numbers_arr[-1]
    candidates_raw = monte_carlo_vectorized(
        trans, last_draw,
        trials=TRIALS,
        focus_mode=focus_mode or FOCUS_MODE_UI,
        random_perturb=RANDOM_PERTURB,
        recent_penalty=True,
        recent_penalty_factor=RECENT_PENALTY_FACTOR
    )
    counts = np.bincount(candidates_raw.flatten() - 1, minlength=45)
    probs = counts / counts.sum()
    groups = divide_into_groups(probs, focus_mode or FOCUS_MODE_UI)

    candidates_bal = generate_group_combinations(
        groups,
        use_balance=True,
        quota_patterns=[(2,2,2), (3,2,1), (2,3,1)]
    )

    n_free = int(len(candidates_bal) * free_mode_ratio)
    if n_free <= 0:
        n_free = 1
    candidates_free = generate_group_combinations(
        groups,
        n_samples=max(1, n_free),
        use_balance=False
    )

    candidates = list({tuple(c): c for c in (candidates_bal + candidates_free)}.values())

    top_hot = np.argsort(-counts)[:HOT_K] + 1
    filtered = [c for c in candidates if len(set(c) & set(top_hot)) <= HOT_CAP]
    candidates = filtered or candidates

    cand_arr = np.array(candidates)
    eff_vals = probs[cand_arr - 1].sum(axis=1)
    v7_vals, circ_vals, morph_vals, diag_vals = evaluate_patterns_batch(candidates)
    combined_pattern = (v7_vals * 0.45 + circ_vals * 0.45 + diag_vals * 0.1)
    rand_factor = np.random.uniform(0.95, 1.05, len(eff_vals))
    recent_pen = recent_number_penalty_dual(
        candidates, numbers_arr,
        short_n=20, long_n=50,
        include_last_draw=INCLUDE_LAST
    )

    if not (focus_mode or FOCUS_MODE_UI):
        total_scores = (0.65 * eff_vals + 0.35 * (combined_pattern / 50)) * rand_factor * recent_pen
    else:
        total_scores = (0.8 * eff_vals + 0.2 * (combined_pattern / 50)) * rand_factor * recent_pen

    top_idx = np.argsort(-total_scores)[:n_sets]
    final_results = []
    for idx in top_idx[:n_sets]:
        c = sorted(candidates[idx])
        final_results.append((
            c,
            float(eff_vals[idx]),
            float(v7_vals[idx]),
            float(circ_vals[idx]),
            float(morph_vals[idx]),
            float(combined_pattern[idx]),
            float(total_scores[idx])
        ))
    return final_results, probs

# =========================
# 7️⃣ 리포트 유틸
# =========================
def compute_historic_freq(numbers_array):
    flat = np.array(numbers_array).flatten()
    counts = np.bincount(flat - 1, minlength=45)
    probs = counts / counts.sum()
    return counts, probs

def cooccurrence_matrix(numbers_array):
    mat = np.zeros((45, 45), dtype=int)
    for draw in numbers_array:
        for a, b in itertools.combinations(draw, 2):
            mat[a - 1, b - 1] += 1
            mat[b - 1, a - 1] += 1
    return mat

def combos_to_df(results_list, start_index=1, label="균형형"):
    rows = []
    for idx, (comb, eff, v7, circ, morph, pat_comb, score) in enumerate(results_list, start=start_index):
        rows.append({
            "rank": idx, "type": label, "combo": comb, "eff": eff,
            "v7": v7, "circ": circ, "morph": morph, "pat": pat_comb, "score": score
        })
    return pd.DataFrame(rows)

# =========================
# 8️⃣ Streamlit UI
# =========================
if st.button("추천 번호 생성 & 분석 리포트"):
    with st.spinner("계산 중..."):
        t0 = time.time()
        res_mixed, _ = generate_final_combinations_mixed(10, focus_mode=False, free_mode_ratio=FREE_MODE_RATIO)
        res_focus, _ = generate_final_combinations_fast(10, focus_mode=True, ignore_group_balance=False)   # 균형형
        res_ignore_balance, _ = generate_final_combinations_fast(10, focus_mode=False, ignore_group_balance=True)  # 비균형형
        t1 = time.time()

        # ----------------------------
        # 추천 결과 표시
        # ----------------------------
        st.subheader("✅ 혼합형 추천 10조합 (균형형+자유형)")
        for _, (comb, eff, v7, circ, morph, pat_comb, score) in enumerate(res_mixed, 1):
            st.write(f"{comb} | 효율:{eff:.4f} | V7:{v7:.1f} | 원형:{circ:.1f} | 형태학:{morph:.1f} | 통합:{pat_comb:.1f} | 점수:{score:.4f}")

        st.subheader("🔥 집중형 추천 10조합 (균형형)")
        for _, (comb, eff, v7, circ, morph, pat_comb, score) in enumerate(res_focus, 1):
            st.write(f"{comb} | 효율:{eff:.4f} | V7:{v7:.1f} | 원형:{circ:.1f} | 형태학:{morph:.1f} | 통합:{pat_comb:.1f} | 점수:{score:.4f}")

        st.subheader("🌟 번호군 균형 제외 추천 10조합 (비균형형)")
        for _, (comb, eff, v7, circ, morph, pat_comb, score) in enumerate(res_ignore_balance, 1):
            st.write(f"{comb} | 효율:{eff:.4f} | V7:{v7:.1f} | 원형:{circ:.1f} | 형태학:{morph:.1f} | 통합:{pat_comb:.1f} | 점수:{score:.4f}")

        st.write(f"계산 소요 시간: {t1 - t0:.2f}초")

        # ----------------------------
        # 추천 결과 → DataFrame
        # ----------------------------
        df_mixed = combos_to_df(res_mixed, label="혼합형")
        df_focus = combos_to_df(res_focus, label="집중형(균형)")
        df_ignore = combos_to_df(res_ignore_balance, label="번호군 균형 제외(비균형)")
        df_all = pd.concat([df_mixed, df_focus, df_ignore], ignore_index=True)
        st.subheader("📋 추천 결과 테이블")
        st.caption("테이블 헤더를 클릭해 정렬/필터를 시도해보세요. (예: score 내림차순)")
        st.dataframe(df_all)

        # ----------------------------
        # 역대 데이터 통계 (캐시 활용)
        # ----------------------------
        st.subheader("📊 번호별 출현 빈도")
        (counts, probs_hist), co_mat = historic_stats_cached(numbers_arr, csv_mtime)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(np.arange(1, 46), counts, color='skyblue')
        ax.set_xlabel("번호")
        ax.set_ylabel("출현 횟수")
        ax.set_title("역대 로또 번호 출현 빈도")
        st.pyplot(fig)

        st.subheader("📊 공출현 히트맵 (최근 로또 번호 기반)")
        fig, ax = plt.subplots(figsize=(14, 12))
        cax = ax.matshow(co_mat, cmap='Reds')
        ax.set_xticks(np.arange(45))
        ax.set_yticks(np.arange(45))
        ax.set_xticklabels(np.arange(1, 46))
        ax.set_yticklabels(np.arange(1, 46))
        plt.setp(ax.get_xticklabels(), rotation=90)
        plt.colorbar(cax)
        st.pyplot(fig)
