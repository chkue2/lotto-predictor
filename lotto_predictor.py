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
from pattern_sets import PATTERN_SETS

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
TRIALS = 10000
FOCUS_MODE_UI = False
RANDOM_PERTURB = 0.003
RECENT_PENALTY_FACTOR = 0.18
INCLUDE_LAST = False
FREE_MODE_RATIO = 0.30
HOT_K = 5
HOT_CAP = 3

MCS_RUNS_A = 1_000_000
MCS_RUNS_B = 1_000_000

PORTFOLIO_BLEND = 0.15
BUCKET_TOP_K = 3
REUSE_PENALTY_WEIGHT = 0.06
TOP_M = 8
MIN_FROM_TOP_M = 2

# =========================
# 설정 및 데이터 불러오기
# =========================
st.set_page_config(page_title="통합 로또 추천기 V14", layout="centered")
st.title("🎯 통합 로또 추천기 V14")

CSV_FILE = "lotto_data.csv"
rng = np.random.default_rng()

@st.cache_data
def load_lotto_data_cached(file_path, mtime):
    df = pd.read_csv(file_path)
    if '회차' in df.columns:
        df = df.rename(columns={'회차': 'round'})
    df['round'] = pd.to_numeric(df['round'], errors='coerce')
    for i in range(1, 7):
        df[f'번호{i}'] = pd.to_numeric(df.get(f'번호{i}'), errors='coerce')
    if '보너스번호' in df.columns:
        df['보너스번호'] = pd.to_numeric(df['보너스번호'], errors='coerce')
    df = df.dropna(subset=['round']).sort_values('round').reset_index(drop=True)
    df['numbers'] = df[[f"번호{i}" for i in range(1,7)]].values.tolist()
    return df

@st.cache_data
def build_transition_matrix_cached(numbers_arr, mtime):
    n = 45
    m = np.zeros((n, n), dtype=float)
    freq = np.zeros(n, dtype=float)
    for i in range(len(numbers_arr) - 1):
        prev = numbers_arr[i]
        nxt  = numbers_arr[i + 1]
        for a in prev:
            freq[a - 1] += 1
            for b in nxt:
                m[a - 1, b - 1] += 1
    alpha = 1e-2
    m = m + alpha
    row_sums = m.sum(axis=1, keepdims=True)
    T = m / row_sums
    return np.nan_to_num(T), freq

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

try:
    csv_mtime = get_file_mtime(CSV_FILE)
    df = load_lotto_data_cached(CSV_FILE, csv_mtime)
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
    alpha = 1e-2
    m = m + alpha
    row_sums = m.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = m / row_sums
    return np.nan_to_num(p)

# =========================
# 2️⃣ Monte Carlo 시뮬레이션
# =========================
def apply_recent_draw_penalty_soft(probs_base, last_draw, penalty_factor=0.2):
    probs = probs_base.copy()
    idx = [n-1 for n in last_draw]
    before = probs[idx].sum()
    probs[idx] *= penalty_factor
    delta = before - probs[idx].sum()
    mask = np.ones_like(probs, dtype=bool); mask[idx] = False
    if probs[mask].sum() > 0:
        probs[mask] += delta * (probs[mask] / probs[mask].sum())
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
    probs_base = np.clip(probs_base, 1e-6, None)
    probs_base /= probs_base.sum()

    if focus_mode:
        probs_base = np.power(probs_base, 2.0)
        probs_base /= probs_base.sum()

    if recent_penalty:
        probs_base = apply_recent_draw_penalty_soft(probs_base, last_draw, penalty_factor=recent_penalty_factor)

    perturb = rng.uniform(-random_perturb, random_perturb, size=probs_base.shape)
    probs_base = np.clip(probs_base + perturb, 1e-6, None)
    probs_base /= probs_base.sum()

    draws = np.array([
        rng.choice(np.arange(1, 46), size=6, replace=False, p=probs_base)
        for _ in range(trials)
    ])
    return draws

# =========================
# (신규) 2-1~2-4: 마르코프 기반 "이중" 대규모 몬테카를로
# =========================
def dual_monte_carlo_next_number_probs(
    numbers_arr,
    T_global,
    freq_global,
    runs_a=MCS_RUNS_A,
    runs_b=MCS_RUNS_B,
    recent_penalty_factor=RECENT_PENALTY_FACTOR,
    include_last_draw=INCLUDE_LAST
):
    last_draw = numbers_arr[-1]

    # --- MC-A ---
    pA = T_global[[n-1 for n in last_draw]].sum(axis=0)
    pA = np.clip(pA, 1e-9, None)
    pA /= pA.sum()
    pA = apply_recent_draw_penalty_soft(pA, last_draw, penalty_factor=recent_penalty_factor)
    counts_a = rng.multinomial(runs_a, pA)
    estA = counts_a / counts_a.sum()

    # --- MC-B ---
    prior_freq = freq_global + 1e-3
    prior_freq /= prior_freq.sum()
    mean_row = T_global.mean(axis=0); mean_row /= mean_row.sum()
    mix = 0.5 * prior_freq + 0.5 * mean_row

    if not include_last_draw and len(numbers_arr) >= 2:
        prev_prev = numbers_arr[-2]
        q = T_global[[n-1 for n in prev_prev]].sum(axis=0)
        q = np.clip(q, 1e-9, None); q /= q.sum()
        pB = 0.6 * mix + 0.4 * q
    else:
        pB = mix.copy()

    pB = np.clip(pB, 1e-9, None); pB /= pB.sum()
    counts_b = rng.multinomial(runs_b, pB)
    estB = counts_b / counts_b.sum()

    p_next = 0.75 * estA + 0.25 * estB
    p_next = np.clip(p_next, 1e-12, None)
    p_next /= p_next.sum()

    transition_vectors = {n: T_global[n-1].copy() for n in range(1, 46)}
    return p_next, {"A": estA, "B": estB}, transition_vectors

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
    if len(groups) > 2 or any(len(g) > 3 for g in groups):
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
    return (len(set(drs)) == 1 and len(set(dcs)) == 1 and abs(drs[0]) >= 1 and abs(dcs[0]) >= 1)

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
    pick += rng.choice(g0, size=a, replace=False).tolist()
    pick += rng.choice(g1, size=b, replace=False).tolist()
    pick += rng.choice(g2, size=c, replace=False).tolist()
    return sorted(pick)

def generate_group_combinations(groups, n_samples=10000, use_balance=True, quota_patterns=None):
    g0, g1, g2 = groups
    candidates = []
    seen = set()
    if use_balance:
        if not quota_patterns:
            quota_patterns = [(2,2,2), (3,2,1), (2,3,1)]
        attempts = 0
        max_attempts = n_samples * 25
        while len(candidates) < n_samples and attempts < max_attempts:
            attempts += 1
            quotas = quota_patterns[rng.integers(len(quota_patterns))]
            comb = sample_with_quotas(g0, g1, g2, quotas)
            if not comb:
                continue
            if not check_consecutive_rule(comb):
                continue
            if is_strict_diagonal(comb) and rng.random() >= 0.1:
                continue
            if morphological_pattern_score(comb) == 0:
                continue
            t = tuple(comb)
            if t in seen:
                continue
            seen.add(t)
            candidates.append(comb)
    else:
        g_all = list(set(g0 + g1 + g2))
        attempts = 0
        max_attempts = n_samples * 25
        while len(candidates) < n_samples and attempts < max_attempts:
            attempts += 1
            comb = rng.choice(g_all, size=6, replace=False).tolist()
            if not check_consecutive_rule(comb):
                continue
            if is_strict_diagonal(comb) and rng.random() >= 0.1:
                continue
            if morphological_pattern_score(comb) == 0:
                continue
            t = tuple(sorted(comb))
            if t in seen:
                continue
            seen.add(t)
            candidates.append(sorted(comb))
    return candidates

# =========================
# 4️⃣ 패턴 점수(지아넬라 응용 + 다차원 버킷)
# =========================
def gianella_pattern_v7(numbers):
    coords = [GRID_POS[n] for n in numbers]
    rows = [0]*7; cols = [0]*7
    for r, c in coords:
        rows[r] += 1; cols[c] += 1
    row_penalty = sum(max(0, x - 3)**2 for x in rows)
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
        for z, rng_zone in zones.items():
            if n in rng_zone:
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

def pattern_bucket_key(numbers):
    coords = [GRID_POS[n] for n in numbers]
    rows = [0]*7; cols = [0]*7
    for r,c in coords:
        rows[r] += 1
        if c < 7: cols[c] += 1
    diag_main = sum(r == c for r,c in coords)
    diag_anti = sum((c == 6 - r) for r,c in coords if c < 7)
    return (tuple(rows), tuple(cols), diag_main, diag_anti)

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
    total_rows = len(numbers_arr)
    short_n = min(short_n, max(1, total_rows-1))
    long_n  = min(long_n,  max(1, total_rows-1))
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

# -------- (4) Top-m 강제 포함 체크 --------
def passes_topm_constraint(comb, probs, top_m=TOP_M, min_from_top=MIN_FROM_TOP_M):
    top_idx = np.argsort(-probs)[:top_m] + 1  # 번호(1~45)
    return (len(set(comb) & set(top_idx)) >= min_from_top)

# -------- (5) 점수용 정규화 유틸 --------
def minmax_norm(x, eps=1e-12):
    x = np.asarray(x, dtype=float)
    mn, mx = np.min(x), np.max(x)
    return (x - mn) / (mx - mn + eps)

# =========================
# (A) 번호 재사용 페널티 기반 그리디 선택
# =========================
def select_with_reuse_penalty(candidates, base_scores, n_sets):
    selected_idx = []
    used_counts = np.zeros(46, dtype=int)  # 1~45
    remaining = set(range(len(candidates)))

    while len(selected_idx) < n_sets and remaining:
        best_idx = None
        best_score = -1.0
        for i in list(remaining):
            comb = candidates[i]
            reuse = sum(used_counts[n] for n in comb)
            penalty = 1.0 / (1.0 + REUSE_PENALTY_WEIGHT * reuse)
            score_adj = base_scores[i] * penalty
            if score_adj > best_score:
                best_score = score_adj
                best_idx = i
        if best_idx is None:
            break
        selected_idx.append(best_idx)
        for n in candidates[best_idx]:
            used_counts[n] += 1
        remaining.remove(best_idx)

    return selected_idx

# =========================
# 6️⃣ 최종 조합 생성 (Fast, 번호군 균형 옵션 반영)
# =========================
def generate_final_combinations_fast(n_sets=10, focus_mode=False, ignore_group_balance=True):
    T, freq = build_transition_matrix_cached(numbers_arr, csv_mtime)

    next_probs, mc_parts, transition_vectors = dual_monte_carlo_next_number_probs(
        numbers_arr, T, freq,
        runs_a=MCS_RUNS_A, runs_b=MCS_RUNS_B,
        recent_penalty_factor=RECENT_PENALTY_FACTOR,
        include_last_draw=INCLUDE_LAST
    )

    last_draw = numbers_arr[-1]
    candidates_raw = monte_carlo_vectorized(
        T, last_draw,
        trials=TRIALS,
        focus_mode=focus_mode or FOCUS_MODE_UI,
        random_perturb=RANDOM_PERTURB,
        recent_penalty=True,
        recent_penalty_factor=RECENT_PENALTY_FACTOR
    )

    counts = np.bincount(candidates_raw.flatten() - 1, minlength=45)
    probs = next_probs.copy()
    groups = divide_into_groups(probs, focus_mode or FOCUS_MODE_UI)

    candidates = generate_group_combinations(
        groups,
        use_balance=(not ignore_group_balance),
        quota_patterns=[(2,2,2), (3,2,1), (2,3,1)]
    )

    top_hot = np.argsort(-probs)[:HOT_K] + 1
    hot_prob_sum = probs[top_hot-1].sum()
    cap = HOT_CAP
    if hot_prob_sum > 0.20:
        cap = max(1, HOT_CAP-1)
    elif hot_prob_sum < 0.12:
        cap = min(3, HOT_CAP+1)
    if focus_mode or FOCUS_MODE_UI:
        cap = max(1, cap-1)

    filtered = [c for c in candidates if len(set(c) & set(top_hot)) <= cap]
    candidates = filtered or candidates

    filtered2 = [c for c in candidates if passes_topm_constraint(c, probs, TOP_M, MIN_FROM_TOP_M)]
    candidates = filtered2 or candidates

    cand_arr = np.array(candidates)
    eff_linear = probs[cand_arr - 1].sum(axis=1)
    logp = np.log(probs + 1e-12)
    eff_log = logp[cand_arr - 1].sum(axis=1)
    eff_log_n = minmax_norm(eff_log)

    v7_vals, circ_vals, morph_vals, diag_vals = evaluate_patterns_batch(candidates)
    combined_pattern = (v7_vals*0.42 + circ_vals*0.42 + diag_vals*0.11 + (morph_vals/20.0)*0.05)
    pat_n = minmax_norm(combined_pattern)

    recent_pen = recent_number_penalty_dual(
        candidates, numbers_arr,
        short_n=20, long_n=50,
        include_last_draw=INCLUDE_LAST
    )
    rand_factor = rng.uniform(0.95, 1.05, len(eff_log_n))

    base_score = 0.85 * eff_log_n + 0.15 * pat_n
    total_scores = base_score * rand_factor * recent_pen

    final_sel = select_with_reuse_penalty(candidates, total_scores, n_sets)

    final_results = []
    for idx in final_sel:
        c = sorted(candidates[idx])
        final_results.append((
            c,
            float(eff_linear[idx]),
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

    T, freq = build_transition_matrix_cached(numbers_arr, csv_mtime)
    next_probs, mc_parts, transition_vectors = dual_monte_carlo_next_number_probs(
        numbers_arr, T, freq,
        runs_a=MCS_RUNS_A, runs_b=MCS_RUNS_B,
        recent_penalty_factor=RECENT_PENALTY_FACTOR,
        include_last_draw=INCLUDE_LAST
    )
    last_draw = numbers_arr[-1]
    candidates_raw = monte_carlo_vectorized(
        T, last_draw,
        trials=TRIALS,
        focus_mode=focus_mode or FOCUS_MODE_UI,
        random_perturb=RANDOM_PERTURB,
        recent_penalty=True,
        recent_penalty_factor=RECENT_PENALTY_FACTOR
    )

    counts = np.bincount(candidates_raw.flatten() - 1, minlength=45)
    probs = next_probs.copy()
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

    top_hot = np.argsort(-probs)[:HOT_K] + 1
    hot_prob_sum = probs[top_hot-1].sum()
    cap = HOT_CAP
    if hot_prob_sum > 0.20:
        cap = max(1, HOT_CAP-1)
    elif hot_prob_sum < 0.12:
        cap = min(3, HOT_CAP+1)

    filtered = [c for c in candidates if len(set(c) & set(top_hot)) <= cap]
    candidates = filtered or candidates

    filtered2 = [c for c in candidates if passes_topm_constraint(c, probs, TOP_M, MIN_FROM_TOP_M)]
    candidates = filtered2 or candidates

    cand_arr = np.array(candidates)
    eff_linear = probs[cand_arr - 1].sum(axis=1)
    logp = np.log(probs + 1e-12)
    eff_log = logp[cand_arr - 1].sum(axis=1)
    eff_log_n = minmax_norm(eff_log)

    v7_vals, circ_vals, morph_vals, diag_vals = evaluate_patterns_batch(candidates)
    combined_pattern = (v7_vals*0.42 + circ_vals*0.42 + diag_vals*0.11 + (morph_vals/20.0)*0.05)
    pat_n = minmax_norm(combined_pattern)

    recent_pen = recent_number_penalty_dual(
        candidates, numbers_arr,
        short_n=20, long_n=50,
        include_last_draw=INCLUDE_LAST
    )
    rand_factor = rng.uniform(0.95, 1.05, len(eff_log_n))

    base_score = 0.85 * eff_log_n + 0.15 * pat_n
    total_scores = base_score * rand_factor * recent_pen

    final_sel = select_with_reuse_penalty(candidates, total_scores, n_sets)

    final_results = []
    for idx in final_sel:
        c = sorted(candidates[idx])
        final_results.append((
            c,
            float(eff_linear[idx]),
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
# 9️⃣ PATTERN_SETS 기반 최근 2단계 전이 영향 → 1200회에 적용해 1201회 예측 (★ 변경)
# =========================
from collections import Counter

def build_group_maps(PATTERN_SETS):
    group_maps = {}
    groups_by_set = {}
    for sid, groups in PATTERN_SETS.items():
        num2grp = {}
        for gi, triplet in enumerate(groups):
            for n in triplet:
                num2grp[n] = gi
        group_maps[sid] = num2grp
        groups_by_set[sid] = groups
    return group_maps, groups_by_set

def pattern_signature(draw_numbers, num2grp_map):
    return [num2grp_map[n] for n in draw_numbers]

def one_hot_group_counts(sig):
    v = np.zeros(15, dtype=float)
    for g in sig:
        v[g] += 1.0
    return v

def recent_pairwise_transition_predict(
    numbers_arr,
    PATTERN_SETS,
    penalty_last_draw=0.55,
    n_sets=10
):
    """
    최근 3회: D1(=t-2), D2(=t-1), D3(=t)
    전이: D1→D2, D2→D3 의 그룹 전이 행렬을 합산(가중)하여
    D3의 그룹 존재벡터에 곱해 D4(=t+1) 그룹 점수를 추정.
    이후 그룹 점수를 번호 확률로 분배하여 6개 조합 10개 샘플링.
    """
    assert len(numbers_arr) >= 3, "최소 3회의 데이터가 필요합니다."
    D1, D2, D3 = numbers_arr[-3], numbers_arr[-2], numbers_arr[-1]

    group_maps, groups_by_set = build_group_maps(PATTERN_SETS)

    # --- (추가) 그룹 내 번호 가중용: 역사적 빈도 ---
    hist_counts, _ = compute_historic_freq(numbers_arr)
    hist_counts = hist_counts.astype(float)

    # 패턴셋별 그룹 전이 행렬(15x15) 구성
    total_group_scores = np.zeros(15, dtype=float)

    for sid, num2grp in group_maps.items():
        sig1 = pattern_signature(D1, num2grp)
        sig2 = pattern_signature(D2, num2grp)
        sig3 = pattern_signature(D3, num2grp)

        v1 = one_hot_group_counts(sig1)  # (15,)
        v2 = one_hot_group_counts(sig2)
        v3 = one_hot_group_counts(sig3)

        T12 = np.outer(v1, v2)  # 15x15
        T23 = np.outer(v2, v3)  # 15x15
        # --- (변경) 전이 가중치 분리 ---
        T_recent = 0.65 * T12 + 0.35 * T23

        # 다음 회차 그룹 점수
        next_g = T_recent.T @ v3  # (15,)
        total_group_scores += next_g

    # --- (변경) 그룹 점수의 번호 분배: 균등(1/3) → 과거 빈도 가중 분배 ---
    # 각 그룹 triplet 내에서 역사적 빈도(hist_counts)에 따라 가중 분배
    number_scores = np.zeros(46, dtype=float)
    eps = 1e-3
    for sid, groups in groups_by_set.items():
        for gi, triplet in enumerate(groups):
            g_score = total_group_scores[gi]
            weights = np.array([hist_counts[n-1] for n in triplet], dtype=float) + eps
            weights /= weights.sum()
            for n, w in zip(triplet, weights):
                number_scores[n] += g_score * w

    # 직전 회차(D3) 번호에 패널티
    for n in D3:
        number_scores[n] *= penalty_last_draw

    # 정규화하여 확률 벡터로
    p = number_scores[1:].copy()
    p = np.clip(p, 1e-12, None)
    p /= p.sum()
    probs = np.zeros(46, dtype=float)
    probs[1:] = p  # 1~45

    # ---- 이 확률로 6개 조합 10개 샘플링 (내부 필터 재활용) ----
    def sample_combos_from_probs(probs, want=10, max_attempts=5000):
        selected = []
        seen = set()
        attempts = 0
        base_idx = np.arange(1, 46)
        while len(selected) < want and attempts < max_attempts:
            attempts += 1
            comb = rng.choice(base_idx, size=6, replace=False, p=probs[1:])
            comb = sorted(comb.tolist())

            t = tuple(comb)
            if t in seen:
                continue
            if not check_consecutive_rule(comb):
                continue
            if is_strict_diagonal(comb) and rng.random() >= 0.1:
                continue
            if morphological_pattern_score(comb) == 0:
                continue

            seen.add(t)
            selected.append(comb)
        return selected

    combos = sample_combos_from_probs(probs, want=n_sets)

    # 점수 출력용(선형 효율 등은 내부 확률로 대체)
    effs = [float(np.sum(probs[np.array(c)])) for c in combos]
    v7_vals, circ_vals, morph_vals, diag_vals = evaluate_patterns_batch(combos)
    pat_comb_vals = (v7_vals*0.42 + circ_vals*0.42 + diag_vals*0.11 + (morph_vals/20.0)*0.05)
    total_scores = pat_comb_vals  # 표시용 간단 처리

    results = []
    for i, c in enumerate(combos):
        results.append((
            c,
            float(effs[i]),
            float(v7_vals[i]),
            float(circ_vals[i]),
            float(morph_vals[i]),
            float(pat_comb_vals[i]),
            float(total_scores[i])
        ))
    return results, probs[1:], (D1, D2, D3)

# =========================
# 8️⃣ Streamlit UI
# =========================
if st.button("추천 번호 생성 & 분석 리포트"):
    with st.spinner("계산 중..."):
        t0 = time.time()
        res_mixed, p_next_mixed  = generate_final_combinations_mixed(10, focus_mode=False, free_mode_ratio=FREE_MODE_RATIO)
        res_focus, p_next_focus  = generate_final_combinations_fast(10, focus_mode=True,  ignore_group_balance=False)
        res_ignore, p_next_ign   = generate_final_combinations_fast(10, focus_mode=False, ignore_group_balance=True)
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
        for _, (comb, eff, v7, circ, morph, pat_comb, score) in enumerate(res_ignore, 1):
            st.write(f"{comb} | 효율:{eff:.4f} | V7:{v7:.1f} | 원형:{circ:.1f} | 형태학:{morph:.1f} | 통합:{pat_comb:.1f} | 점수:{score:.4f}")

        st.subheader("🧩 패턴셋 기반: (t-2→t-1, t-1→t 전이) 를 t에 적용해 t+1 예측 10조합")
        pattern_combo_res, pattern_probs, (D1, D2, D3) = recent_pairwise_transition_predict(
            numbers_arr,
            PATTERN_SETS,
            penalty_last_draw=0.55,
            n_sets=10
        )
        for c, eff, v7, circ, morph, patc, sc in pattern_combo_res:
            st.write(f"{c} | 효율:{eff:.4f} | V7:{v7:.1f} | 원형:{circ:.1f} | 형태학:{morph:.1f} | 통합:{patc:.1f}")

        st.write(f"계산 소요 시간: {t1 - t0:.2f}초")

        # ----------------------------
        # 추천 결과 → DataFrame
        # ----------------------------
        df_mixed = combos_to_df(res_mixed, label="혼합형")
        df_focus = combos_to_df(res_focus, label="집중형(균형)")
        df_ignore = combos_to_df(res_ignore, label="번호군 균형 제외(비균형)")
        df_all = pd.concat([df_mixed, df_focus, df_ignore], ignore_index=True)
        st.subheader("📋 추천 결과 테이블")
        st.caption("테이블 헤더를 클릭해 정렬/필터를 시도해보세요. (예: score 내림차순)")
        st.dataframe(df_all)

        # ----------------------------
        # (참고) 다음 회차 단일번호 확률 시각 검토
        # ----------------------------
        st.subheader("📈 다음 회차 번호별 추정 확률 (이중 MC 기반)")
        fig, ax = plt.subplots(figsize=(10, 3.6))
        ax.bar(np.arange(1, 46), p_next_focus, label="focus run probs")
        ax.set_xlabel("번호"); ax.set_ylabel("확률")
        ax.set_title("다음 회차 번호별 등장 확률(추정)")
        ax.legend()
        st.pyplot(fig)

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

        # 번호별 확률 바차트 (패턴 전이 기반)
        st.subheader("📈 패턴 전이 기반 번호 확률")
        fig2, ax2 = plt.subplots(figsize=(10, 3.6))
        ax2.bar(np.arange(1, 46), pattern_probs)
        ax2.set_xlabel("번호"); ax2.set_ylabel("확률")
        ax2.set_title("패턴 전이 기반 다음 회차 번호 확률(정규화)")
        st.pyplot(fig2)
