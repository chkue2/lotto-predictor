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
st.set_page_config(page_title="통합 로또 추천기 V14 (Optimized)", layout="centered")
st.title("🎯 통합 로또 추천기 V14 (Optimized)")

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
def monte_carlo_vectorized(trans_matrix, last_draw, trials=3000, focus_mode=False, random_perturb=0.02):
    probs_base = trans_matrix[[n-1 for n in last_draw]].sum(0)
    probs_base = np.maximum(probs_base, 0.01)
    if focus_mode:
        probs_base = probs_base ** 2
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

# =========================
# 행+열 균형 + 대각선 일부 허용
# =========================
def generate_group_combinations(groups):
    combs = []
    pattern = (2,2,2)
    g0, g1, g2 = groups
    combs0 = list(itertools.combinations(g0, pattern[0]))
    combs1 = list(itertools.combinations(g1, pattern[1]))
    combs2 = list(itertools.combinations(g2, pattern[2]))

    for c1 in combs0:
        for c2 in combs1:
            for c3 in combs2:
                comb = tuple(sorted(set(c1 + c2 + c3)))
                if len(comb) != 6:
                    continue
                if not check_consecutive_rule(comb):
                    continue
                if is_strict_diagonal(comb):
                    # 10% 확률로 대각선 허용
                    if np.random.rand() < 0.1:
                        pass
                    else:
                        continue
                rows = [GRID_POS[n][0] for n in comb]
                cols = [GRID_POS[n][1] for n in comb]
                if max([rows.count(r) for r in range(7)]) > 3:
                    continue
                if max([cols.count(c) for c in range(7)]) > 3:
                    continue
                combs.append(list(comb))
    return combs

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
# 6️⃣ 최종 조합 생성 (개선)
# =========================
def generate_final_combinations_fast(n_sets=10, focus_mode=False):
    trans = build_transition_matrix(numbers_arr)
    last_draw = numbers_arr[-1]
    candidates_raw = monte_carlo_vectorized(trans, last_draw, trials=5000, focus_mode=focus_mode)
    counts = np.bincount(candidates_raw.flatten()-1, minlength=45)
    probs = counts / counts.sum()
    groups = divide_into_groups(probs, focus_mode)
    candidates = generate_group_combinations(groups)
    candidates = [c for c in candidates if morphological_pattern_score(c)!=0]
    unique = {tuple(c):c for c in candidates}
    candidates = list(unique.values())
    cand_arr = np.array(candidates)
    eff_vals = probs[cand_arr-1].sum(axis=1)
    v7_vals, circ_vals, morph_vals, diag_vals = evaluate_patterns_batch(candidates)
    combined_pattern = (v7_vals*0.45 + circ_vals*0.45 + diag_vals*0.1)
    
    # ✅ 상위 후보 풀 확대 + 랜덤 가중치 섞기
    rand_factor = np.random.uniform(0.95,1.05,len(eff_vals))
    total_scores = (0.65*eff_vals + 0.35*(combined_pattern/50)) * rand_factor if not focus_mode else (0.8*eff_vals + 0.2*(combined_pattern/50)) * rand_factor
    
    top_n = int(len(total_scores)*0.8)  # 후보 풀 확대
    rand_n = n_sets - top_n if n_sets > top_n else 0
    top_idx = np.argsort(-total_scores)[:top_n]
    if rand_n>0 and len(total_scores)>top_n:
        remaining_idx = np.argsort(-total_scores)[top_n:]
        rand_idx = np.random.choice(remaining_idx, size=rand_n, replace=False)
        top_idx = np.concatenate([top_idx, rand_idx])
    
    final_results=[]
    for idx in top_idx[:n_sets]:
        final_results.append((
            candidates[idx],
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
if st.button("추천 번호 생성 & 분석 리포트 (Optimized V14)"):
    with st.spinner("계산 중... (V14 최적화)"):
        t0=time.time()
        res_bal,_=generate_final_combinations_fast(10,focus_mode=False)
        res_focus,_=generate_final_combinations_fast(10,focus_mode=True)
        t1=time.time()

        st.subheader("✅ 균형형 추천 10조합")
        for _,(comb,eff,v7,circ,morph,pat_comb,score) in enumerate(res_bal,1):
            st.write(f"{comb} | 효율:{eff:.4f} | V7:{v7:.1f} | 원형:{circ:.1f} | 형태학:{morph:.1f} | 통합:{pat_comb:.1f} | 점수:{score:.4f}")

        st.subheader("🔥 집중형 추천 10조합")
        for _,(comb,eff,v7,circ,morph,pat_comb,score) in enumerate(res_focus,1):
            st.write(f"{comb} | 효율:{eff:.4f} | V7:{v7:.1f} | 원형:{circ:.1f} | 형태학:{morph:.1f} | 통합:{pat_comb:.1f} | 점수:{score:.4f}")

        st.write(f"계산 소요 시간: {t1-t0:.2f}초")

        df_bal=combos_to_df(res_bal,start_index=1,label="균형형")
        df_focus=combos_to_df(res_focus,start_index=1,label="집중형")
        result_df=pd.concat([df_bal,df_focus],ignore_index=True)

        st.markdown("---")
        st.subheader("📊 강화된 분석 리포트")
        hist_counts,hist_probs=compute_historic_freq(numbers_arr)
        hot_idx=np.argsort(-hist_counts)[:10]+1
        cold_idx=np.argsort(hist_counts)[:10]+1
        st.write("**과거 데이터(전체) — 핫 10 / 콜드 10**")
        st.write(f"Hot: {hot_idx.tolist()}, Cold: {cold_idx.tolist()}")

        fig1,ax1=plt.subplots(figsize=(9,3))
        idxs=np.arange(1,46)
        ax1.bar(idxs,hist_counts,label='출현 횟수')
        ax2=ax1.twinx()
        ax2.plot(idxs,np.cumsum(hist_probs),marker='o',label='누적확률')
        ax1.set_xlabel("번호"); ax1.set_ylabel("등장 횟수"); ax2.set_ylabel("누적확률")
        ax1.set_title("과거 데이터 번호 등장 횟수 및 누적확률")
        ax1.legend(loc='upper left'); ax2.legend(loc='upper right')
        st.pyplot(fig1)

        mat=cooccurrence_matrix(numbers_arr)
        fig2,ax2=plt.subplots(figsize=(7,6))
        im=ax2.imshow(mat,interpolation='nearest',cmap='YlOrRd')
        ax2.set_title("과거 데이터 공출현 행렬")
        ax2.set_xlabel("번호"); ax2.set_ylabel("번호")
        ax2.set_xticks(range(45)); ax2.set_xticklabels(range(1,46))
        ax2.set_yticks(range(45)); ax2.set_yticklabels(range(1,46))
        fig2.colorbar(im,ax=ax2,fraction=0.046,pad=0.04)
        st.pyplot(fig2)

        co_pairs=[((i+1,j+1),mat[i,j]) for i in range(45) for j in range(i+1,45)]
        co_pairs.sort(key=lambda x:-x[1])
        st.write("**상위 10 공출현 번호 쌍:**",[p[0] for p in co_pairs[:10]])

        all_generated=[c for c in result_df['combo']]
        flat_generated=np.array(all_generated).flatten()
        gen_counts=np.bincount(flat_generated-1,minlength=45)
        gen_order=np.argsort(-gen_counts)+1
        st.write("Generated 번호 빈도 상위 10:",gen_order[:10].tolist())

        fig3,ax3=plt.subplots(figsize=(9,3))
        ax3.bar(result_df['rank'],result_df['v7'],alpha=0.7,label='V7 패턴')
        ax3.bar(result_df['rank'],result_df['circ'],alpha=0.5,label='원형 패턴')
        ax3.set_xlabel("조합 순위"); ax3.set_ylabel("패턴 점수"); ax3.set_title("20조합 패턴 점수 비교")
        ax3.legend(); st.pyplot(fig3)

        group1=flat_generated[(flat_generated>=1)&(flat_generated<=15)]
        group2=flat_generated[(flat_generated>=16)&(flat_generated<=30)]
        group3=flat_generated[(flat_generated>=31)&(flat_generated<=45)]
        st.write("**번호 그룹별 등장 횟수** 1~15:{}, 16~30:{}, 31~45:{}".format(len(group1),len(group2),len(group3)))

        overlaps=[len(set(a)&set(b)) for a,b in itertools.combinations(all_generated,2)]
        overlaps=np.array(overlaps) if overlaps else np.array([0])
        st.write(f"조합 간 평균 중복: {overlaps.mean():.3f}, 최대: {overlaps.max()}, 최소: {overlaps.min()}")

        bal_combos=[c for c in df_bal['combo']]
        foc_combos=[c for c in df_focus['combo']]
        inter_counts=[len(set(a)&set(b)) for a in bal_combos for b in foc_combos] if bal_combos and foc_combos else [0]
        inter_counts=np.array(inter_counts)
        st.write(f"균형형 vs 집중형 평균 교집합: {inter_counts.mean():.3f}, 최대: {inter_counts.max()}")
