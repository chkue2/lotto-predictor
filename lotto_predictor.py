import streamlit as st
import numpy as np, random, itertools, time, re, requests, pandas as pd
import networkx as nx

st.set_page_config(page_title="통합 로또 추천기 V4 Optimized", layout="centered")

# =========================
# 1️⃣ 데이터 가져오기
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
def fetch_recent(draw_count=200):
    latest = get_latest_draw_no()
    if not latest: return pd.DataFrame()
    start = max(1, latest - draw_count + 1)
    records = []
    for drw in range(start, latest + 1):
        rec = fetch_lotto(drw)
        if rec: records.append(rec)
        time.sleep(0.01)
    return pd.DataFrame(records)

# =========================
# 2️⃣ 마르코프 전이 확률
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
# 3️⃣ Monte Carlo 시뮬레이션 (벡터화)
# =========================
def monte_carlo_vectorized(trans_matrix, last_draw, trials=3000):
    probs_base = trans_matrix[[n-1 for n in last_draw]].sum(0)
    probs_base = np.maximum(probs_base,0.01)
    probs_base /= probs_base.sum()
    draws = np.random.choice(np.arange(1,46), size=(trials,6), p=probs_base)
    counts = np.bincount(draws.flatten()-1, minlength=45)
    return counts / counts.sum()

# =========================
# 4️⃣ 그룹 기반 조합
# =========================
def divide_into_groups(probabilities):
    sorted_idx = np.argsort(-probabilities)
    g1 = sorted_idx[:15]+1
    g2 = sorted_idx[15:30]+1
    g3 = sorted_idx[30:]+1
    return g1.tolist(), g2.tolist(), g3.tolist()

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
# 5️⃣ 연번/패턴 체크
# =========================
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

# =========================
# 6️⃣ Gianella 패턴
# =========================
lotto_grid=[
 [1,2,3,4,5,6,7],
 [8,9,10,11,12,13,14],
 [15,16,17,18,19,20,21],
 [22,23,24,25,26,27,28],
 [29,30,31,32,33,34,35],
 [36,37,38,39,40,41,42],
 [43,44,45]
]

def gianella_pattern(numbers):
    coords = [(r,c) for r,row in enumerate(lotto_grid) for c,v in enumerate(row) if v in numbers]
    rows = [0]*7; cols=[0]*7
    for r,c in coords: rows[r]+=1; cols[c]+=1
    diag1 = sum(r==c for r,c in coords)
    diag2 = sum(c==6-r for r,c in coords)
    return sum(x*x for x in rows) + sum(x*x for x in cols) + diag1 + diag2

# =========================
# 7️⃣ Fitness
# =========================
def fitness_func(comb, probabilities):
    eff = sum(probabilities[i-1] for i in comb)
    pat = gianella_pattern(comb)
    return 0.7*eff + 0.3*(pat/50)

# =========================
# 8️⃣ Mutation
# =========================
def mutate(child, mutation_rate=0.3):
    if random.random() < mutation_rate:
        idx = random.randint(0,5)
        r = random.randint(1,45)
        while r in child:
            r = random.randint(1,45)
        child[idx] = r
    return sorted(child)

# =========================
# 9️⃣ 유전 알고리즘 최적화
# =========================
def evolve_combinations(candidates, probabilities, total_combs=5000, generations=12):
    # 초기 후보 풀 제한
    if len(candidates) > total_combs:
        candidates = random.sample(candidates, total_combs)
    pop = candidates.copy()
    
    for _ in range(generations):
        scored = [(c, fitness_func(c, probabilities)) for c in pop]
        scored.sort(key=lambda x:x[1], reverse=True)
        parents = [c for c,_ in scored[:total_combs//2]]
        children = []
        while len(children) < total_combs//2:
            p1, p2 = random.sample(parents, 2)
            child = sorted(list(set(random.sample(p1,3) + random.sample(p2,3))))
            while len(child) < 6:
                r = random.randint(1,45)
                if r not in child: child.append(r)
            child = mutate(child)
            if check_consecutive_rule(child):
                children.append(child)
        pop = parents + children
    
    scored = [(c, fitness_func(c, probabilities)) for c in pop]
    scored.sort(key=lambda x:x[1], reverse=True)
    return scored[:10]

# =========================
# Streamlit UI
# =========================
st.title("🎯 통합 로또 추천기 V4")
st.write("최적화된 Monte Carlo + 유전 알고리즘 기반 10세트 추천 번호 생성")
st.write("이전 100회차를 불러오기 때문에 실행시 시간이 오래걸려요! 기다려주세요!")

if st.button("추천 번호 생성"):
    with st.spinner("계산 중... 잠시만 기다려주세요."):
        df = fetch_recent(200)
        if df.empty:
            st.warning("데이터를 가져오지 못했습니다.")
        else:
            numbers = np.array(df['numbers'].tolist())
            trans = build_transition_matrix(numbers)
            last_draw = numbers[-1]
            
            # 벡터화 Monte Carlo 2종 평균
            mc1 = monte_carlo_vectorized(trans, last_draw)
            mc2 = monte_carlo_vectorized(trans, last_draw)
            probs = (mc1 + mc2)/2
            
            # 그룹 나누기 & 후보 조합 생성
            groups = divide_into_groups(probs)
            candidates = generate_group_combinations(groups)
            
            # 유전 알고리즘으로 최종 10세트 선택
            final = evolve_combinations(candidates, probs)
            
            st.success("✅ 추천 번호 생성 완료!")
            for i,(comb,score) in enumerate(final,1):
                st.write(f"{i:02d}. {comb} | 점수: {score:.4f}")
