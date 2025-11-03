"""
통합 로또 추천기 (Streamlit UI)
- 버튼 클릭으로 최신 데이터 기반 10세트 추천
- 유전 알고리즘 개선: 다양성 확보, 돌연변이 적용
"""
import streamlit as st
import numpy as np, random, itertools, time, re, requests, pandas as pd
from sklearn.cluster import KMeans
import networkx as nx
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

st.set_page_config(page_title="통합 로또 추천기", layout="centered")

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

# === 마르코프 전이 확률 ===
def build_transition_matrix(numbers):
    n = 45
    m = np.zeros((n,n))
    for i in range(len(numbers)-1):
        for a in numbers[i]:
            for b in numbers[i+1]:
                m[a-1, b-1] += 1
    p = m / m.sum(1, keepdims=True)
    return np.nan_to_num(p)

# === LSTM 시계열 보정 ===
def lstm_forecast(history, lookback=30):
    X, y = [], []
    for i in range(len(history)-lookback):
        X.append(history[i:i+lookback])
        y.append(history[i+lookback])
    X, y = np.array(X), np.array(y)
    model = Sequential([LSTM(16,input_shape=(lookback,45)),Dense(45,activation='sigmoid')])
    model.compile(loss='binary_crossentropy',optimizer='adam')
    model.fit(X,y,epochs=5,batch_size=8,verbose=0)
    pred = model.predict(X[-1].reshape(1,lookback,45), verbose=0)[0]
    return pred

# === 그래프 중심도 기반 가중치 ===
def graph_centrality(numbers):
    G = nx.Graph()
    for draw in numbers:
        for a, b in itertools.combinations(draw,2):
            G.add_edge(a,b,weight=G[a][b]['weight']+1 if G.has_edge(a,b) else 1)
    cent = nx.eigenvector_centrality_numpy(G)
    arr = np.array([cent.get(i,0) for i in range(1,46)])
    return arr/arr.sum()

# === 군집화 기반 가중치 ===
def cluster_weights(numbers):
    arr = np.zeros((len(numbers),45))
    for i, nums in enumerate(numbers):
        arr[i, [n-1 for n in nums]] = 1
    km = KMeans(n_clusters=5,n_init='auto').fit(arr)
    labels = km.labels_
    last_label = labels[-1]
    cluster_mean = arr[labels==last_label].mean(0)
    return cluster_mean/cluster_mean.sum()

# === 연번 조건 ===
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

# === Gianella 패턴 ===
def gianella_pattern(numbers, grid):
    coords = [(r,c) for r,row in enumerate(grid) for c,v in enumerate(row) if v in numbers]
    rows = [0]*7; cols=[0]*7
    for r,c in coords: rows[r]+=1; cols[c]+=1
    diag1 = sum(r==c for r,c in coords)
    diag2 = sum(c==6-r for r,c in coords)
    return sum(x*x for x in rows) + sum(x*x for x in cols) + diag1 + diag2

# === 통합 확률 계산 ===
def compute_combined_probabilities(df,grid):
    numbers = np.array(df['numbers'].tolist())
    trans = build_transition_matrix(numbers)
    last = numbers[-1]
    markov_p = trans[[n-1 for n in last]].sum(0)
    history = np.zeros((len(numbers),45))
    for i, nums in enumerate(numbers): history[i, [n-1 for n in nums]] = 1
    lstm_p = lstm_forecast(history)
    graph_p = graph_centrality(numbers)
    cluster_p = cluster_weights(numbers)
    probs = 0.4*markov_p + 0.25*lstm_p + 0.2*graph_p + 0.15*cluster_p
    probs = np.maximum(probs,0.02); probs /= probs.sum()
    return probs

# === Fitness 함수 ===
def fitness_func(comb, probs):
    eff = sum(probs[i-1] for i in comb)
    pat = gianella_pattern(comb, lotto_grid)
    return 0.7*eff + 0.3*(pat/50)

# === 부모 선택 (점수 비례) ===
def select_parents(scored, num_parents):
    scores = np.array([s for _, s in scored])
    candidates = [c for c,_ in scored]
    
    # 점수가 모두 0이면 균등 선택
    if scores.sum() == 0:
        probs = None
    else:
        probs = scores / scores.sum()
    
    # random.choices는 replace=True를 기본으로 하므로 충분히 선택 가능
    parents = random.choices(candidates, weights=probs, k=num_parents)
    return parents

# === 돌연변이 적용 ===
def mutate(child, mutation_rate=0.3):
    if random.random() < mutation_rate:
        idx = random.randint(0,5)
        r = random.randint(1,45)
        while r in child:
            r = random.randint(1,45)
        child[idx] = r
    return sorted(child)

# === 유전 알고리즘식 조합 최적화 (개선) ===
def evolve_combinations(base_probs, fitness_func, pop_size=100, generations=30):
    pop = [sorted(random.sample(range(1,46),6)) for _ in range(pop_size)]
    for _ in range(generations):
        scored = [(c, fitness_func(c, base_probs)) for c in pop]
        scored.sort(key=lambda x:x[1], reverse=True)
        parents = select_parents(scored, pop_size//2)
        children = []
        while len(children) < pop_size//2:
            p1, p2 = random.sample(parents, 2)
            child = sorted(list(set(random.sample(p1,3)+random.sample(p2,3))))
            while len(child) < 6:
                r = random.randint(1,45)
                if r not in child: child.append(r)
            child = mutate(child)
            if check_consecutive_rule(child):
                children.append(child)
        pop = parents + children
    scored = [(c, fitness_func(c, base_probs)) for c in pop]
    scored.sort(key=lambda x:x[1], reverse=True)
    return scored[:10]

# === 대한민국 로또 grid ===
lotto_grid=[
 [1,2,3,4,5,6,7],
 [8,9,10,11,12,13,14],
 [15,16,17,18,19,20,21],
 [22,23,24,25,26,27,28],
 [29,30,31,32,33,34,35],
 [36,37,38,39,40,41,42],
 [43,44,45]
]

# =========================
# Streamlit UI
# =========================
st.title("🎯 통합 로또 추천기 V2")
st.write("최신 데이터를 기반으로 10세트 추천 번호를 생성합니다.")

if st.button("추천 번호 생성"):
    with st.spinner("로또 번호 계산 중... 잠시만 기다려주세요."):
        df = fetch_recent(200)
        if df.empty:
            st.warning("데이터를 가져오지 못했습니다.")
        else:
            probs = compute_combined_probabilities(df, lotto_grid)
            final = evolve_combinations(probs, fitness_func)
            
            st.success("✅ 추천 번호 생성 완료!")
            for i,(comb,score) in enumerate(final,1):
                st.write(f"{i:02d}. {comb} | 점수: {score:.4f}")
