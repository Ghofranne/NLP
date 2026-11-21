# -*- coding: utf-8 -*-
"""FoodReviews AI : interface de démonstration du projet NLP + graphes (Amazon Fine Food Reviews)."""
import os
import re
import json
import html
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

ART = Path(os.environ.get("ART_DIR", Path(__file__).resolve().parent / "artifacts"))
LABELS_DEFAULT = ["négatif", "neutre", "positif"]
LABEL_STYLE = {"négatif": ("😠", "neg", "#ef4444"), "neutre": ("😐", "neu", "#9ca3af"), "positif": ("😍", "pos", "#10b981")}

st.set_page_config(page_title="FoodReviews AI", page_icon="🍽️", layout="wide", initial_sidebar_state="expanded")

CSS = """
<style>
:root {
    --ink: #292b30;
    --muted: #70747b;
    --accent: #795b7e;
    --accent-dark: #614766;
    --accent-soft: #eee7ef;
    --line: #dce1e5;
}
html {scroll-behavior: smooth;}
#MainMenu, footer {visibility: hidden;}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 92% 3%, rgba(128, 94, 132, .09), transparent 29rem),
        radial-gradient(ellipse at 8% 40%, rgba(119, 133, 145, .055), transparent 25rem),
        #e9edef;
}
[data-testid="stHeader"] {background: rgba(233,237,239,.82); backdrop-filter: blur(14px);}
.block-container {padding: 1.35rem 2.2rem 3.5rem; max-width: 1510px;}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f9fa 0%, #eef1f3 100%);
    border-right: 1px solid rgba(207,213,218,.95);
    box-shadow: 8px 0 30px rgba(27,35,68,.025);
}
[data-testid="stSidebar"] > div:first-child {padding: 1.35rem 1rem 1rem;}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {line-height: 1.55;}
[data-testid="stSidebar"] [role="radiogroup"] {gap: .28rem;}
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding: .67rem .72rem; border: 1px solid transparent; border-radius: 12px;
    transition: background .2s ease, border-color .2s ease, transform .2s ease, box-shadow .2s ease; cursor: pointer;
}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #e9edef; transform: translateX(2px);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background: linear-gradient(105deg, #eee6ef, #f7f3f7);
    border-color: #dfd1e1; color: #614766; font-weight: 700;
    box-shadow: inset 3px 0 0 #795b7e, 0 4px 12px rgba(82,58,87,.07);
}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {color: #614766;}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {display: none;}
[data-testid="stSidebar"] [data-testid="stRadio"] > label {display: none;}

.hero {
    background: linear-gradient(116deg, #eef0f1 0%, #e2e5e7 56%, #d7dce0 100%);
    padding: clamp(1.75rem, 4vw, 3.1rem); border-radius: 26px; color: var(--ink);
    margin-bottom: 1.7rem; box-shadow: 0 22px 48px rgba(42,48,54,.11);
    border: 1px solid rgba(255,255,255,.72);
    position: relative; overflow: hidden;
    isolation: isolate;
    animation: hero-arrive .65s cubic-bezier(.2,.7,.25,1) both;
}
.hero:after {
    content: ""; position: absolute; z-index: -1; width: 340px; height: 340px; border-radius: 50%;
    right: -85px; top: -190px;
    border: 1px solid rgba(121,91,126,.23);
    background: radial-gradient(circle at 35% 65%, rgba(121,91,126,.17), rgba(121,91,126,.015) 68%);
    box-shadow: 0 0 0 28px rgba(121,91,126,.035), 0 0 0 58px rgba(121,91,126,.025);
}
.hero-kicker {font-size: .7rem; font-weight: 800; letter-spacing: .15em; color: #4b4d51; margin-bottom: .75rem;}
.hero h1 {margin: 0; font-size: clamp(2.15rem, 4vw, 3.1rem); line-height: 1.08; color: #292b30 !important; letter-spacing: -.05em;}
.hero h1 span {color: #795b7e !important;}
.hero p {margin: .8rem 0 0; max-width: 700px; color: #62676d; font-size: 1rem; line-height: 1.7;}
.hero-meta {display: flex; flex-wrap: wrap; gap: .55rem; margin-top: 1.35rem;}
.hero-meta span {padding: .42rem .76rem; border: 1px solid rgba(121,91,126,.18); border-radius: 999px;
    background: #795b7e; font-size: .76rem; color: #fff; box-shadow: 0 4px 10px rgba(88,62,93,.13);}
.section-heading {
    display: flex; align-items: end; justify-content: space-between; gap: 1rem;
    margin: .2rem 0 1.1rem; animation: content-arrive .4s ease both;
}
.section-heading h2 {margin: 0; color: var(--ink); font-size: 1.55rem; letter-spacing: -.035em;}
.section-heading p {margin: .28rem 0 0; color: var(--muted); font-size: .9rem;}
.section-eyebrow {color: var(--accent-dark); font-size: .68rem; font-weight: 800; letter-spacing: .13em; margin-bottom: .32rem;}
.dashboard-intro {padding: .1rem .1rem .4rem;}

.stTabs [data-baseweb="tab-list"] {
    gap: .35rem; padding: .4rem; background: #fff; border: 1px solid var(--line);
    border-radius: 15px; box-shadow: 0 4px 14px rgba(16,24,40,.035);
}
.stTabs [data-baseweb="tab"] {height: 2.8rem; border-radius: 11px; padding: 0 1rem; color: #596174;}
.stTabs [data-baseweb="tab"]:hover {background: #f2ebf3; color: var(--accent-dark);}
.stTabs [aria-selected="true"] {background: var(--accent-soft); color: var(--accent-dark); font-weight: 700;}
.stTabs [data-baseweb="tab-highlight"] {background: transparent;}

div[data-testid="stMetric"] {
    background: linear-gradient(150deg, #fff, #fdfdff);
    border: 1px solid rgba(229,232,241,.95); padding: 1.02rem 1.1rem;
    border-radius: 17px; box-shadow: 0 7px 20px rgba(24,33,66,.045);
    transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
}
[data-testid="stHorizontalBlock"] [data-testid="stMetric"]:hover {
    transform: translateY(-3px); border-color: #d8c8da; box-shadow: 0 13px 26px rgba(62,48,65,.09);
}
[data-testid="stMetricLabel"] {color: var(--muted); font-weight: 600;}
[data-testid="stMetricValue"] {color: var(--ink); letter-spacing: -.03em;}
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"] {
    border-color: rgba(229,232,241,.95); border-radius: 17px; background: #fff;
    box-shadow: 0 6px 20px rgba(24,33,66,.035);
}
[data-testid="stExpander"] {border: 1px solid var(--line); border-radius: 14px; background: #fff; transition: box-shadow .2s ease;}
[data-testid="stExpander"]:hover {box-shadow: 0 6px 18px rgba(24,33,66,.045);}
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea,
[data-testid="stSelectbox"] [data-baseweb="select"] > div {border-radius: 11px;}
[data-testid="stButton"] button {
    border-radius: 10px; font-weight: 600;
    transition: color .18s ease, background .18s ease, border-color .18s ease, box-shadow .18s ease, transform .18s ease;
}
[data-testid="stButton"] button[kind="primary"] {
    border: 0; background: linear-gradient(110deg, #805f84, #624768);
    box-shadow: 0 6px 16px rgba(101,72,105,.2);
}
[data-testid="stButton"] button[kind="primary"]:hover {
    filter: brightness(1.06); transform: translateY(-1px); box-shadow: 0 9px 22px rgba(101,72,105,.27);
}
[data-testid="stButton"] button:active {transform: translateY(0) scale(.985);}
[data-testid="stDataFrame"], [data-testid="stTable"] {border: 1px solid var(--line); border-radius: 12px; overflow: hidden;}
[data-testid="stAlert"] {border-radius: 12px;}
[data-testid="stPlotlyChart"] {background: #fff; border: 1px solid var(--line); border-radius: 17px; padding: .35rem;}
[data-testid="stChatMessage"] {border: 1px solid var(--line); border-radius: 15px; background: #fff; margin-bottom: .65rem;}

.sidebar-brand {display: flex; align-items: center; gap: .7rem; padding: .1rem .25rem 1.15rem;
    border-bottom: 1px solid var(--line); margin-bottom: 1.2rem;}
.sidebar-logo {display: grid; place-items: center; width: 2.45rem; height: 2.45rem;
    border-radius: 12px; color: #fff; font-size: 1.25rem;
    background: linear-gradient(140deg, #8b6a90, #624768); box-shadow: 0 5px 12px rgba(99,73,103,.2);
    animation: logo-breathe 4s ease-in-out infinite;}
.sidebar-brand-name {font-size: .98rem; font-weight: 800; color: var(--ink); letter-spacing: -.025em;}
.sidebar-brand-sub {font-size: .72rem; color: var(--muted); margin-top: .08rem;}
.sidebar-label {font-size: .68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .12em; color: #8790a2; margin: .35rem 0 .45rem;}
.sidebar-status {display: flex; align-items: center; gap: .5rem; padding: .62rem .7rem;
    border: 1px solid var(--line); border-radius: 11px; background: #fafbfe; color: #344054;
    font-size: .79rem; font-weight: 600;}
.status-dot {width: .48rem; height: .48rem; border-radius: 50%; background: #20a779;
    box-shadow: 0 0 0 3px #e4f6ef; flex: 0 0 auto;}
.status-dot.fallback {background: #d99121; box-shadow: 0 0 0 3px #fff3dc;}
.sidebar-stats {display: grid; grid-template-columns: 1fr 1fr; gap: .55rem; margin: .45rem 0 .85rem;}
.sidebar-stat {padding: .62rem .68rem; border: 1px solid var(--line); border-radius: 11px; background: #fff;}
.sidebar-stat strong {display: block; color: var(--ink); font-size: 1rem; letter-spacing: -.02em;}
.sidebar-stat span {display: block; margin-top: .12rem; color: var(--muted); font-size: .7rem;}
.sidebar-foot {padding-top: .8rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .74rem;}
.pill {display: inline-block; padding: .3rem .9rem; border-radius: 999px; font-weight: 700; font-size: .95rem;}
.pill.pos {background: #d1fae5; color: #065f46;} .pill.neg {background: #fee2e2; color: #991b1b;}
.pill.neu {background: #e5e7eb; color: #374151;}
.badge {display: inline-block; padding: .12rem .6rem; border-radius: 999px; font-size: .78rem; font-weight: 600;
        background: #eee7ef; color: #614766; margin-right: .3rem;}
.badge.ok {background: #d1fae5; color: #065f46;}
.rank {font-size: 1.6rem; font-weight: 800; color: var(--accent);}
.muted {color: var(--muted); font-size: .85rem;}

@keyframes hero-arrive {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}
@keyframes content-arrive {
    from {opacity: 0; transform: translateY(5px);}
    to {opacity: 1; transform: translateY(0);}
}
@keyframes logo-breathe {
    0%, 100% {box-shadow: 0 5px 12px rgba(99,73,103,.2);}
    50% {box-shadow: 0 6px 17px rgba(121,91,126,.3);}
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {scroll-behavior: auto !important; animation-duration: .01ms !important;
        animation-iteration-count: 1 !important; transition-duration: .01ms !important;}
}
@media (max-width: 700px) {
    .block-container {padding: 1rem 1rem 2rem;}
    .hero {border-radius: 18px;}
    .stTabs [data-baseweb="tab"] {padding: 0 .65rem; font-size: .82rem;}
    [data-testid="stSidebar"] > div:first-child {padding: 1rem .8rem;}
    .section-heading {align-items: start; flex-direction: column;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

FR_EN = {"chien": "dog", "chiens": "dog", "chat": "cat", "chats": "cat", "cafe": "coffee", "café": "coffee",
         "thé": "tea", "the vert": "green tea", "chocolat": "chocolate", "biscuit": "cookie", "biscuits": "cookie",
         "bonbon": "candy", "bonbons": "candy", "pâtes": "pasta", "pates": "pasta", "riz": "rice", "jus": "juice",
         "eau": "water", "céréales": "cereal", "cereales": "cereal", "gâteau": "cake", "miel": "honey",
         "gingembre": "ginger", "fromage": "cheese", "boisson": "drink", "boissons": "drink", "bébé": "baby",
         "sucre": "sugar", "épices": "spices", "epices": "spices", "huile": "oil", "noix": "nuts",
         "avoine": "oats", "shampoing": "shampoo", "friandises": "treats", "pain": "bread", "soupe": "soup",
         "nouilles": "noodles", "sans gluten": "gluten free", "sauce": "sauce", "chips": "chips"}

EXAMPLES = {
    "😍 Exemple positif": "Absolutely delicious! Fresh, great flavor and arrived quickly. I will definitely buy this again.",
    "😠 Exemple négatif": "Stale and tasteless. The package was damaged and the taste was awful. Total waste of money.",
    "😐 Exemple neutre": "It is okay, nothing special. Not bad but I probably would not order it again.",
}


def zs(v):
    return (v - v.mean()) / (v.std() + 1e-9)


# --------------------------------------------------------------------------------------
# Chargement des données et des modèles (mis en cache)
# --------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Chargement des données…")
def load_store():
    S = SimpleNamespace()
    df = pd.read_csv(ART / "reviews.csv.gz")
    for c in ("summary", "text"):
        df[c] = df[c].fillna("").astype(str)
    topics = {int(k): v for k, v in json.load(open(ART / "topics.json", encoding="utf-8")).items()}
    df["theme"] = df["topic"].map(topics).fillna("autres")
    S.df = df
    S.meta = json.load(open(ART / "meta.json", encoding="utf-8"))
    S.labels = S.meta.get("labels", LABELS_DEFAULT)
    S.users = json.load(open(ART / "users.json", encoding="utf-8"))
    S.prods = json.load(open(ART / "prods.json", encoding="utf-8"))
    S.u_idx = {u: i for i, u in enumerate(S.users)}
    S.p_idx = {p: i for i, p in enumerate(S.prods)}
    S.Zu = np.load(ART / "bpr_user.npy"); S.Zp = np.load(ART / "bpr_prod.npy")
    S.Zu_n = S.Zu / (np.linalg.norm(S.Zu, axis=1, keepdims=True) + 1e-9)
    emb = ART / "review_emb.npy"
    S.E = np.load(emb).astype(np.float32) if emb.exists() else None
    S.beta = float(S.meta.get("beta", 0.0))
    S.mu = float(df.Score.mean())

    train = df[df.split == "train"]
    S.train_seen = {u: np.array([S.p_idx[p] for p in s if p in S.p_idx], dtype=int)
                    for u, s in train.groupby("UserId").ProductId}
    pos = train[train.Score >= 4]
    S.liked = [set() for _ in S.users]
    for u, p in zip(pos.UserId, pos.ProductId):
        if u in S.u_idx and p in S.p_idx:
            S.liked[S.u_idx[u]].add(S.p_idx[p])
    held = df[(df.split != "train") & (df.Score >= 4)]
    S.heldout = {u: {S.p_idx[p] for p in s if p in S.p_idx} for u, s in held.groupby("UserId").ProductId}
    popc = np.bincount([S.p_idx[p] for p in pos.ProductId if p in S.p_idx], minlength=len(S.prods)).astype(float)
    S.logpop = zs(np.log1p(popc))

    g = df.groupby("ProductId")
    S.pstats = pd.DataFrame({"n": g.size(), "mean": g.Score.mean(),
                             "theme": g.theme.agg(lambda s: s.mode().iat[0]), "comm": g.prod_comm.first()})
    S.pstats["bayes"] = (S.pstats.n * S.pstats["mean"] + 5 * S.mu) / (S.pstats.n + 5)
    S.pstats["example"] = df.sort_values("Score", ascending=False).drop_duplicates("ProductId").set_index("ProductId")["summary"]
    liked_df = df[df.Score >= 4]
    fav = liked_df.groupby("UserId").theme.agg(lambda s: s.mode().iat[0])
    ug = df.groupby("UserId")
    S.uprof = pd.DataFrame({"n": ug.size(), "mean": ug.Score.mean(), "comm": ug.user_comm.first()})
    S.uprof["fav"] = fav
    S.uprof["fav"] = S.uprof["fav"].fillna("-")
    S.comm_table = pd.read_csv(ART / "comm_table.csv") if (ART / "comm_table.csv").exists() else None
    return S


@st.cache_resource(show_spinner="Chargement du modèle de sentiment…")
def load_sentiment():
    """DistilBERT fine-tuné si disponible (dossier local ou Hub), sinon TF-IDF + régression logistique."""
    src = os.environ.get("SENTIMENT_MODEL_ID") or (str(ART / "sentiment_model") if (ART / "sentiment_model").exists() else None)
    if src:
        try:
            from transformers import pipeline
            return "DistilBERT fine-tuné", pipeline("text-classification", model=src, tokenizer=src, top_k=None)
        except Exception as e:  # noqa
            print("DistilBERT indisponible :", e)
    import joblib
    obj = joblib.load(ART / "tfidf_lr.joblib")
    return "TF-IDF + régression logistique", (obj["tfidf"], obj["lr"])


@st.cache_resource(show_spinner="Chargement de l'encodeur sémantique…")
def load_encoder():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:  # noqa
        print("Sentence-BERT indisponible :", e)
        return None


@st.cache_resource(show_spinner="Construction de l'index de recherche…")
def load_tfidf_index(_S):
    from sklearn.feature_extraction.text import TfidfVectorizer
    tf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
    X = tf.fit_transform((_S.df.summary + ". " + _S.df.text).tolist())
    return tf, X


# --------------------------------------------------------------------------------------
# Fonctions métier
# --------------------------------------------------------------------------------------
def predict_sentiment(model, texts, labels):
    kind, obj = model
    n = len(labels)
    if kind.startswith("DistilBERT"):
        res = obj(list(texts), truncation=True, max_length=128, top_k=None)
        out = np.zeros((len(texts), n))
        for i, r in enumerate(res):
            for d in r:
                lab = d["label"]
                j = int(lab.split("_")[-1]) if lab.upper().startswith("LABEL") else labels.index(lab)
                out[i, j] = d["score"]
        return out
    tfidf, lr = obj
    proba = lr.predict_proba(tfidf.transform(list(texts)))
    out = np.zeros((len(texts), n))
    for k, c in enumerate(lr.classes_):
        out[:, int(c)] = proba[:, k]
    return out


def fr2en(q):
    low = q.lower()
    for fr in sorted(FR_EN, key=len, reverse=True):
        low = re.sub(rf"\b{re.escape(fr)}\b", FR_EN[fr], low)
    return low


def semantic_scores(S, query):
    q = fr2en(query)
    enc = load_encoder()
    if enc is not None and S.E is not None:
        return S.E @ enc.encode([q], normalize_embeddings=True)[0].astype(np.float32)
    tf, X = load_tfidf_index(S)
    return (X @ tf.transform([q]).T).toarray().ravel()


def search_products(S, query, k=8, n_cand=600, min_hits=2):
    sims = semantic_scores(S, query)
    n_cand = min(n_cand, len(sims) - 1)
    cand = np.argpartition(-sims, n_cand)[:n_cand]
    hits = S.df.iloc[cand].assign(sim=sims[cand])
    hits = hits[hits.sim > 0]
    if hits.empty:
        return pd.DataFrame()
    g = hits.groupby("ProductId").agg(avis_pertinents=("Score", "size"), note_pertinente=("Score", "mean"), similarité=("sim", "mean"))
    if (g.avis_pertinents >= min_hits).any():
        g = g[g.avis_pertinents >= min_hits]
    g["score"] = (g.avis_pertinents * g.note_pertinente + 3 * S.mu) / (g.avis_pertinents + 3)
    g = g.join(S.pstats[["n", "mean", "theme"]].rename(columns={"n": "avis (total)", "mean": "note globale", "theme": "thème"}))
    g = g.sort_values(["score", "avis_pertinents"], ascending=False).head(k)
    return g.round(2).reset_index()


def recommend(S, uid, k=10, use_pop=True):
    ui = S.u_idx[uid]
    sc = zs(S.Zp @ S.Zu[ui])
    if use_pop:
        sc = sc + S.beta * S.logpop
    sc = sc.astype(float)
    sc[S.train_seen.get(uid, np.array([], dtype=int))] = -np.inf
    top = [i for i in np.argsort(-sc)[:k] if np.isfinite(sc[i])]
    sims = S.Zu_n @ S.Zu_n[ui]; sims[ui] = -1
    neigh = np.argsort(-sims)[:20]
    held = S.heldout.get(uid, set())
    rows = []
    for t in top:
        pid = S.prods[t]; ps = S.pstats.loc[pid]
        rows.append({"rang": len(rows) + 1, "ProductId": pid, "thème": ps.theme, "note": round(float(ps["mean"]), 2),
                     "avis": int(ps.n), "voisins": int(sum(t in S.liked[j] for j in neigh)),
                     "confirmé": t in held, "exemple": ps.example})
    return pd.DataFrame(rows)


def similar_users(S, uid, k=5):
    ui = S.u_idx[uid]
    sims = S.Zu_n @ S.Zu_n[ui]; sims[ui] = -np.inf
    mine, rows = S.liked[ui], []
    for j in np.argsort(-sims)[:k]:
        v = S.users[j]
        rows.append({"utilisateur": v, "similarité": round(float(sims[j]), 3),
                     "produits aimés en commun": len(mine & S.liked[j]),
                     "même communauté": bool(S.uprof.loc[v, "comm"] == S.uprof.loc[uid, "comm"]),
                     "thème préféré": S.uprof.loc[v, "fav"], "nb avis": int(S.uprof.loc[v, "n"])})
    return pd.DataFrame(rows)


def also_liked(S, pid, k=8):
    users_who = S.df[(S.df.ProductId == pid) & (S.df.Score >= 4) & (S.df.split == "train")].UserId.unique()
    sub = S.df[S.df.UserId.isin(users_who) & (S.df.Score >= 4) & (S.df.split == "train") & (S.df.ProductId != pid)]
    c = sub.ProductId.value_counts().head(k)
    t = pd.DataFrame({"ProductId": c.index, "utilisateurs en commun": c.values})
    t["thème"] = t.ProductId.map(S.pstats.theme); t["note"] = t.ProductId.map(S.pstats["mean"]).round(2)
    return t, len(users_who)


def worst_products(S, k=8, min_n=10):
    p = S.pstats[S.pstats.n >= min_n].sort_values("bayes").head(k)
    return pd.DataFrame({"ProductId": p.index, "thème": p.theme.values, "note": p["mean"].round(2).values, "avis": p.n.values})


def best_products(S, k=8):
    p = S.pstats.sort_values("bayes", ascending=False).head(k)
    return pd.DataFrame({"ProductId": p.index, "thème": p.theme.values, "note": p["mean"].round(2).values, "avis": p.n.values})


USER_RE = re.compile(r"\bA[0-9A-Z]{8,20}\b"); PROD_RE = re.compile(r"\bB[0-9A-Z]{9}\b|\b\d{9}[0-9X]\b")


def llm_rewrite(question, text, table):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:  # noqa
            key = None
    if not key:
        return None
    try:
        import anthropic
        ctx = table.head(8).to_string(index=False) if table is not None and len(table) else ""
        msg = anthropic.Anthropic(api_key=key).messages.create(
            model=os.environ.get("LLM_MODEL", "claude-sonnet-5-5"), max_tokens=400,
            messages=[{"role": "user", "content": "Réponds en français, brièvement, uniquement à partir du contexte.\n\n"
                       f"Contexte : {text}\n{ctx}\n\nQuestion : {question}"}])
        return msg.content[0].text
    except Exception as e:  # noqa
        print("LLM indisponible :", e)
        return None


def answer(S, question, use_llm=False):
    """Routeur de questions : choisit l'outil graphe/NLP adapté puis formule la réponse."""
    up = question.upper()
    uids = [x for x in USER_RE.findall(up) if x in S.u_idx]
    pids = [x for x in PROD_RE.findall(up) if x in S.p_idx]
    low = question.lower()
    has = lambda *w: any(x in low for x in w)
    table, intent = None, ""
    if uids and has("similaire", "similar", "goût", "gout", "taste", "ressembl"):
        intent = "utilisateurs similaires"; table = similar_users(S, uids[0])
        text = f"Utilisateurs aux goûts les plus proches de **{uids[0]}** (similarité des embeddings du GNN) :"
    elif uids:
        intent = "recommandation"; r = recommend(S, uids[0], 8)
        table = r[["rang", "ProductId", "thème", "note", "avis", "voisins"]]
        text = f"Produits recommandés pour **{uids[0]}** (LightGCN + BPR, produits déjà notés exclus) :"
    elif pids and has("aussi", "also", "autres", "similaire", "similar", "alternative", "ressembl"):
        intent = "produits liés"; table, n = also_liked(S, pids[0])
        text = f"{n} utilisateurs ont aimé **{pids[0]}**. Voici les autres produits qu'ils ont aimés :"
    elif pids:
        intent = "fiche produit"; ps = S.pstats.loc[pids[0]]
        text = (f"**{pids[0]}** : thème « {ps.theme} », note moyenne {ps['mean']:.2f}★ sur {int(ps.n)} avis. "
                f"Avis représentatif : « {ps.example} »")
    elif has("éviter", "eviter", "pire", "mauvais", "worst", "avoid", "decevant", "décevant"):
        intent = "produits à éviter"; table = worst_products(S)
        text = "Produits les moins bien notés (au moins 10 avis, note corrigée par le nombre d'avis) :"
    elif has("communaut", "communit"):
        intent = "communautés"; table = S.comm_table
        text = "Communautés d'utilisateurs détectées par Louvain :" if table is not None else "Pas de table de communautés disponible."
    elif has("meilleur", "top", "best", "mieux") and not has(" pour ", " for ", "café", "coffee", "tea", "thé", "chien", "dog", "chat"):
        intent = "meilleurs produits"; table = best_products(S)
        text = "Produits les mieux notés (note bayésienne) :"
    else:
        intent = "recherche sémantique"; table = search_products(S, question)
        text = ("Meilleurs produits pour votre requête (recherche sémantique dans les avis puis agrégation par produit) :"
                if len(table) else "Je n'ai trouvé aucun avis correspondant. Essayez avec d'autres mots (anglais ou français).")
    final = llm_rewrite(question, text, table) if use_llm else None
    return {"intent": intent, "text": final or text, "table": table}


# --------------------------------------------------------------------------------------
# Onglets
# --------------------------------------------------------------------------------------
def tab_overview(S):
    df = S.df
    st.markdown(
        "<div class='section-heading'><div><div class='section-eyebrow'>VOTRE ESPACE ANALYTIQUE</div>"
        "<h2>Vue d'ensemble</h2><p>Un aperçu des avis, du catalogue et des tendances de satisfaction.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    c = st.columns(5)
    c[0].metric("Avis", f"{len(df):,}".replace(",", " "))
    c[1].metric("Utilisateurs", f"{len(S.users):,}".replace(",", " "))
    c[2].metric("Produits", f"{len(S.prods):,}".replace(",", " "))
    c[3].metric("Thèmes", df.theme.nunique())
    c[4].metric("Communautés", S.meta.get("n_communities", df.user_comm.nunique()))
    a, b = st.columns(2)
    d = df.Score.value_counts().sort_index().rename_axis("note").reset_index(name="avis")
    fig = px.bar(d, x="note", y="avis", title="Distribution des notes", color_discrete_sequence=["#795b7e"])
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    a.plotly_chart(fig)
    t = df.groupby("theme").Score.agg(["mean", "size"]).query("size >= 30").sort_values("mean").tail(15).reset_index()
    fig = px.bar(t, x="mean", y="theme", orientation="h", title="Note moyenne par thème",
                 color="mean", color_continuous_scale=["#e7dce8", "#795b7e"], labels={"mean": "note moyenne", "theme": ""})
    fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), coloraxis_showscale=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    b.plotly_chart(fig)
    st.info("Utilisez les onglets ci-dessus : **Sentiment** pour classer un avis, **Recommandations** pour un utilisateur, "
            "**Q&A** pour interroger le graphe en langage naturel.")


def tab_sentiment(S):
    st.markdown(
        "<div class='section-heading'><div><div class='section-eyebrow'>COMPRÉHENSION DU LANGAGE</div>"
        "<h2>Analyse de sentiment</h2><p>Évaluez la tonalité d'un avis ou traitez un fichier CSV.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    model = load_sentiment()
    st.caption(f"Modèle actif : **{model[0]}**")
    st.session_state.setdefault("review_text", "")
    left, right = st.columns([3, 2])
    with left:
        st.text_area("Avis à analyser (en anglais)", key="review_text", height=160,
                     placeholder="Ex. : The coffee was fresh and smelled amazing…")
        cols = st.columns(3)
        for col, (name, txt) in zip(cols, EXAMPLES.items()):
            col.button(name, key=f"ex_{name}", on_click=lambda t=txt: st.session_state.update(review_text=t))
        st.button("Analyser", type="primary", key="go_sent")
    text = st.session_state.get("review_text", "").strip()
    with right:
        if text:
            p = predict_sentiment(model, [text], S.labels)[0]
            lab = S.labels[int(p.argmax())]
            emoji, cls, _ = LABEL_STYLE.get(lab, ("", "neu", "#999"))
            st.markdown(f"<div style='margin-top:1.7rem'><span class='pill {cls}'>{emoji} {lab.upper()}</span> "
                        f"<span class='muted'>confiance {p.max():.0%}</span></div>", unsafe_allow_html=True)
            d = pd.DataFrame({"classe": S.labels, "probabilité": p})
            fig = px.bar(d, x="probabilité", y="classe", orientation="h", color="classe",
                         color_discrete_map={l: LABEL_STYLE.get(l, ("", "", "#999"))[2] for l in S.labels}, range_x=[0, 1])
            fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=210,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig)
        else:
            st.markdown("<div class='muted' style='margin-top:2rem'>Saisissez un avis ou choisissez un exemple.</div>", unsafe_allow_html=True)
    with st.expander("📁 Tester un fichier CSV d'avis (colonne « text »)"):
        up = st.file_uploader("Fichier CSV", type=["csv"], key="csv_sent")
        if up is not None:
            try:
                d = pd.read_csv(up)
                col = "text" if "text" in d.columns else d.columns[0]
                d = d.head(500).copy(); d[col] = d[col].fillna("").astype(str)
                P = predict_sentiment(model, d[col].tolist(), S.labels)
                d["sentiment"] = [S.labels[i] for i in P.argmax(1)]
                d["confiance"] = P.max(1).round(3)
                st.dataframe(d)
                st.download_button("Télécharger les résultats", d.to_csv(index=False).encode("utf-8"), "sentiment_resultats.csv", "text/csv")
            except Exception as e:  # noqa
                st.error(f"Fichier non lisible : {e}")


def tab_reco(S):
    st.markdown(
        "<div class='section-heading'><div><div class='section-eyebrow'>DÉCOUVERTE PERSONNALISÉE</div>"
        "<h2>Recommandations</h2><p>Explorez les produits suggérés par le graphe pour chaque profil.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    top_users = S.uprof.sort_values("n", ascending=False).head(300).index.tolist()
    testable = [u for u in top_users if u in S.heldout]
    st.session_state.setdefault("reco_user", top_users[0])

    def pick_random():
        st.session_state["reco_user"] = random.choice(testable or top_users)

    c1, c2, c3 = st.columns([3, 2, 1])
    c1.selectbox("Utilisateur", top_users, key="reco_user",
                 format_func=lambda u: f"{u}  ·  {int(S.uprof.loc[u, 'n'])} avis")
    custom = c2.text_input("…ou saisir un UserId", key="reco_custom", placeholder="ex. A3OXHLG6DIBRW8")
    c3.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    c3.button("🎲 Au hasard", on_click=pick_random, key="rand_user")
    cc = st.columns([1, 1, 2])
    k = cc[0].slider("Nombre de produits", 3, 20, 8)
    use_pop = cc[1].toggle("Bonus de popularité", value=S.beta > 0, help="Hybride BPR + popularité (coefficient choisi sur la validation).")
    uid = st.session_state["reco_user"]
    if custom.strip():
        c = custom.strip().upper()
        if c in S.u_idx:
            uid = c
        else:
            st.warning("Cet UserId n'est pas dans le graphe (il faut au moins quelques avis pour y figurer).")
    prof = S.uprof.loc[uid]
    m = st.columns(4)
    m[0].metric("Avis écrits", int(prof.n)); m[1].metric("Note moyenne donnée", f"{prof['mean']:.2f}★")
    m[2].metric("Thème préféré", prof.fav); m[3].metric("Communauté", "-" if pd.isna(prof.comm) else int(prof.comm))

    r = recommend(S, uid, k, use_pop)
    held = S.heldout.get(uid, set())
    if held:
        n_hit = int(r.confirmé.sum())
        st.success(f"Test : {n_hit} recommandation(s) sur {len(r)} correspondent à des produits que cet utilisateur a réellement aimés "
                   f"(avis masqués à l'entraînement, {len(held)} au total).")
    else:
        st.caption("Aucun avis masqué pour cet utilisateur : pas de vérification possible.")
    for _, row in r.iterrows():
        with st.container(border=True):
            a, b, c = st.columns([1, 4, 3])
            a.markdown(f"<div class='rank'>#{row['rang']}</div>", unsafe_allow_html=True)
            stars = "★" * int(round(row.note)) + "☆" * (5 - int(round(row.note)))
            b.markdown(f"**{row.ProductId}** &nbsp; <span class='badge'>{html.escape(str(row['thème']))}</span><br>"
                       f"{stars} {row.note:.2f} · {row.avis} avis", unsafe_allow_html=True)
            badges = f"<span class='badge'>👥 aimé par {row.voisins}/20 utilisateurs proches</span>"
            if row.confirmé:
                badges += " <span class='badge ok'>✅ confirmé dans le jeu de test</span>"
            c.markdown(badges, unsafe_allow_html=True)
            st.caption("Avis représentatif : « " + (row.exemple[:140] or "…") + " »")
    with st.expander("👥 Utilisateurs aux goûts similaires"):
        st.dataframe(similar_users(S, uid, 8))


def tab_qa(S):
    st.markdown(
        "<div class='section-heading'><div><div class='section-eyebrow'>EXPLORATION EN LANGAGE NATUREL</div>"
        "<h2>Assistant de recherche</h2><p>Posez une question pour explorer les avis et les liens du graphe.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.caption("Interrogez le graphe en langage naturel (français ou anglais). Les questions utilisent des UserId / ProductId "
               "du jeu de données quand c'est pertinent.")
    ex_u = S.uprof.sort_values("n", ascending=False).index[0]
    ex_p = S.pstats.sort_values("n", ascending=False).index[0]
    suggestions = ["Quels produits sont les mieux notés pour le café ?", "Best dark roast coffee",
                   "Quels produits faut-il éviter ?", f"Recommande des produits à {ex_u}",
                   f"Quels utilisateurs ont des goûts similaires à {ex_u} ?", f"Quels autres produits aiment les fans de {ex_p} ?",
                   "Quelles sont les communautés d'utilisateurs ?"]
    st.session_state.setdefault("messages", [])
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        cols[i % 3].button(s, key=f"sug_{i}", on_click=lambda q=s: st.session_state.update(pending_q=q))
    use_llm = st.toggle("Reformuler la réponse avec un LLM (nécessite ANTHROPIC_API_KEY)", value=False, key="use_llm")
    q = st.chat_input("Posez votre question…")
    if st.session_state.get("pending_q"):
        q = st.session_state.pop("pending_q")
    if q:
        st.session_state["messages"].append({"role": "user", "text": q})
        with st.spinner("Recherche dans le graphe…"):
            res = answer(S, q, use_llm)
        st.session_state["messages"].append({"role": "assistant", **res})
    for m in st.session_state["messages"]:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.write(m["text"])
            else:
                st.markdown(m["text"])
                if m.get("table") is not None and len(m["table"]):
                    st.dataframe(m["table"])
                st.caption(f"Outil utilisé : {m['intent']}")


def tab_communities(S):
    st.markdown(
        "<div class='section-heading'><div><div class='section-eyebrow'>STRUCTURE DU RÉSEAU</div>"
        "<h2>Communautés</h2><p>Visualisez les groupes d'utilisateurs et leurs relations avec les produits.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    if S.comm_table is not None:
        st.subheader("Profil des communautés (Louvain)")
        st.dataframe(S.comm_table)
        num = S.comm_table.select_dtypes("number").columns
        for c in ("utilisateurs", "noeuds"):
            if c in S.comm_table.columns:
                fig = px.bar(S.comm_table, x=S.comm_table.columns[0].__str__(), y=c, title=f"Taille des communautés ({c})",
                             color_discrete_sequence=["#795b7e"])
                fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig)
                break
    g_html = ART / "graph_communities.html"
    g_png = ART / "graph_communities.png"
    st.subheader("Graphe utilisateurs-produits")
    if g_html.exists():
        components.html(g_html.read_text(encoding="utf-8"), height=720, scrolling=True)
    elif g_png.exists():
        st.image(str(g_png))
    else:
        st.info("Aucune visualisation du graphe fournie.")


def tab_eval(S):
    m = S.meta
    st.markdown(
        "<div class='section-heading'><div><div class='section-eyebrow'>QUALITÉ DES MODÈLES</div>"
        "<h2>Évaluation</h2><p>Consultez les performances et les indicateurs disponibles.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.subheader("Sentiment")
    if m.get("sentiment"):
        st.dataframe(pd.DataFrame(m["sentiment"]))
    c1, c2 = st.columns(2)
    if m.get("rating"):
        c1.subheader("Prédiction de notes"); c1.dataframe(pd.DataFrame(m["rating"]))
    if m.get("ranking"):
        c2.subheader("Recommandation (top-10)"); rk = pd.DataFrame(m["ranking"]); c2.dataframe(rk)
        key = rk.columns[0]
        fig = px.bar(rk, x=key, y="Recall@10", color_discrete_sequence=["#795b7e"], title="Recall@10")
        fig.update_layout(margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        c2.plotly_chart(fig)
    st.caption("La popularité est une baseline très forte sur ce type de données : l'intérêt d'un modèle de graphe se juge aussi "
               "sur la diversité et la couverture du catalogue, pas seulement sur le Recall@10.")


def main():
    if not (ART / "meta.json").exists():
        st.error("Fichiers du modèle introuvables. Exécutez le notebook d'export pour générer le dossier `artifacts/`.")
        st.stop()
    S = load_store()
    with st.sidebar:
        st.markdown(
            "<div class='sidebar-brand'><div class='sidebar-logo'>🍽</div><div>"
            "<div class='sidebar-brand-name'>FoodReviews AI</div>"
            "<div class='sidebar-brand-sub'>Espace d'analyse</div></div></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='sidebar-label'>Espace de travail</div>", unsafe_allow_html=True)
        pages = {
            "🏠  Vue générale": tab_overview,
            "💬  Sentiment": tab_sentiment,
            "🎯  Recommandations": tab_reco,
            "🧠  Q&A": tab_qa,
            "🕸️  Communautés": tab_communities,
            "📊  Évaluation": tab_eval,
        }
        selected_page = st.radio(
            "Navigation",
            options=list(pages),
            key="active_page",
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown("<div class='sidebar-label'>Données du projet</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='sidebar-stats'><div class='sidebar-stat'><strong>{len(S.df):,}</strong>"
            f"<span>Avis analysés</span></div><div class='sidebar-stat'><strong>{len(S.prods):,}</strong>"
            f"<span>Produits</span></div></div>".replace(",", " "),
            unsafe_allow_html=True,
        )
        encoder_ready = S.E is not None and load_encoder() is not None
        encoder_name = "Sentence-BERT" if encoder_ready else "TF-IDF (mode dégradé)"
        status_class = "" if encoder_ready else " fallback"
        st.markdown(
            f"<div class='sidebar-label'>État du système</div><div class='sidebar-status'>"
            f"<span class='status-dot{status_class}'></span>Recherche · {html.escape(encoder_name)}</div>",
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown(
            "<div class='sidebar-foot'>Source : Amazon Fine Food Reviews<br>"
            "Analyse NLP · Recommandation par graphe</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<div class='hero'><div class='hero-kicker'>PLATEFORME D'ANALYSE · NLP & GRAPHES</div>"
        "<h1>FoodReviews <span>AI</span></h1>"
        "<p>Explorez les avis clients, découvrez des recommandations personnalisées et interrogez "
        "les données en langage naturel.</p>"
        "<div class='hero-meta'><span>500 000+ avis Amazon</span><span>Analyse de sentiment</span>"
        "<span>Recommandation par graphe</span></div></div>",
        unsafe_allow_html=True,
    )
    pages[selected_page](S)


main()
