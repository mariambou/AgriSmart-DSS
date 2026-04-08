import os, joblib, warnings
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import random
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
#  CHEMINS
# ══════════════════════════════════════════════════════════════
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "df_encoded1.csv")

CROPS_LIST = ["corn", "wheat", "rice", "soybean"]

PKL_YIELD_MODELS = {c: os.path.join(BASE_DIR, f"rf_yield_{c}.pkl") for c in CROPS_LIST}
PKL_YIELD_FEATS  = {c: os.path.join(BASE_DIR, f"rf_feat_{c}.pkl")  for c in CROPS_LIST}

PKL_PRICE_MODELS = {c: os.path.join(BASE_DIR, f"gb_price_{c}.pkl") for c in CROPS_LIST}
PKL_PRICE_FEATS  = {c: os.path.join(BASE_DIR, f"gb_feat_{c}.pkl")  for c in CROPS_LIST}

# ===============================
# TRAIN
# ===============================
def train_and_save():
    df = pd.read_csv(CSV_PATH)

    for crop in CROPS_LIST:
        sub = df[df[f"Product_{crop}"] == 1].copy()

        # ===== YIELD =====
        y_y = sub["Crop_Yield_ton"]
        X_y = sub.drop(columns=["Crop_Yield_ton"], errors="ignore")

        rs1 = random.randint(0, 1000)

        Xtr, Xte, ytr, yte = train_test_split(
            X_y, y_y, test_size=0.2, random_state=rs1
        )

        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            random_state=random.randint(0, 1000),
            n_jobs=-1
        )

        rf.fit(Xtr, ytr)

        joblib.dump(rf, PKL_YIELD_MODELS[crop])
        joblib.dump(list(Xtr.columns), PKL_YIELD_FEATS[crop])

        # ===== PRICE =====
        y_p = sub["Market_Price_per_ton"]
        X_p = sub.drop(columns=["Market_Price_per_ton"], errors="ignore")

        rs2 = random.randint(0, 1000)

        Xtr2, Xte2, ytr2, yte2 = train_test_split(
            X_p, y_p, test_size=0.2, random_state=rs2
        )

        gb = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            random_state=random.randint(0, 1000)
        )

        gb.fit(Xtr2, ytr2)

        joblib.dump(gb, PKL_PRICE_MODELS[crop])
        joblib.dump(list(Xtr2.columns), PKL_PRICE_FEATS[crop])


# ===============================
# CONTROL TRAINING
# ===============================
RETRAIN = True   # 🔥 change ici

if RETRAIN:
    train_and_save()

# ===============================
# LOAD MODELS
# ===============================
models_yield = {c: joblib.load(PKL_YIELD_MODELS[c]) for c in CROPS_LIST}
feats_yield  = {c: joblib.load(PKL_YIELD_FEATS[c])  for c in CROPS_LIST}

models_price = {c: joblib.load(PKL_PRICE_MODELS[c]) for c in CROPS_LIST}
feats_price  = {c: joblib.load(PKL_PRICE_FEATS[c])  for c in CROPS_LIST}

df = pd.read_csv(CSV_PATH)
MEDIANS = df.median(numeric_only=True).to_dict()




# ===============================
# BUILD ROW (FIX BUG HERE)
# ===============================
def build_row(features, crop=None):
    row = {}

    for col in features:

        # 🚫 IMPORTANT : NE PAS UTILISER TARGETS
        if col in ["Crop_Yield_ton", "Market_Price_per_ton"]:
            continue

        if col == "Soil_pH":
            row[col] = soil_ph
        elif col == "Soil_Moisture":
            row[col] = soil_moisture
        elif col == "Temperature_C":
            row[col] = temperature
        elif col == "Rainfall_mm":
            row[col] = rainfall
        elif col == "Fertilizer_Usage_kg":
            row[col] = fertilizer
        elif col == "Pesticide_Usage_kg":
            row[col] = pesticide

        elif col.startswith("Product_"):
            row[col] = 1.0 if col == f"Product_{crop}" else 0.0

        else:
            row[col] = MEDIANS.get(col, 0.0)

    return pd.DataFrame([row])[features]

# ===============================
# PREDICTIONS
# ===============================
def predict_yield(crop):
    row = build_row(feats_yield[crop], crop)
    return float(models_yield[crop].predict(row)[0])

def predict_price(crop):
    row = build_row(feats_price[crop], crop)
    return float(models_price[crop].predict(row)[0])



# ══════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════
ALL_CROPS = ["corn", "wheat", "rice", "soybean"]
CROP_META = {
    "corn":    {"label":"Maïs",   "icon":"🌽", "color":"#D48B0A", "rgb":(212,139,10), "light":"#FEF3D8"},
    "wheat":   {"label":"Blé",    "icon":"🌾", "color":"#8B6914", "rgb":(139,105,20), "light":"#F5EDD4"},
    "rice":    {"label":"Riz",    "icon":"🍚", "color":"#1A7A5A", "rgb":(26,122,90),  "light":"#D6F0E6"},
    "soybean": {"label":"Soja",   "icon":"🫘", "color":"#4A7C20", "rgb":(74,124,32),  "light":"#E0EDD0"},
}
MEDALS = ["🥇","🥈","🥉","4️⃣"]

OPTIMA = {
    "corn":    {"pH":6.0,"Hum":40,"Tmp":25,"Pluie":120,"Eng":150,"Pest":80},
    "wheat":   {"pH":6.5,"Hum":30,"Tmp":18,"Pluie": 80,"Eng":120,"Pest":60},
    "rice":    {"pH":5.5,"Hum":55,"Tmp":28,"Pluie":200,"Eng":100,"Pest":50},
    "soybean": {"pH":6.8,"Hum":35,"Tmp":22,"Pluie": 90,"Eng": 60,"Pest":30},
}

CROP_TIPS = {
    "corn": [
        "Semer entre le 15 avril et le 15 mai quand le sol dépasse 10°C",
        "Maintenir un espacement de 70–75 cm entre les rangs",
        "Apporter l'azote en deux fois : à la levée et au stade 8 feuilles",
        "Irriguer abondamment pendant la floraison — c'est le stade le plus sensible",
        "Surveiller la pyrale et la sésamie en juillet–août",
    ],
    "wheat": [
        "Semis d'automne recommandé : entre octobre et mi-novembre",
        "Préparer un lit de semences fin et bien tassé avant le semis",
        "Appliquer un herbicide anti-graminées en automne si nécessaire",
        "Fractionner l'engrais azoté : tallage + montaison + épiaison",
        "Traiter contre la rouille jaune dès l'apparition des premières taches",
    ],
    "rice": [
        "Inonder la parcelle 2 à 3 semaines avant le repiquage",
        "Repiquer des plants de 20–25 jours, 3 à 4 plants par poquet",
        "Maintenir une lame d'eau de 5 à 10 cm pendant toute la végétation",
        "Drainer 2 semaines avant la récolte pour faciliter le passage des machines",
        "Récolter quand 85 à 90% des grains sont bien dorés",
    ],
    "soybean": [
        "Inoculer les semences avec Rhizobium juste avant le semis",
        "Attendre que le sol soit à 15°C minimum — semis en mai ou juin",
        "Respecter un espacement de 40–50 cm entre rangs, 5–8 cm entre plants",
        "Ne pas apporter d'azote — la plante produit le sien naturellement",
        "Surveiller les pucerons et les acariens pendant les périodes sèches",
    ],
}

PT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(248,250,245,0.6)",
    font=dict(family="'DM Sans', sans-serif", size=11, color="#1A2E1B"),
    margin=dict(l=16, r=16, t=44, b=16),
)


# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AgriSmart — Conseil Agricole",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Playfair+Display:wght@600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'DM Sans', -apple-system, sans-serif; color: #1A2E1B; }

.stApp {
    background-color: #F0F4EB;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix type='saturate' values='0'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23n)' opacity='0.025'/%3E%3C/svg%3E");
    background-repeat: repeat;
}

/* ── Sidebar ─────────────────────────────────────────── */
section[data-testid="stSidebar"] { background: #0F1E10 !important; border-right: 1px solid rgba(255,255,255,0.06) !important; }
section[data-testid="stSidebar"] > div { padding-top: 0 !important; }
section[data-testid="stSidebar"] * { color: #C8DEC9 !important; }
section[data-testid="stSidebar"] .stMarkdown p {
    font-size: 0.72rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 1.5px !important;
    color: rgba(200,222,201,0.5) !important; margin-bottom: 4px !important;
}
section[data-testid="stSidebar"] .stSlider > div { margin-bottom: 2px !important; }
section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div:first-child {
    background: rgba(80,160,80,0.2) !important; height: 3px !important;
}
section[data-testid="stSidebar"] [data-baseweb="slider"] [role="slider"] {
    background: #5CA85C !important; border: 2px solid #fff !important;
    width: 14px !important; height: 14px !important;
}
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
section[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
    font-size: 0.65rem !important; color: rgba(200,222,201,0.35) !important;
}
section[data-testid="stSidebar"] .stFormSubmitButton > button {
    background: linear-gradient(135deg, #2E6B30 0%, #4A9448 100%) !important;
    color: #fff !important; border: none !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 700 !important;
    font-size: 0.9rem !important; padding: 14px 0 !important; width: 100% !important;
    transition: all 0.2s ease !important; box-shadow: 0 4px 14px rgba(46,107,48,0.4) !important;
}
section[data-testid="stSidebar"] .stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #255228 0%, #3D7C3B 100%) !important;
    box-shadow: 0 6px 20px rgba(46,107,48,0.5) !important; transform: translateY(-1px) !important;
}
section[data-testid="stSidebar"] hr { border: none !important; border-top: 1px solid rgba(255,255,255,0.07) !important; margin: 12px 0 !important; }

/* ── Header ─────────────────────────────────────────── */
.header-bar { position: relative; background: #0F1E10; border-radius: 16px; overflow: hidden; margin-bottom: 20px; }
.header-bg { position: absolute; inset: 0; background: radial-gradient(ellipse 60% 120% at 80% 50%, rgba(74,148,72,0.18) 0%, transparent 70%), radial-gradient(ellipse 40% 80% at 15% 80%, rgba(212,139,10,0.08) 0%, transparent 60%); pointer-events: none; }
.header-content { position: relative; z-index: 2; display: flex; align-items: center; justify-content: space-between; padding: 36px 40px; gap: 24px; }
.header-left { display: flex; align-items: center; gap: 18px; }
.header-logo { width: 64px; height: 64px; background: linear-gradient(135deg, #2E6B30, #5CA85C); border-radius: 16px; display: flex; align-items: center; justify-content: center; font-size: 2rem; box-shadow: 0 4px 20px rgba(46,107,48,0.55); flex-shrink: 0; }
.header-eyebrow { font-size: 0.65rem; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; color: #5CA85C; margin-bottom: 6px; }
.header-title { font-family: 'Playfair Display', Georgia, serif; font-size: 2.3rem; font-weight: 800; color: #F0F7F0; line-height: 1; margin: 0; letter-spacing: -0.3px; }
.header-title span { color: #7ECB7E; }
.header-sub { font-size: 0.88rem; color: rgba(200,222,201,0.5); margin-top: 8px; font-weight: 400; }
.header-right { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
.hbadge { background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.1); border-radius: 20px; padding: 8px 20px; font-size: 0.78rem; font-weight: 500; color: rgba(200,222,201,0.7); white-space: nowrap; }

/* ── Section headers ─────────────────────────────────── */
.sec { display: flex; align-items: center; gap: 12px; margin: 32px 0 16px; }
.sec-icon { width: 32px; height: 32px; background: linear-gradient(135deg, #2E6B30, #4A9448); border-radius: 9px; display: flex; align-items: center; justify-content: center; font-size: 0.95rem; flex-shrink: 0; box-shadow: 0 2px 8px rgba(46,107,48,0.3); }
.sec-title { font-family: 'Playfair Display', serif; font-size: 1.05rem; font-weight: 700; color: #0F1E10; letter-spacing: -0.2px; }
.sec-line { flex: 1; height: 1px; background: linear-gradient(to right, #C8D8C0, transparent); }

/* ── Recommendation card ─────────────────────────────── */
.rec-card { background: #fff; border-radius: 16px; padding: 24px 28px; border: 1px solid rgba(46,107,48,0.15); box-shadow: 0 2px 20px rgba(15,30,16,0.06); position: relative; overflow: hidden; }
.rec-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #2E6B30, #7ECB7E, #D4940A); border-radius: 16px 16px 0 0; }
.rec-badge { display: inline-flex; align-items: center; gap: 5px; background: linear-gradient(135deg, #EAF5EA, #D6EDD6); border: 1px solid rgba(46,107,48,0.2); border-radius: 20px; padding: 3px 12px; font-size: 0.62rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; color: #2E6B30; margin-bottom: 12px; }
.rec-crop-name { font-family: 'Playfair Display', serif; font-size: 2rem; font-weight: 800; color: #0F1E10; line-height: 1; margin-bottom: 16px; display: flex; align-items: center; gap: 10px; }
.rec-stats { display: flex; border: 1px solid #EEF3EC; border-radius: 12px; overflow: hidden; flex-wrap: wrap; }
.rs-item { flex: 1; min-width: 80px; padding: 12px 16px; border-right: 1px solid #EEF3EC; background: #FAFCF8; }
.rs-item:last-child { border-right: none; }
.rs-val { font-family: 'Playfair Display', serif; font-size: 1.4rem; font-weight: 700; line-height: 1; }
.rs-lbl { font-size: 0.59rem; text-transform: uppercase; letter-spacing: 1.2px; color: #8A9E8B; font-weight: 600; margin-top: 3px; }

/* ── KPI cards ───────────────────────────────────────── */
.kpi-card { background: #fff; border-radius: 14px; padding: 16px 14px 14px; text-align: center; border: 1px solid rgba(0,0,0,0.06); box-shadow: 0 1px 10px rgba(15,30,16,0.05); height: 100%; transition: transform 0.2s, box-shadow 0.2s; }
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 4px 20px rgba(15,30,16,0.1); }
.kpi-ico  { font-size: 1.3rem; margin-bottom: 6px; display: block; }
.kpi-val  { font-family: 'Playfair Display', serif; font-size: 1.65rem; font-weight: 700; line-height: 1; margin-bottom: 2px; }
.kpi-unit { font-size: 0.58rem; color: #8A9E8B; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; }
.kpi-lbl  { font-size: 0.68rem; color: #5A6E5B; margin-top: 4px; font-weight: 500; }

/* ── Rank cards ──────────────────────────────────────── */
.rank-card { background: #fff; border-radius: 12px; padding: 14px 16px; margin: 6px 0; display: flex; align-items: center; gap: 12px; border: 1px solid rgba(0,0,0,0.06); box-shadow: 0 1px 6px rgba(15,30,16,0.04); transition: all 0.18s ease; cursor: default; }
.rank-card:hover { transform: translateX(4px); box-shadow: 0 4px 16px rgba(15,30,16,0.08); border-color: rgba(46,107,48,0.2); }
.rank-badge { width: 4px; border-radius: 4px; align-self: stretch; flex-shrink: 0; }
.rank-medal { font-size: 1.3rem; flex-shrink: 0; }
.rank-crop-name { font-weight: 600; font-size: 0.88rem; color: #1A2E1B; }
.rank-sub  { font-size: 0.7rem; color: #7A8E7B; margin-top: 2px; }
.rank-val  { font-family: 'Playfair Display', serif; font-size: 1.15rem; font-weight: 700; text-align: right; white-space: nowrap; }

/* ── Action cards ────────────────────────────────────── */
.action-card { border-radius: 14px; padding: 20px 22px; margin: 10px 0; border: 1px solid transparent; }
.ac-ok   { background: #F0F9F0; border-color: rgba(46,107,48,0.2);  border-left: 4px solid #2E6B30; }
.ac-warn { background: #FDF8EE; border-color: rgba(212,148,10,0.2); border-left: 4px solid #D4940A; }
.ac-bad  { background: #FDF0EF; border-color: rgba(192,57,43,0.2);  border-left: 4px solid #C0392B; }
.ac-info { background: #EFF5FD; border-color: rgba(41,128,185,0.2); border-left: 4px solid #2980B9; }
.ac-title { font-weight: 700; font-size: 0.9rem; margin-bottom: 12px; color: #1A2E1B; }
.action-tag { display: inline-block; background: rgba(15,30,16,0.06); border: 1px solid rgba(15,30,16,0.08); border-radius: 6px; padding: 5px 11px; font-size: 0.76rem; font-weight: 500; margin: 3px 3px; color: #2C3E2D; }

/* ── Diagnostic cards ────────────────────────────────── */
.diag-card { background: #fff; border-radius: 12px; padding: 14px 16px; border: 1px solid rgba(0,0,0,0.06); box-shadow: 0 1px 8px rgba(15,30,16,0.04); display: flex; gap: 12px; align-items: flex-start; }
.diag-ok   { border-left: 4px solid #2E6B30; }
.diag-warn { border-left: 4px solid #D4940A; }
.diag-bad  { border-left: 4px solid #C0392B; }
.diag-ico  { font-size: 1.25rem; flex-shrink: 0; margin-top: 1px; }
.diag-lbl  { font-size: 0.64rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #5A6E5B; }
.diag-msg  { font-size: 0.8rem; color: #3A4E3B; margin-top: 2px; line-height: 1.5; }

/* ── Tip cards ───────────────────────────────────────── */
.tip-card { background: #fff; border-radius: 12px; padding: 14px 18px; margin: 5px 0; border: 1px solid #E8F0E4; box-shadow: 0 1px 6px rgba(15,30,16,0.04); display: flex; align-items: flex-start; gap: 10px; transition: background 0.15s; }
.tip-card:hover { background: #F8FCF6; }
.tip-num { width: 22px; height: 22px; border-radius: 50%; background: linear-gradient(135deg, #2E6B30, #4A9448); color: #fff; font-size: 0.65rem; font-weight: 700; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 1px; }
.tip-text { font-size: 0.82rem; color: #2C3E2D; line-height: 1.55; }

/* ── Welcome screen ──────────────────────────────────── */
.welcome-card { background: #fff; border-radius: 16px; padding: 28px 22px; text-align: center; border: 1px solid rgba(0,0,0,0.06); box-shadow: 0 2px 14px rgba(15,30,16,0.05); transition: all 0.22s ease; height: 100%; }
.welcome-card:hover { transform: translateY(-4px); box-shadow: 0 8px 28px rgba(15,30,16,0.1); }
.welcome-ico { font-size: 2rem; margin-bottom: 12px; display: block; }
.welcome-ttl { font-family: 'Playfair Display', serif; font-size: 0.95rem; font-weight: 700; color: #0F1E10; margin-bottom: 8px; }
.welcome-dsc { font-size: 0.76rem; color: #5A6E5B; line-height: 1.6; }

/* ── Tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { background: #E8EFE3 !important; border-radius: 10px !important; padding: 4px !important; gap: 2px !important; border: 1px solid rgba(0,0,0,0.05) !important; }
.stTabs [data-baseweb="tab"] { font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; font-size: 0.78rem !important; border-radius: 7px !important; color: #5A6E5B !important; padding: 6px 14px !important; }
.stTabs [aria-selected="true"] { background: #fff !important; color: #0F1E10 !important; box-shadow: 0 1px 8px rgba(15,30,16,0.08) !important; }

/* ── Sidebar brand & sections ────────────────────────── */
.sb-brand { background: linear-gradient(180deg, rgba(74,148,72,0.15) 0%, transparent 100%); padding: 18px 16px 14px; margin: 0 -1rem 8px; border-bottom: 1px solid rgba(255,255,255,0.06); }
.sb-section { display: flex; align-items: center; gap: 8px; padding: 10px 0 6px; margin-bottom: 2px; }
.sb-section-line { flex: 1; height: 1px; background: rgba(255,255,255,0.08); }
.sb-section-label { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 2px; color: rgba(200,222,201,0.45) !important; }

/* ── Animations & misc ───────────────────────────────── */
.stAlert { border-radius: 10px !important; font-size: 0.82rem !important; }
.element-container { animation: fadeUp 0.3s ease both; }
@keyframes fadeUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #C8D8C0; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #A8C0A8; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-bar">
  <div class="header-bg"></div>
  <div class="header-content">
    <div class="header-left">
      <div class="header-logo">🌿</div>
      <div>
        <div class="header-eyebrow">Conseil Agricole Intelligent</div>
        <div class="header-title">Agri<span>Smart</span></div>
        <div class="header-sub">Décrivez votre terrain et obtenez des recommandations personnalisées pour votre saison</div>
      </div>
    </div>
    <div class="header-right">
      <div class="hbadge">🌽 Maïs</div>
      <div class="hbadge">🌾 Blé</div>
      <div class="hbadge">🍚 Riz</div>
      <div class="hbadge">🫘 Soja</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
      <div style='font-family:Playfair Display,serif;font-size:1.1rem;font-weight:700;color:#7ECB7E'>
        👨‍🌾 Ma Parcelle
      </div>
      <div style='font-size:0.68rem;color:rgba(200,222,201,0.38);margin-top:3px'>
        Décrivez vos conditions actuelles
      </div>
    </div>""", unsafe_allow_html=True)

    with st.form("form_parcelle"):

        st.markdown('<div class="sb-section"><div class="sb-section-label">🪨 État du Sol</div><div class="sb-section-line"></div></div>', unsafe_allow_html=True)
        soil_ph       = st.slider("Acidité du sol (pH)",       4.0,  9.0,  6.5, 0.1,
                                  help="Sol acide < 6 · Sol neutre 6–7 · Sol basique > 7")
        soil_moisture = st.slider("Humidité du sol (%)",       5.0, 60.0, 30.0, 1.0,
                                  help="Estimez l'humidité actuelle de votre sol en surface")

        st.markdown('<div class="sb-section"><div class="sb-section-label">🌦️ Conditions Météo</div><div class="sb-section-line"></div></div>', unsafe_allow_html=True)
        temperature   = st.slider("Température moyenne (°C)",  5.0, 45.0, 25.0, 0.5)
        rainfall      = st.slider("Pluie mensuelle (mm)",      0.0,300.0,150.0, 5.0)

        st.markdown('<div class="sb-section"><div class="sb-section-label">🧪 Intrants Agricoles</div><div class="sb-section-line"></div></div>', unsafe_allow_html=True)
        fertilizer    = st.slider("Engrais apportés (kg/ha)", 50.0,220.0,125.0, 5.0)
        pesticide     = st.slider("Pesticides (kg/ha)",        0.0, 80.0, 10.0, 1.0)

        st.markdown('<div class="sb-section"><div class="sb-section-label">💶 Marché Local</div><div class="sb-section-line"></div></div>', unsafe_allow_html=True)
        competitor_price   = st.slider("Prix du marché local (€/t)", 100.0,500.0,300.0,  5.0,
                                       help="Prix pratiqué dans votre région")
        consumer_trend     = st.slider("Demande des acheteurs",        50.0,150.0,100.0,  1.0,
                                       help="100 = normale · au-dessus = forte demande")
        economic_indicator = st.slider("Situation économique",          0.5,  1.5,  1.0, 0.05,
                                       help="1.0 = normale · au-dessus = favorable")

        submitted = st.form_submit_button("🔍  Obtenir mes Recommandations", use_container_width=True)


    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.63rem;color:rgba(200,222,201,0.28);line-height:1.8;padding:2px 4px'>
      🌿 AgriSmart — Conseil Agricole<br>
      PFE · Agriculture Intelligente
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def sec(icon, title):
    st.markdown(f"""
    <div class="sec">
      <div class="sec-icon">{icon}</div>
      <span class="sec-title">{title}</span>
      <div class="sec-line"></div>
    </div>""", unsafe_allow_html=True)


def build_row(features, crop=None, for_price=False):
    """
    Construit un DataFrame d'une ligne avec les valeurs des sliders.
    - Pour le yield  : features sans Product_* (modèle spécialisé par culture)
    - Pour le prix   : features sans Product_* mais avec variables marché
    """
    row = {}
    for col in features:
        if   col == "Soil_pH":                  row[col] = soil_ph
        elif col == "Soil_Moisture":            row[col] = soil_moisture
        elif col == "Temperature_C":            row[col] = temperature
        elif col == "Rainfall_mm":              row[col] = rainfall
        elif col == "Fertilizer_Usage_kg":      row[col] = fertilizer
        elif col == "Pesticide_Usage_kg":       row[col] = pesticide
        elif col == "Competitor_Price_per_ton": row[col] = competitor_price
        elif col == "Consumer_Trend_Index":     row[col] = consumer_trend
        elif col == "Economic_Indicator":       row[col] = economic_indicator
        elif col == "Input_Intensity":          row[col] = fertilizer + pesticide
        elif col == "Climate_Stress_Score":     row[col] = (temperature / 45.0) * (rainfall / 300.0)
        elif col == "demand_supply_ratio":
            d = MEDIANS.get("Demand_Index", 124.0)
            s = MEDIANS.get("Supply_Index", 124.0)
            row[col] = d / s if s > 0 else MEDIANS.get("demand_supply_ratio", 0.99)
        elif col == "market_stability":         row[col] = MEDIANS.get("market_stability", 92.95)
        elif col == "Demand_Index":             row[col] = MEDIANS.get("Demand_Index", 124.21)
        elif col == "Supply_Index":             row[col] = MEDIANS.get("Supply_Index", 124.68)
        elif col == "Sustainability_Score":     row[col] = MEDIANS.get("Sustainability_Score", 50.23)
        elif col == "Weather_Impact_Score":     row[col] = MEDIANS.get("Weather_Impact_Score", 50.76)
        elif col == "Seasonal_Factor":          row[col] = MEDIANS.get("Seasonal_Factor", 1.0)
        elif col == "Crop_Yield_ton":           row[col] = MEDIANS.get("Crop_Yield_ton", 5.49)
        # Les colonnes Product_* ne devraient plus apparaître dans les modèles
        # de yield (correction du bug), mais on les gère par sécurité
        elif col.startswith("Product_"):
            row[col] = 1.0 if (crop and col == f"Product_{crop}") else 0.0
        else:
            row[col] = MEDIANS.get(col, 0.0)
    return pd.DataFrame([row])[features]


def predict_yield(crop):
    """Prédit le rendement avec le modèle spécialisé pour cette culture."""
    feats = feats_yield[crop]
    row   = build_row(feats, crop=crop, for_price=False)
    return max(0.0, float(models_yield[crop].predict(row)[0]))


def predict_price(crop):
    """Prédit le prix de marché avec le modèle GB spécialisé."""
    feats = feats_price[crop]
    row   = build_row(feats, crop=crop, for_price=True)
    return max(0.0, float(models_price[crop].predict(row)[0]))


def adq(crop):
    """Score d'adéquation terrain / culture sur 10."""
    uv   = {"pH": soil_ph, "Hum": soil_moisture, "Tmp": temperature,
            "Pluie": rainfall, "Eng": fertilizer, "Pest": pesticide}
    rngs = {"pH":(4,9), "Hum":(5,60), "Tmp":(5,45),
            "Pluie":(0,300), "Eng":(50,220), "Pest":(0,80)}
    sc = [max(0, (1 - abs(uv[k] - OPTIMA[crop][k]) / (rngs[k][1] - rngs[k][0]) * 2) * 10)
          for k in uv]
    return round(float(np.mean(sc)), 1)


def profit_style(p):
    if p >= 2000: return "#1A7A40", "ac-ok"
    if p >= 1000: return "#C4880A", "ac-warn"
    return "#B03428", "ac-bad"


def gauge(val, title, lo, hi, color, sfx=""):
    span = hi - lo
    fig  = go.Figure(go.Indicator(
        mode="gauge+number", value=val,
        title={"text": title, "font": {"size": 10, "color": "#3A4E3B", "family": "DM Sans"}},
        number={"suffix": sfx, "font": {"size": 16, "color": color, "family": "Playfair Display, serif"}},
        gauge={
            "axis":    {"range": [lo, hi], "tickfont": {"size": 7, "color": "#A0B4A0"},
                        "tickcolor": "#C8D8C8", "nticks": 5},
            "bar":     {"color": color, "thickness": 0.55},
            "bgcolor": "#F6F9F4", "borderwidth": 0,
            "steps": [
                {"range": [lo,           lo + span * .33], "color": "#FDF0EF"},
                {"range": [lo + span*.33, lo + span * .66], "color": "#FEF9EC"},
                {"range": [lo + span*.66, hi],              "color": "#EEF8EE"},
            ],
        }
    ))
    fig.update_layout(height=175, margin=dict(l=8, r=8, t=28, b=4),
                      paper_bgcolor="rgba(0,0,0,0)", font={"family": "DM Sans"})
    return fig


# ══════════════════════════════════════════════════════════════
#  ÉCRAN D'ACCUEIL
# ══════════════════════════════════════════════════════════════
if not submitted:
    items = [
        ("🌱","#2E6B30","Quelle culture planter ?",
         "Comparez les 4 cultures sur vos conditions réelles de sol et de climat."),
        ("💶","#D4940A","Combien vais-je gagner ?",
         "Estimez votre revenu à l'hectare avant de semer pour mieux décider."),
        ("🗺️","#1A6B9A","Mon terrain est-il adapté ?",
         "Visualisez l'adéquation entre votre parcelle et les besoins de chaque culture."),
        ("🧭","#8B4513","Que faire concrètement ?",
         "Recevez un plan d'action précis et des conseils adaptés à votre situation."),
    ]
    cols = st.columns(4, gap="medium")
    for col, (ico, clr, ttl, dsc) in zip(cols, items):
        with col:
            st.markdown(f"""
            <div class="welcome-card" style="border-top:3px solid {clr}">
              <span class="welcome-ico">{ico}</span>
              <div class="welcome-ttl">{ttl}</div>
              <div class="welcome-dsc">{dsc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align:center;padding:30px 20px;
                background:linear-gradient(135deg,#F5F9F2,#EEF5EA);
                border-radius:16px;border:1.5px dashed #C0D4B8;'>
      <div style='font-size:2.2rem;margin-bottom:10px'>🌱</div>
      <div style='font-family:Playfair Display,serif;font-size:1.1rem;
                  font-weight:700;color:#0F1E10;margin-bottom:8px'>
        Prêt à analyser votre parcelle ?
      </div>
      <div style='font-size:0.82rem;color:#5A6E5B;line-height:1.7'>
        ⬅️ Décrivez vos conditions dans le panneau de gauche<br>
        puis cliquez sur <strong style="color:#2E6B30">Obtenir mes Recommandations</strong>
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ══════════════════════════════════════════════════════════════
#  PRÉDICTIONS — utilise les nouvelles valeurs des sliders
# ══════════════════════════════════════════════════════════════
results = {}
for crop in ALL_CROPS:
    yv = predict_yield(crop)   # modèle spécialisé par culture
    pv = predict_price(crop)
    results[crop] = {
        "yield":  yv,
        "price":  pv,
        "profit": yv * pv,
        "adeq":   adq(crop),
        # Ligne de référence pour les graphes de sensibilité (yield)
        "row_y":  build_row(feats_yield[crop], crop=crop, for_price=False),
    }

ranked    = sorted(results.items(), key=lambda x: x[1]["profit"], reverse=True)
best_crop = ranked[0][0]
best      = ranked[0][1]
bm        = CROP_META[best_crop]
pc        = profit_style(best["profit"])[0]
adq_col   = "#1A7A40" if best["adeq"] >= 7 else ("#C4880A" if best["adeq"] >= 4 else "#B03428")


# ══════════════════════════════════════════════════════════════
#  ① CULTURE RECOMMANDÉE
# ══════════════════════════════════════════════════════════════
sec("🏆", "Culture Recommandée pour Votre Parcelle")

c_rec, c_g1, c_g2, c_g3 = st.columns([2.5, 1, 1, 1])

with c_rec:
    marge = f"{best['profit']/best['yield']:.0f}" if best["yield"] > 0 else "–"
    st.markdown(f"""
    <div class="rec-card">
      <div class="rec-badge">✦ Meilleur choix pour vos conditions actuelles</div>
      <div class="rec-crop-name"><span>{bm['icon']}</span> {bm['label']}</div>
      <div class="rec-stats">
        <div class="rs-item">
          <div class="rs-val" style="color:{profit_style(best['profit'])[0]}">{best['profit']:.0f} €</div>
          <div class="rs-lbl">Revenu estimé / ha</div>
        </div>
        <div class="rs-item">
          <div class="rs-val" style="color:#C4880A">{best['price']:.0f} €</div>
          <div class="rs-lbl">Prix de vente / t</div>
        </div>
        <div class="rs-item">
          <div class="rs-val" style="color:#1A7A40">{best['yield']:.2f} t</div>
          <div class="rs-lbl">Récolte prévue / ha</div>
        </div>
        <div class="rs-item">
          <div class="rs-val" style="color:{adq_col}">{best['adeq']}/10</div>
          <div class="rs-lbl">Adéquation terrain</div>
        </div>
        <div class="rs-item">
          <div class="rs-val" style="color:#1A6B9A">{marge} €</div>
          <div class="rs-lbl">Gain net / tonne</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

with c_g1:
    st.plotly_chart(gauge(best["profit"], "Revenu (€/ha)", 0, 3500,
                          profit_style(best["profit"])[0], "€"), use_container_width=True)
with c_g2:
    st.plotly_chart(gauge(best["price"], "Prix vente (€/t)", 100, 500, "#C4880A", "€"),
                    use_container_width=True)
with c_g3:
    st.plotly_chart(gauge(best["yield"], "Récolte (t/ha)", 0, 10, "#1A7A40"),
                    use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  ② KPI STRIP
# ══════════════════════════════════════════════════════════════
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
k1, k2, k3, k4, k5 = st.columns(5)
marge_t = f"{best['profit']/best['yield']:.0f}" if best["yield"] > 0 else "–"
kpis = [
    (k1, "💶", f"{best['profit']:.0f}", "€ / ha",    "Revenu total estimé",  profit_style(best["profit"])[0]),
    (k2, "🏷️", f"{best['price']:.0f}",  "€ / tonne", "Prix de vente prévu",  "#C4880A"),
    (k3, "🌾", f"{best['yield']:.2f}",  "t / ha",    "Récolte attendue",     "#1A7A40"),
    (k4, "💰", marge_t,                 "€ / tonne", "Gain net par tonne",   "#1A6B9A"),
    (k5, "🎯", f"{best['adeq']}",       "/ 10",      "Adéquation terrain",   adq_col),
]
for col, ico, val, unit, lbl, clr in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-card" style="border-top:3px solid {clr}">
          <span class="kpi-ico">{ico}</span>
          <div class="kpi-val" style="color:{clr}">{val}</div>
          <div class="kpi-unit">{unit}</div>
          <div class="kpi-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  ③ COMPARAISON + GRAPHE
# ══════════════════════════════════════════════════════════════
sec("🥇", "Comparaison des 4 Cultures")

c_left, c_right = st.columns([1, 2], gap="medium")

with c_left:
    for i, (crop, data) in enumerate(ranked):
        m   = CROP_META[crop]
        clr = profit_style(data["profit"])[0]
        st.markdown(f"""
        <div class="rank-card">
          <div class="rank-badge" style="background:{m['color']}"></div>
          <span class="rank-medal">{MEDALS[i]}</span>
          <div style="flex:1;min-width:0">
            <div class="rank-crop-name">{m['icon']} {m['label']}</div>
            <div class="rank-sub">
              🌾 {data['yield']:.2f} t/ha &nbsp;·&nbsp;
              🏷️ {data['price']:.0f} €/t &nbsp;·&nbsp;
              🎯 {data['adeq']}/10
            </div>
          </div>
          <div>
            <div class="rank-val" style="color:{clr}">{data['profit']:.0f} €</div>
            <div style="font-size:0.6rem;color:#A0B4A0;text-align:right">/ha</div>
          </div>
        </div>""", unsafe_allow_html=True)

    profits = [d["profit"] for _, d in ranked]
    max_e   = max(profits) - min(profits)
    if   max_e < 300: st.warning(f"Les 4 cultures donnent des résultats proches (écart : {max_e:.0f} €/ha)")
    elif max_e < 800: st.info(f"Avantage modéré pour {bm['icon']} {bm['label']} (+{max_e:.0f} €/ha)")
    else:             st.success(f"{bm['icon']} {bm['label']} est nettement plus rentable (+{max_e:.0f} €/ha)")

with c_right:
    lbls   = [f"{CROP_META[c]['icon']} {CROP_META[c]['label']}" for c in ALL_CROPS]
    colors = [CROP_META[c]["color"] for c in ALL_CROPS]
    fig = make_subplots(rows=1, cols=3,
        subplot_titles=("Récolte prévue (t/ha)", "Prix de vente (€/t)", "Revenu estimé (€/ha)"),
        horizontal_spacing=0.07)
    for col_idx, (key, sfx, extra) in enumerate(
        [("yield", "t", None), ("price", "€/t", competitor_price), ("profit", "€/ha", None)], 1
    ):
        vals = [results[c][key] for c in ALL_CROPS]
        fig.add_trace(go.Bar(
            x=lbls, y=vals, marker_color=colors, marker_line_width=0,
            text=[f"{v:.1f}{sfx}" if key == "yield" else f"{v:.0f}€" for v in vals],
            textposition="outside", textfont=dict(size=9, color="#1A2E1B"),
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>%{y:.1f}<extra></extra>",
        ), row=1, col=col_idx)
    fig.update_layout(height=290, **PT)
    fig.update_yaxes(showgrid=True, gridcolor="#E4EDE0", zeroline=False,
                     tickfont=dict(size=9, color="#A0B4A0"))
    fig.update_xaxes(tickfont=dict(size=10))
    fig.update_annotations(font=dict(size=10, color="#1A2E1B"))
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  ④ VISUALISATIONS — TABS
# ══════════════════════════════════════════════════════════════
sec("🔍", "Analyse Détaillée")
t1, t2, t3 = st.tabs(["🕸️ Profil Global", "♨️ Adéquation Terrain", "📦 Récolte vs Revenu"])

with t1:
    dims = ["Récolte", "Prix", "Revenu", "Terrain", "Valeur totale"]
    maxs = [10.0, 500.0, 3500.0, 10.0, 5000.0]
    fig_r = go.Figure()
    for crop, data in results.items():
        m   = CROP_META[crop]
        raw = [data["yield"], data["price"], data["profit"], data["adeq"], data["yield"] * data["price"]]
        nrm = [min(10, v / mx * 10) for v, mx in zip(raw, maxs)]
        cats = dims + [dims[0]]; vals = nrm + [nrm[0]]
        r, g, b = m["rgb"]
        fig_r.add_trace(go.Scatterpolar(
            r=vals, theta=cats, fill="toself",
            name=f"{m['icon']} {m['label']}",
            line=dict(color=m["color"], width=2.5),
            fillcolor=f"rgba({r},{g},{b},0.13)",
            marker=dict(size=5),
        ))
    fig_r.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10],
                            tickfont=dict(size=8, color="#A0B4A0"), gridcolor="#D4E0CC"),
            angularaxis=dict(tickfont=dict(size=10, color="#1A2E1B"), gridcolor="#D4E0CC"),
            bgcolor="rgba(248,250,245,0.8)",
        ),
        legend=dict(orientation="h", y=-0.18, font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        paper_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(l=28, r=28, t=28, b=70),
        title=dict(text="Performance globale de chaque culture sur votre parcelle",
                   font=dict(size=12, color="#0F1E10", family="Playfair Display"), x=0.5),
    )
    st.plotly_chart(fig_r, use_container_width=True)

with t2:
    param_keys = ["pH", "Hum", "Tmp", "Pluie", "Eng", "Pest"]
    ph_lbl     = ["pH Sol", "Humidité", "Température", "Pluie", "Engrais", "Pesticides"]
    rngs = {"pH":(4,9), "Hum":(5,60), "Tmp":(5,45), "Pluie":(0,300), "Eng":(50,220), "Pest":(0,80)}
    uv   = {"pH": soil_ph, "Hum": soil_moisture, "Tmp": temperature,
            "Pluie": rainfall, "Eng": fertilizer, "Pest": pesticide}
    mat = []
    for crop in ALL_CROPS:
        row_h = []
        for k in param_keys:
            dist = abs(uv[k] - OPTIMA[crop][k]) / (rngs[k][1] - rngs[k][0])
            row_h.append(max(0, round((1 - dist * 2) * 10, 1)))
        mat.append(row_h)
    z    = np.array(mat)
    ylbl = [f"{CROP_META[c]['icon']} {CROP_META[c]['label']}" for c in ALL_CROPS]
    fig_hm = go.Figure(go.Heatmap(
        z=z, x=ph_lbl, y=ylbl,
        text=[[f"<b>{v:.0f}</b>" for v in r] for r in z],
        texttemplate="%{text}",
        textfont=dict(size=14, family="Playfair Display"),
        colorscale=[[0,"#D9534F"],[0.35,"#E8A020"],[0.6,"#F5E090"],[1,"#2E8B57"]],
        zmin=0, zmax=10,
        hovertemplate="<b>%{y}</b> — %{x} : %{z:.1f}/10<extra></extra>",
        colorbar=dict(thickness=12, outlinewidth=0,
                      title=dict(text="/10", side="right", font=dict(size=9)),
                      tickfont=dict(size=9)),
    ))
    fig_hm.update_layout(
        title=dict(text="Adéquation de vos conditions avec les besoins de chaque culture  (10 = parfait)",
                   font=dict(size=12, color="#0F1E10", family="Playfair Display"), x=0.5),
        height=290, **PT,
        xaxis=dict(tickfont=dict(size=11, color="#1A2E1B")),
        yaxis=dict(tickfont=dict(size=11, color="#1A2E1B")),
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    l1, l2, l3 = st.columns(3)
    for col, cls, ico, lbl, msg in [
        (l1, "diag-ok",   "🟢", "Score ≥ 7", "Votre terrain est idéal pour cette culture"),
        (l2, "diag-warn", "🟡", "Score 4–7", "Conditions correctes, améliorations possibles"),
        (l3, "diag-bad",  "🔴", "Score < 4", "Conditions difficiles — à éviter ou corriger"),
    ]:
        with col:
            st.markdown(f"""
            <div class="diag-card {cls}">
              <div class="diag-ico">{ico}</div>
              <div><div class="diag-lbl">{lbl}</div><div class="diag-msg">{msg}</div></div>
            </div>""", unsafe_allow_html=True)

with t3:
    fig_b = go.Figure()
    for crop, data in results.items():
        m = CROP_META[crop]; r, g, b = m["rgb"]
        fig_b.add_trace(go.Scatter(
            x=[data["yield"]], y=[data["profit"]],
            mode="markers+text",
            marker=dict(size=max(24, data["price"] / 16),
                        color=f"rgba({r},{g},{b},0.88)",
                        line=dict(width=2.5, color="white")),
            text=[f"{m['icon']} {m['label']}"],
            textposition="top center",
            textfont=dict(size=10, color="#1A2E1B"),
            name=m["label"],
            hovertemplate=(f"<b>{m['icon']} {m['label']}</b><br>"
                           f"Récolte : {data['yield']:.2f} t/ha<br>"
                           f"Prix : {data['price']:.0f} €/t<br>"
                           f"Revenu : {data['profit']:.0f} €/ha<extra></extra>"),
        ))
    fig_b.update_layout(
        title=dict(text="Plus la bulle est en haut à droite, plus la culture est intéressante pour vous",
                   font=dict(size=12, color="#0F1E10", family="Playfair Display"), x=0.5),
        xaxis=dict(title="Quantité récoltée (t/ha)", showgrid=True, gridcolor="#E4EDE0",
                   zeroline=False, tickfont=dict(size=9, color="#A0B4A0")),
        yaxis=dict(title="Revenu estimé (€/ha)", showgrid=True, gridcolor="#E4EDE0",
                   zeroline=False, tickfont=dict(size=9, color="#A0B4A0")),
        showlegend=False, height=360, **PT,
    )
    st.plotly_chart(fig_b, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  ⑤ ÉTAT DU TERRAIN
# ══════════════════════════════════════════════════════════════
sec("🌍", "État de Votre Terrain")

s1, s2 = st.columns([3, 2], gap="medium")

with s1:
    conds = {
        "pH Sol":        (soil_ph,       4.0,  9.0,  6.0, 7.0),
        "Humidité (%)":  (soil_moisture, 5.0, 60.0, 25.0,45.0),
        "Température":   (temperature,   5.0, 45.0, 18.0,28.0),
        "Pluie (mm)":    (rainfall,      0.0,300.0, 50.0,200.0),
        "Engrais (kg)":  (fertilizer,   50.0,220.0, 80.0,170.0),
        "Pesticides":    (pesticide,     0.0, 80.0,  5.0, 30.0),
    }
    pn  = list(conds.keys())
    vc  = [v[0] for v in conds.values()]
    mnc = [v[1] for v in conds.values()]
    mxc = [v[2] for v in conds.values()]
    imn = [v[3] for v in conds.values()]
    imx = [v[4] for v in conds.values()]
    nv  = [(v-mn)/(mx-mn)*100 for v, mn, mx in zip(vc, mnc, mxc)]
    ni  = [(v-mn)/(mx-mn)*100 for v, mn, mx in zip(imn, mnc, mxc)]
    nx  = [(v-mn)/(mx-mn)*100 for v, mn, mx in zip(imx, mnc, mxc)]
    bcs = ["#1A7A40" if ni[i] <= nv[i] <= nx[i] else
           ("#C4880A" if abs(nv[i]-50) < 30 else "#B03428") for i in range(len(pn))]
    fig_s = go.Figure()
    fig_s.add_trace(go.Bar(
        name="Zone idéale", x=pn,
        y=[x-n for x, n in zip(nx, ni)], base=ni,
        marker_color="rgba(46,107,48,0.12)",
        marker_line_color="rgba(46,107,48,0.35)",
        marker_line_width=1.5, hoverinfo="skip",
    ))
    fig_s.add_trace(go.Bar(
        name="Votre terrain", x=pn,
        y=[3]*len(pn), base=[v-1.5 for v in nv],
        marker_color=bcs, marker_line_width=0,
        text=[f"{v:.1f}" for v in vc],
        textposition="outside", textfont=dict(size=9, color="#1A2E1B"),
        hovertemplate="<b>%{x}</b> : %{text}<extra></extra>",
    ))
    fig_s.update_layout(
        title=dict(text="Position de vos conditions par rapport à la zone idéale",
                   font=dict(size=12, color="#0F1E10", family="Playfair Display")),
        barmode="overlay",
        yaxis=dict(title="Position (%)", ticksuffix="%", showgrid=True,
                   gridcolor="#E4EDE0", tickfont=dict(size=9, color="#A0B4A0")),
        legend=dict(orientation="h", y=1.12, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        height=290, **PT,
    )
    st.plotly_chart(fig_s, use_container_width=True)

with s2:
    ds = {
        "pH Sol":      min(10, max(0, 10 - abs(soil_ph - 6.5) * 2.5)),
        "Humidité":    min(10, soil_moisture / 6),
        "Température": min(10, max(0, 10 - abs(temperature - 22) * 0.5)),
        "Pluie":       min(10, rainfall / 30),
        "Engrais":     min(10, (fertilizer - 50) / 17),
        "Pesticides":  min(10, max(0, 10 - pesticide / 10)),
    }
    ck = list(ds.keys()) + [list(ds.keys())[0]]
    cv = list(ds.values()) + [list(ds.values())[0]]
    fig_rp = go.Figure()
    fig_rp.add_trace(go.Scatterpolar(
        r=[10]*len(ck), theta=ck, fill="toself", name="Idéal",
        line=dict(color="#C4D8BC", width=1.5, dash="dot"),
        fillcolor="rgba(196,216,188,0.08)",
    ))
    fig_rp.add_trace(go.Scatterpolar(
        r=cv, theta=ck, fill="toself", name="Ma parcelle",
        line=dict(color="#2E6B30", width=2.5),
        fillcolor="rgba(46,107,48,0.18)",
        marker=dict(size=5, color="#2E6B30"),
    ))
    fig_rp.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10],
                            tickfont=dict(size=7, color="#A0B4A0"), gridcolor="#D4E0CC"),
            angularaxis=dict(tickfont=dict(size=9, color="#1A2E1B"), gridcolor="#D4E0CC"),
            bgcolor="rgba(248,250,245,0.8)",
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.2, font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        height=290, margin=dict(l=16, r=16, t=28, b=55),
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Profil de votre terrain vs l'idéal",
                   font=dict(size=11, color="#0F1E10", family="Playfair Display"), x=0.5),
    )
    st.plotly_chart(fig_rp, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  ⑥ AMÉLIORER LA RÉCOLTE
# ══════════════════════════════════════════════════════════════
sec("📈", f"Comment Augmenter Votre Récolte — {bm['icon']} {bm['label']}")
st.caption("Simulez l'impact de chaque action sur votre récolte prévue, toutes autres conditions inchangées.")

best_feats = feats_yield[best_crop]

sens = [
    ("Fertilizer_Usage_kg", "Engrais (kg/ha)",      np.linspace(50, 220, 60), bm["color"]),
    ("Rainfall_mm",         "Pluie mensuelle (mm)",  np.linspace(0, 300, 60),  "#1A6B9A"),
    ("Temperature_C",       "Température (°C)",      np.linspace(5, 45, 60),   "#B03428"),
    ("Pesticide_Usage_kg",  "Pesticides (kg/ha)",    np.linspace(0, 80, 60),   "#8A9E8B"),
    ("Soil_pH",             "Acidité du sol (pH)",   np.linspace(4, 9, 60),    "#4A7C20"),
    ("Soil_Moisture",       "Humidité du sol (%)",   np.linspace(5, 60, 60),   "#1A7A5A"),
]
avail = [(c, l, v, cl) for c, l, v, cl in sens if c in best_feats]

if avail:
    nc = 3; nr = (len(avail) + nc - 1) // nc
    fig_sv = make_subplots(rows=nr, cols=nc,
        subplot_titles=[l for _, l, _, _ in avail],
        vertical_spacing=0.14, horizontal_spacing=0.07)

    base_row = best["row_y"]   # référence = conditions actuelles

    for idx, (col_n, lbl, vals, clr) in enumerate(avail):
        prd = []
        for v in vals:
            Xt = base_row.copy()
            Xt[col_n] = v
            prd.append(max(0.0, float(models_yield[best_crop].predict(Xt)[0])))
        ro, co = divmod(idx, nc)
        r2s, g2s, b2s = int(clr[1:3],16), int(clr[3:5],16), int(clr[5:7],16)
        fig_sv.add_trace(go.Scatter(
            x=vals, y=prd, mode="lines",
            line=dict(color=clr, width=2),
            fill="tozeroy", fillcolor=f"rgba({r2s},{g2s},{b2s},0.07)",
            showlegend=False,
        ), row=ro+1, col=co+1)
        cv  = float(base_row[col_n].values[0])
        cp  = float(models_yield[best_crop].predict(base_row)[0])
        fig_sv.add_trace(go.Scatter(
            x=[cv], y=[max(0, cp)], mode="markers",
            marker=dict(size=10, color="#B03428", symbol="star",
                        line=dict(width=1.5, color="white")),
            name="Situation actuelle", showlegend=(idx == 0),
        ), row=ro+1, col=co+1)
        fig_sv.add_hline(y=best["yield"], line_dash="dot",
                         line_color="#A0B4A0", line_width=1, row=ro+1, col=co+1)
    fig_sv.update_layout(
        height=nr * 230,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,245,0.6)",
        font=dict(family="DM Sans", size=10, color="#1A2E1B"),
        margin=dict(l=10, r=10, t=52, b=16),
        legend=dict(orientation="h", y=-0.04, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
    )
    fig_sv.update_yaxes(title_text="t/ha", showgrid=True, gridcolor="#E4EDE0",
                        zeroline=False, tickfont=dict(size=8, color="#A0B4A0"))
    fig_sv.update_xaxes(tickfont=dict(size=8, color="#A0B4A0"))
    fig_sv.update_annotations(font=dict(size=10, color="#0F1E10"))
    st.caption("⭐ Votre situation actuelle · ·· Récolte de référence")
    st.plotly_chart(fig_sv, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  ⑦ CONSEILS PRATIQUES
# ══════════════════════════════════════════════════════════════
sec("📋", f"Conseils Pratiques — {bm['icon']} {bm['label']}")

tips = CROP_TIPS[best_crop]
col_t1, col_t2 = st.columns(2, gap="medium")
for i, tip in enumerate(tips):
    col = col_t1 if i < 3 else col_t2
    with col:
        st.markdown(f"""
        <div class="tip-card">
          <div class="tip-num">{i+1}</div>
          <div class="tip-text">{tip}</div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  ⑧ QUE FAIRE MAINTENANT
# ══════════════════════════════════════════════════════════════
sec("🧭", "Que Faire Maintenant ?")

ycls = ("ac-bad" if best["yield"] < 3 else "ac-warn" if best["yield"] < 6 else "ac-ok")

PLANS = {
    "ac-bad": (
        f"🔴 Attention — Conditions difficiles pour {bm['icon']} {bm['label']} cette saison",
        ["💧 Vérifier et améliorer votre système d'irrigation",
         "🧪 Corriger l'acidité de votre sol (chaux ou soufre selon le pH)",
         "🐛 Renforcer la protection contre les maladies et insectes",
         "📞 Contacter un conseiller agricole de votre région",
         "🔄 Envisager une culture mieux adaptée à votre terrain"],
    ),
    "ac-warn": (
        f"🟡 Conditions correctes — Quelques améliorations pour {bm['icon']} {bm['label']}",
        ["⚖️ Augmenter légèrement vos apports en engrais (+10 à 15%)",
         "📅 Planifier vos irrigations selon la météo des 15 prochains jours",
         "🌡️ Surveiller les pics de chaleur pendant la floraison",
         "🏷️ Comparer les prix d'achat des coopératives de votre région",
         "🌱 Travailler le sol en profondeur avant le prochain semis"],
    ),
    "ac-ok": (
        f"🟢 Excellentes conditions — Profitez-en ! {bm['icon']} {bm['label']}",
        ["📦 Négocier un contrat de vente à terme pour sécuriser votre prix",
         "💰 Évaluer l'agrandissement de votre surface cultivée",
         "🗂️ Notez ce qui fonctionne pour reproduire les résultats l'an prochain",
         "🌱 Commandez vos semences certifiées tôt pour la saison suivante",
         "📈 Souscrire une assurance récolte pour protéger votre revenu"],
    ),
}
pt, pa = PLANS[ycls]
st.markdown(f"""
<div class="action-card {ycls}">
  <div class="ac-title">{pt}</div>
  <div>{''.join(f'<span class="action-tag">{a}</span>' for a in pa)}</div>
</div>""", unsafe_allow_html=True)

# Diagnostics rapides
d1c, d2c, d3c = st.columns(3, gap="medium")
ph_ok   = 6.0 <= soil_ph <= 7.0
rain_ok = 50 <= rainfall <= 200
temp_ok = 15 <= temperature <= 30
dgs = [
    (d1c, "🪨", "Qualité du Sol",
     ("diag-ok",   f"pH {soil_ph:.1f} — Sol en bonne santé") if ph_ok else
     ("diag-warn", f"pH {soil_ph:.1f} — Sol à amender avant le semis")),
    (d2c, "🌧️", "Alimentation en Eau",
     ("diag-ok",   f"{rainfall:.0f} mm — Eau suffisante pour la culture")  if rain_ok else
     ("diag-bad",  f"{rainfall:.0f} mm — Irrigation nécessaire urgemment") if rainfall < 50 else
     ("diag-warn", f"{rainfall:.0f} mm — Excès d'eau, prévoir drainage")),
    (d3c, "🌡️", "Conditions Climatiques",
     ("diag-ok",   f"{temperature:.0f}°C — Températures favorables")       if temp_ok else
     ("diag-warn", f"{temperature:.0f}°C — Froid : préférer des variétés d'hiver") if temperature < 15 else
     ("diag-bad",  f"{temperature:.0f}°C — Chaleur excessive, risque de stress")),
]
for col, ico, lbl, (cls, msg) in dgs:
    with col:
        st.markdown(f"""
        <div class="diag-card {cls}">
          <div class="diag-ico">{ico}</div>
          <div><div class="diag-lbl">{lbl}</div><div class="diag-msg">{msg}</div></div>
        </div>""", unsafe_allow_html=True)

# Culture alternative
if len(ranked) > 1:
    alt_c, alt_d = ranked[1]
    alt_m = CROP_META[alt_c]
    diff  = ((best["profit"] - alt_d["profit"]) / max(1, alt_d["profit"])) * 100
    st.markdown(f"""
    <div class="action-card ac-info" style="margin-top:12px">
      <div class="ac-title">
        💡 Autre option intéressante : {alt_m['icon']} {alt_m['label']}
        &nbsp;—&nbsp; {alt_d['profit']:.0f} €/ha
        &nbsp;·&nbsp; Soit {diff:.1f}% de moins que {bm['label']}
        &nbsp;·&nbsp; Adéquation terrain : {alt_d['adeq']}/10
      </div>
      <div style="font-size:.8rem;color:#1A2E1B;line-height:1.6">
        À envisager pour répartir les risques ou si les conditions de votre terrain
        changent en cours de saison.
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='background:#fff;border-radius:14px;padding:18px 28px;
            display:flex;justify-content:space-between;align-items:center;
            flex-wrap:wrap;gap:12px;
            border:1px solid rgba(0,0,0,0.06);
            box-shadow:0 1px 10px rgba(15,30,16,0.05);
            border-top:3px solid #D4E8CC'>
  <div style='font-size:.73rem;color:#7A8E7B;line-height:1.7;max-width:680px'>
    ⚠️ Les estimations de récolte et de prix sont indicatives et basées sur des données
    historiques agricoles. Les résultats réels dépendent de nombreux facteurs locaux.
    <strong>Consultez toujours un conseiller agricole avant toute décision importante.</strong>
  </div>
  <div style='text-align:right'>
    <div style='font-family:Playfair Display,serif;font-size:1.05rem;
                font-weight:700;color:#2E6B30;letter-spacing:-0.2px'>
      🌿 AgriSmart
    </div>
    <div style='font-size:0.62rem;color:#A8BCA8;margin-top:3px;font-weight:500'>
      Conseil Agricole Intelligent · PFE
    </div>
  </div>
</div>
""", unsafe_allow_html=True)