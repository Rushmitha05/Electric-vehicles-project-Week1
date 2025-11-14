# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib, os, json
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from math import sqrt

st.set_page_config("⚡ EV IntelliSense", layout="wide", page_icon="⚡")

# -------------------------
# Styling
# -------------------------
st.markdown(
    """
    <style>
      .stApp { font-family: "Segoe UI", Roboto, Arial; background: #f6f7f9; }
      header {display:none}
      .topbar { background: white; padding: 18px; border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,0.06); }
      .brand { display:flex; align-items:center; gap:12px; }
      .brand h1{ margin:0; font-size:22px; }
      .sub{ color: #6b7280; margin-top:2px; font-size:13px;}
      .card { background:white; padding:18px; border-radius:12px; box-shadow: 0 1px 8px rgba(0,0,0,0.04); }
      .metric { padding: 14px; border-radius:10px; background:white; box-shadow: 0 1px 6px rgba(0,0,0,0.04); }
      .navtabs { display:flex; gap:8px; margin-top:14px; }
      .navtabs button { padding:10px 22px; border-radius:12px; border: none; background: #e9e9e9; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Utilities & Demo data
# -------------------------
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def to_numeric(v):
    try:
        if pd.isna(v): return np.nan
        s = str(v).replace(',', '').replace('₹','').replace('$','')
        import re
        s = re.sub(r'[^\d\.\-]', '', s)
        if s in ['', '.', '-']: return np.nan
        return float(s)
    except:
        return np.nan

def clean_colnames(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(' ', '_').replace('%','pct').replace('/','_').replace('-','_').replace('.','') for c in df.columns]
    return df

def demo_ev():
    return pd.DataFrame([
        {"make":"Tesla","model":"Model 3","model_year":2022,"battery_kwh":75,"electric_range_km":510,"efficiency":150,"base_msrp_numeric":3500000},
        {"make":"Hyundai","model":"Kona EV","model_year":2023,"battery_kwh":64,"electric_range_km":450,"efficiency":142,"base_msrp_numeric":1695000},
        {"make":"Tata","model":"Nexon EV","model_year":2022,"battery_kwh":40,"electric_range_km":312,"efficiency":128,"base_msrp_numeric":1400000},
        {"make":"Nissan","model":"Leaf","model_year":2019,"battery_kwh":40,"electric_range_km":240,"efficiency":160,"base_msrp_numeric":1200000},
        {"make":"Chevrolet","model":"Bolt","model_year":2020,"battery_kwh":66,"electric_range_km":416,"efficiency":158,"base_msrp_numeric":2000000},
    ])

def demo_stations():
    return pd.DataFrame([
        {"station_id":"S1","address":"Downtown Fast Hub","latitude":12.9716,"longitude":77.5946,"charger_type":"DC Fast","charging_capacity_kw":150,"cost_usd_per_kwh":0.25,"station_operator":"EVgo"},
        {"station_id":"S2","address":"Mall Level 2","latitude":12.97,"longitude":77.59,"charger_type":"AC Level 2","charging_capacity_kw":22,"cost_usd_per_kwh":0.12,"station_operator":"ChargePoint"},
        {"station_id":"S3","address":"Highway Supercharger","latitude":12.98,"longitude":77.60,"charger_type":"DC Fast","charging_capacity_kw":250,"cost_usd_per_kwh":0.30,"station_operator":"SuperNet"},
        {"station_id":"S4","address":"Market Street","latitude":12.96,"longitude":77.58,"charger_type":"AC Level 2","charging_capacity_kw":11,"cost_usd_per_kwh":0.15,"station_operator":"GreenLots"},
    ])

# -------------------------
# Header UI
# -------------------------
with st.container():
    st.markdown('<div class="topbar"><div class="brand"><div style="width:58px;height:58px;border-radius:12px;background:#0b5ed7;color:white;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:28px">⚡</div>'
                '<div><h1>EV IntelliSense</h1><div class="sub">AI-Powered EV Analytics & Insights</div></div></div></div>', unsafe_allow_html=True)

# -------------------------
# Sidebar: Upload + options
# -------------------------
st.sidebar.header("Data & Settings")
ev_file = st.sidebar.file_uploader("Upload Electric_Vehicle_Population_Data.csv", type=["csv"])
st_file = st.sidebar.file_uploader("Upload detailed_ev_charging_stations.csv", type=["csv"])
use_demo = st.sidebar.checkbox("Use demo data", True)
enable_openai = st.sidebar.checkbox("Enable OpenAI Chat (optional)", False)
openai_key = None
if enable_openai:
    openai_key = st.sidebar.text_input("OpenAI Key (sk-...)", type="password")

train_models_now = st.sidebar.button("Train models (optional)")

# load data
if ev_file:
    ev_raw = pd.read_csv(ev_file, low_memory=False)
else:
    ev_raw = demo_ev() if use_demo else pd.DataFrame()

if st_file:
    st_raw = pd.read_csv(st_file, low_memory=False)
else:
    st_raw = demo_stations() if use_demo else pd.DataFrame()

# normalize
if not ev_raw.empty: ev_raw = clean_colnames(ev_raw)
if not st_raw.empty: st_raw = clean_colnames(st_raw)

# -------------------------
# Tabs / Navigation
# -------------------------
tabs = st.tabs(["Dashboard", "Predict", "Chargers", "Chat"])
# -------------------------
# Cleaning helper functions
# -------------------------
def clean_ev(df):
    e = df.copy()
    # map common columns
    if 'electric_range' in e.columns and 'electric_range_km' not in e.columns:
        e['electric_range_km'] = e['electric_range'].apply(to_numeric)
    if 'base_msrp' in e.columns and 'base_msrp_numeric' not in e.columns:
        e['base_msrp_numeric'] = e['base_msrp'].apply(to_numeric)
    if 'model_year' in e.columns:
        e['model_year'] = pd.to_numeric(e['model_year'], errors='coerce').astype('Int64')
        e['vehicle_age'] = 2025 - e['model_year']
    if 'electric_range_km' in e.columns:
        e['electric_range_km'] = e['electric_range_km'].apply(to_numeric)
    if 'efficiency' in e.columns:
        e['efficiency'] = e['efficiency'].apply(to_numeric)
    # drop exact duplicates
    dedup_cols = [c for c in ['make','model','model_year'] if c in e.columns]
    if dedup_cols:
        e = e.drop_duplicates(subset=dedup_cols, keep='first')
    # impute missing range with make median
    if 'electric_range_km' in e.columns and 'make' in e.columns:
        med = e.groupby('make')['electric_range_km'].median()
        def imp(r):
            if pd.isna(r['electric_range_km']):
                m = r.get('make')
                if pd.notna(m) and m in med.index:
                    return med.loc[m]
                return e['electric_range_km'].median(skipna=True)
            return r['electric_range_km']
        e['electric_range_km'] = e.apply(imp, axis=1)
    return e

def clean_stations(df):
    s = df.copy()
    # unify cost
    poss = [c for c in s.columns if 'cost' in c and 'kwh' in c]
    if poss:
        s['cost_usd_per_kwh'] = s[poss[0]].apply(to_numeric)
    # charger type
    for c in s.columns:
        if 'charger' in c or 'connector' in c or 'plug' in c:
            s = s.rename(columns={c:'charger_type'}); break
    # capacity
    cap = next((c for c in s.columns if 'kw' in c or 'capacity' in c), None)
    if cap:
        s['charging_capacity_kw'] = s[cap].apply(to_numeric)
    # lat lon
    for lat in ['latitude','lat','station_latitude']:
        for lon in ['longitude','lon','lng','station_longitude']:
            if lat in s.columns and lon in s.columns:
                s['latitude'] = pd.to_numeric(s[lat], errors='coerce')
                s['longitude'] = pd.to_numeric(s[lon], errors='coerce')
                break
    # operator
    for c in s.columns:
        if 'operator' in c or 'owner' in c:
            s = s.rename(columns={c:'station_operator'}); break
    # fill medians
    if 'cost_usd_per_kwh' in s.columns:
        s['cost_usd_per_kwh'] = s['cost_usd_per_kwh'].fillna(s['cost_usd_per_kwh'].median(skipna=True))
    if 'charging_capacity_kw' in s.columns:
        s['charging_capacity_kw'] = s['charging_capacity_kw'].fillna(s['charging_capacity_kw'].median(skipna=True))
    return s

# run cleaning once
if 'ev_clean' not in st.session_state:
    st.session_state['ev_clean'] = clean_ev(ev_raw) if not ev_raw.empty else pd.DataFrame()
if 'st_clean' not in st.session_state:
    st.session_state['st_clean'] = clean_stations(st_raw) if not st_raw.empty else pd.DataFrame()

# -------------------------
# Dashboard Tab
# -------------------------
with tabs[0]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Dashboard")
    st.write("Overview of loaded data and key metrics")
    evc = st.session_state['ev_clean']
    stc = st.session_state['st_clean']
    # Top metric cards
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
        total = int(evc.shape[0]) if not evc.empty else 0
        st.markdown(f"<div class='metric'><h3 style='margin:0'>{total:,}</h3><div style='color:#6b7280'>Total Vehicles</div></div>", unsafe_allow_html=True)
    with col2:
        avg_soh =  np.nan
        # if battery_soh or battery_soh_proxy exists
        if 'battery_soh' in evc.columns and evc['battery_soh'].notna().any():
            avg_soh = evc['battery_soh'].mean()
        elif 'battery_soh_proxy' in evc.columns and evc['battery_soh_proxy'].notna().any():
            avg_soh = evc['battery_soh_proxy'].mean()
        else:
            avg_soh =  np.nan
        st.metric("Avg Battery SoH", f"{np.round(avg_soh,2) if not np.isnan(avg_soh) else 'N/A'}%")
    with col3:
        eco_avg = evc.eval("electric_range_km/efficiency") if ('electric_range_km' in evc.columns and 'efficiency' in evc.columns) else None
        if eco_avg is not None:
            st.metric("Eco Score (proxy)", f"{np.round((eco_avg.fillna(0).mean()*10),2)} / 100")
        else:
            st.metric("Eco Score (proxy)", "N/A")
    with col4:
        active_ch = int(stc.shape[0]) if not stc.empty else 0
        st.markdown(f"<div class='metric'><h3 style='margin:0'>{active_ch}</h3><div style='color:#6b7280'>Active Chargers</div></div>", unsafe_allow_html=True)

    st.markdown("---")
    # Charts row
    a,b = st.columns([2,1])
    with a:
        st.subheader("EV Models Distribution")
        if not evc.empty and 'make' in evc.columns:
            fig = px.histogram(evc, x='make', title="Vehicle count by manufacturer")
            fig.update_layout(margin=dict(t=40,l=10,r=10,b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Upload EV dataset to view charts")
    with b:
        st.subheader("Battery SoH trend (sample)")
        # synthetic time-series for demo if no soh
        if 'battery_soh' in evc.columns and evc['battery_soh'].notna().any():
            tmp = evc[['battery_soh']].copy()
            tmp['month'] = pd.date_range(end=pd.Timestamp.today(), periods=len(tmp)).strftime('%b')
            fig2 = px.line(tmp, x='month', y='battery_soh', title="Battery SoH")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Battery SoH unavailable — proxy available after training")

    st.markdown("---")
    st.subheader("EcoScore Distribution")
    if not evc.empty and 'electric_range_km' in evc.columns and 'efficiency' in evc.columns:
        evc['eco_raw'] = (evc['electric_range_km'] / evc['efficiency']) * 10.0
        evc['eco_cat'] = pd.cut(evc['eco_raw'], bins=[-1,30,50,70,999], labels=['Poor','Average','Good','Excellent'])
        fig3 = px.pie(evc, names='eco_cat', title="EcoScore Distribution", hole=0.35)
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Upload EV data with range & efficiency to compute EcoScore")

# -------------------------
# Predict Tab
# -------------------------
with tabs[1]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Vehicle Prediction")
    evc = st.session_state['ev_clean']
    # form
    with st.form("predict_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            make = st.selectbox("Make", options=sorted(evc['make'].dropna().unique()) if ('make' in evc.columns and not evc.empty) else ['Tesla','Hyundai','Tata'])
            model = st.text_input("Model")
            year = st.number_input("Model Year", min_value=1990, max_value=2025, value=2022)
        with col2:
            battery = st.number_input("Battery (kWh)", min_value=10.0, max_value=200.0, value=float(evc['battery_kwh'].median()) if 'battery_kwh' in evc.columns and evc['battery_kwh'].notna().any() else 64.0)
            rng = st.number_input("Range (km)", min_value=50.0, max_value=1000.0, value=float(evc['electric_range_km'].median()) if 'electric_range_km' in evc.columns and evc['electric_range_km'].notna().any() else 400.0)
            eff = st.number_input("Efficiency (Wh/km)", min_value=50.0, max_value=400.0, value=float(evc['efficiency'].median()) if 'efficiency' in evc.columns and evc['efficiency'].notna().any() else 150.0)
        with col3:
            fast_ch = st.number_input("Fast charges/week", min_value=0.0, max_value=50.0, value=0.5)
            miles_day = st.number_input("Daily distance (km)", min_value=1, max_value=500, value=40)
            seats = st.number_input("Seats", min_value=1, max_value=9, value=5)
        submitted = st.form_submit_button("Generate Predictions")
    if submitted:
        # Heuristic Battery SoH estimator (fallback)
        age = 2025 - year
        soh = max(20.0, 100.0 - 0.9*age - 0.5*fast_ch - max(0, (30 - 25)*0.05))
        # Eco score
        eco = round(((rng / eff) * 10.0) - age*1.5, 2)
        eco = max(0, min(100, eco))
        # maintenance cost heuristic
        maint = 2000 + age*500 + (100 - soh)*50
        # price heuristic (simple linear)
        price = 50000 + battery*15000 + rng*200 + (seats-4)*10000
        price = int(price)
        colA, colB, colC, colD = st.columns(4)
        colA.metric("Battery SoH (est)", f"{round(soh,2)}%")
        colB.metric("EcoScore", f"{eco}/100")
        colC.metric("Est. Annual Maintenance", f"₹{int(maint)}")
        colD.metric("Predicted Price", f"₹{price:,}")
        st.markdown("#### Range vs Efficiency")
        df_scatter = pd.DataFrame({"model":[model],"range_km":[rng],"eff_wh_km":[eff]})
        fig = px.scatter(df_scatter, x='range_km', y='eff_wh_km', text='model', size=[12])
        st.plotly_chart(fig, use_container_width=True)

    # Optional training of simple models from client data
    st.markdown("---")
    st.subheader("Train simple models from uploaded EV dataset (optional)")
    st.caption("Creates model artifacts under ./models (battery_soh_proxy, eco_score_norm, maintenance_cost_proxy, price if available).")
    if train_models_now:
        evtrain = st.session_state['ev_clean']
        if evtrain.empty:
            st.error("No EV data to train. Upload data or enable demo.")
        else:
            with st.spinner("Training models..."):
                # create proxies if not present
                df = evtrain.copy()
                if 'vehicle_age' not in df.columns and 'model_year' in df.columns:
                    df['vehicle_age'] = 2025 - df['model_year'].astype(float)
                if 'battery_soh' not in df.columns:
                    df['fast_charges_per_week'] = df.get('fast_charges_per_week', 0)
                    df['battery_soh_proxy'] = df.apply(lambda r: max(20.0, 100.0 - 0.9*(r.get('vehicle_age') or 3) - 0.5*(r.get('fast_charges_per_week') or 0)), axis=1)
                # features & categorical
                NUM = [c for c in ['battery_kwh','electric_range_km','efficiency','vehicle_age','fast_charges_per_week'] if c in df.columns]
                CAT = [c for c in ['make','vehicle_type','fuel_type'] if c in df.columns]
                def encode_cols(X, cols):
                    enc = {}
                    Xc = pd.DataFrame(index=X.index)
                    for c in cols:
                        le = LabelEncoder(); Xc[c] = le.fit_transform(X[c].astype(str).fillna('nan')); enc[c]=le
                    return Xc, enc
                def train_target(target):
                    if target not in df.columns: return None
                    sub = df.dropna(subset=[target]).copy()
                    if sub.shape[0] < 8:
                        return None
                    Xn = sub[NUM].applymap(to_numeric) if NUM else pd.DataFrame(index=sub.index)
                    Xn = Xn.fillna(Xn.median()) if not Xn.empty else Xn
                    Xc, enc = encode_cols(sub, CAT) if CAT else (pd.DataFrame(index=sub.index), {})
                    X = pd.concat([Xn, Xc], axis=1) if not Xn.empty else Xc.copy()
                    y = sub[target].astype(float)
                    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
                    scaler = StandardScaler(); Xtr_s = scaler.fit_transform(Xtr); Xte_s = scaler.transform(Xte)
                    rf = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1); rf.fit(Xtr_s, ytr)
                    preds = rf.predict(Xte_s)
                    mae = mean_absolute_error(yte, preds); r2 = r2_score(yte, preds)
                    joblib.dump(rf, os.path.join(MODELS_DIR, f"rf_{target}.pkl"))
                    joblib.dump(scaler, os.path.join(MODELS_DIR, f"scaler_{target}.pkl"))
                    joblib.dump(enc, os.path.join(MODELS_DIR, f"enc_{target}.pkl"))
                    return {"target":target,"r2":r2,"mae":mae}
                results = {}
                # battery target
                batt_t = 'battery_soh' if 'battery_soh' in df.columns else 'battery_soh_proxy'
                r1 = train_target(batt_t)
                results['battery'] = r1
                # eco score
                if 'electric_range_km' in df.columns and 'efficiency' in df.columns:
                    df['eco_raw'] = (df['electric_range_km']/df['efficiency'])*10.0
                    df['eco_score_norm'] = ((df['eco_raw'] - (df['vehicle_age'].fillna(0)*1.5))).clip(0,100)
                    r2 = train_target('eco_score_norm')
                    results['eco'] = r2
                # maintenance proxy
                df['maintenance_cost_proxy'] = df.apply(lambda r: 2000 + (r.get('vehicle_age') or 3)*500 + (100 - (r.get(batt_t,90) or 90))*50, axis=1)
                results['maint'] = train_target('maintenance_cost_proxy')
                st.success("Training finished")
                st.json(results)

# -------------------------
# Chargers Tab
# -------------------------
with tabs[2]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("Find Charging Stations")
    st.write("AI-powered recommendations based on priority")
    stc = st.session_state['st_clean']
    if stc.empty:
        st.info("No stations data loaded. Upload CSV or enable demo to see stations.")
    else:
        left, right = st.columns([1,2])
        with left:
            lat = st.number_input("Your Latitude", float(stc['latitude'].median() if 'latitude' in stc.columns else 12.97))
            lon = st.number_input("Your Longitude", float(stc['longitude'].median() if 'longitude' in stc.columns else 77.59))
            priority = st.selectbox("Sort by Priority", options=["balanced","lowest_cost","fastest"])
            cur_soc = st.slider("Current SOC (%)", 0,100,20)
            tgt_soc = st.slider("Target SOC (%)", 0,100,80)
            find = st.button("Find Stations")
        def recommend(df, lat, lon, cur, tgt, priority):
            s = df.copy()
            # distance rough haversine approx ~ using degrees->km factor
            if 'latitude' in s.columns and 'longitude' in s.columns:
                s['dist_km'] = np.sqrt((s['latitude']-lat)**2 + (s['longitude']-lon)**2)*111
            else:
                s['dist_km'] = 9999
            s['charging_capacity_kw'] = s.get('charging_capacity_kw', np.nan)
            s['cost_usd_per_kwh'] = s.get('cost_usd_per_kwh', np.nan)
            batt_kwh = float((evc['battery_kwh'].median() if (not evc.empty and 'battery_kwh' in evc.columns) else 60.0))
            kwh_needed = batt_kwh * max(0, (tgt-cur))/100.0
            s['est_time_min'] = (kwh_needed / s['charging_capacity_kw'].replace(0,10)) * 60
            # score combining cost, distance, speed
            s['score'] = 0.0
            if s['cost_usd_per_kwh'].notna().any(): s['score'] += (1/(s['cost_usd_per_kwh']+1e-9))*0.4
            s['score'] += (1/(s['dist_km']+1e-9))*0.3
            if s['charging_capacity_kw'].notna().any(): s['score'] += (s['charging_capacity_kw']/(s['charging_capacity_kw'].max()+1e-9))*0.3
            if priority == 'lowest_cost':
                s = s.sort_values('cost_usd_per_kwh').head(10)
            elif priority == 'fastest':
                s = s.sort_values('charging_capacity_kw', ascending=False).head(10)
            else:
                s = s.sort_values('score', ascending=False).head(10)
            return s
        if 'find' in locals() and find:
            recs = recommend(stc, lat, lon, cur_soc, tgt_soc, priority)
            st.dataframe(recs[['station_id','address','dist_km','charging_capacity_kw','cost_usd_per_kwh','est_time_min']].fillna("N/A"))
        else:
            st.info("Set location & press Find Stations to get recommendations.")

# -------------------------
# Chat Tab
# -------------------------
with tabs[3]:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.header("EV IntelliSense Chat")
    st.write("Ask about range, battery health, charging costs, or maintenance.")
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [("Bot","Hello! I'm EV IntelliSense AI. Ask me anything about EVs, battery health, charging, or maintenance.")]
    for who, text in st.session_state['chat_history'][-12:]:
        if who=="You":
            st.markdown(f"<div style='text-align:right; background:#1f2937; color:white; padding:12px; border-radius:8px; margin-bottom:6px'>{text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:left; background:#e6e6e6; color:black; padding:12px; border-radius:8px; margin-bottom:6px'>{text}</div>", unsafe_allow_html=True)

    q = st.text_input("Ask about battery, charging, maintenance...", key="query_input")
    if st.button("Send", key="send_msg"):
        if q:
            st.session_state['chat_history'].append(("You", q))
            # rule-based
            ql = q.lower()
            reply = None
            if "best range" in ql or "longest range" in ql:
                if not evc.empty and 'electric_range_km' in evc.columns:
                    top = evc.sort_values('electric_range_km', ascending=False).iloc[0]
                    reply = f"Top range: {top.get('make','')} {top.get('model','')} ~{int(top.get('electric_range_km',0))} km."
                else:
                    reply = "Upload EV data to compute best range."
            elif "charge" in ql and "cost" in ql:
                if not stc.empty and 'cost_usd_per_kwh' in stc.columns:
                    reply = f"Median charging cost (loaded stations): ${stc['cost_usd_per_kwh'].median():.2f}/kWh"
                else:
                    reply = "Charging station data not loaded."
            elif "battery health" in ql or "soh" in ql:
                reply = "Provide model year and fast-charge frequency and I can estimate SoH. (e.g., 'Nexon EV 2022, 2 fast charges/week')"
            else:
                # fallback generic
                reply = "Sorry, I don't know that directly. Ask about 'best range', 'charging cost', or 'battery health'."
            st.session_state['chat_history'].append(("Bot", reply))
            # clear text input
            st.session_state['query_input'] = ""

    st.caption("Tip: provide an OpenAI key in the sidebar to enable generative responses (paid feature).")

# -------------------------
# Footer
# -------------------------
st.markdown("<div style='margin-top:24px; color:#6b7280'>© EV IntelliSense — Rushmitha Arelli</div>", unsafe_allow_html=True)
