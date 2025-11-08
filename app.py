# Cell 1: imports
import pandas as pd
import numpy as np
import io
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Configure pandas display
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 180)

print("Libraries loaded. pandas version:", pd.__version__)
# Cell 2: Upload CSVs (Colab)
# If running locally, replace with paths like "/content/Electric_Vehicle_Population_Data.csv"
try:
    from google.colab import files
    print("Please upload these two files when prompted:")
    print(" - Electric_Vehicle_Population_Data.csv")
    print(" - detailed_ev_charging_stations.csv")
    uploaded = files.upload()  # interactively upload files
    # Save to local filenames
    for fname in uploaded.keys():
        with open(fname, 'wb') as f:
            f.write(uploaded[fname])
    ev_pop_path = "Electric_Vehicle_Population_Data.csv"
    stations_path = "detailed_ev_charging_stations.csv"
except Exception as e:
    print("No google.colab.files available (not running in Colab).")
    print("Set file paths manually.")
    ev_pop_path = "/content/Electric_Vehicle_Population_Data.csv"
    stations_path = "/content/detailed_ev_charging_stations.csv"

# Quick existence check
print("EV population path:", ev_pop_path, "exists?", os.path.exists(ev_pop_path))
print("Charging stations path:", stations_path, "exists?", os.path.exists(stations_path))
# Cell 3: helper functions for cleaning
def clean_column_names(df):
    # lowercase, strip, replace spaces & weird chars with underscores
    df = df.copy()
    df.columns = [(
        str(c).strip()
              .lower()
              .replace('%','pct')
              .replace(' ', '_')
              .replace('-', '_')
              .replace('/', '_')
              .replace('.', '')
    ) for c in df.columns]
    return df

def to_numeric_safely(val):
    try:
        if pd.isna(val): return np.nan
        # Remove common non-numeric characters (commas, $ ,₹)
        s = str(val).replace(',','').replace('$','').replace('₹','').strip()
        # also remove text like "kW" or " miles"
        import re
        s = re.sub(r'[^\d\.\-]', '', s)
        if s == '' or s == '-' or s == '.': return np.nan
        return float(s)
    except Exception:
        return np.nan

def standardize_datetime_series(s):
    # Try to parse common date formats, return series of datetimes or NaT
    return pd.to_datetime(s, errors='coerce', infer_datetime_format=True)
# Cell 4: load dataframes
ev = pd.read_csv(ev_pop_path, low_memory=False)
stations = pd.read_csv(stations_path, low_memory=False)

print("Raw EV population shape:", ev.shape)
print("Raw stations shape:", stations.shape)
display(ev.head())
display(stations.head())

# Clean column names
ev = clean_column_names(ev)
stations = clean_column_names(stations)

print("After normalizing column names:")
print("EV columns:", ev.columns.tolist()[:30])
print("Stations columns:", stations.columns.tolist()[:30])
# Cell 5: basic diagnostics
def diagnostics(df, name="df", max_display=5):
    print(f"--- {name} diagnostics ---")
    print("shape:", df.shape)
    print("missing per column (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))
    print("duplicate rows:", df.duplicated().sum())
    print()
    
diagnostics(ev, "EV Population")
diagnostics(stations, "Charging Stations")
# Cell 6: Clean charging stations
s = stations.copy()

# Normalize common fields
if 'cost_(usd/kwh)' in s.columns:
    s['cost_usd_per_kwh'] = s['cost_(usd/kwh)'].apply(to_numeric_safely)
elif 'cost (usd/kwh)' in s.columns:
    s['cost_usd_per_kwh'] = s['cost (usd/kwh)'].apply(to_numeric_safely)
else:
    # try common names
    possible = [c for c in s.columns if 'cost' in c and 'kwh' in c]
    if possible:
        s['cost_usd_per_kwh'] = s[possible[0]].apply(to_numeric_safely)

# Charger type normalization
if 'charger_type' in s.columns:
    s['charger_type'] = s['charger_type'].astype(str).str.strip().str.title()
    # unify common variants
    s['charger_type'] = s['charger_type'].replace({
        'Ac Level 2':'AC Level 2','Dc Fast Charger':'DC Fast Charger','Ac Level 1':'AC Level 1',
        'Level2':'AC Level 2','Level1':'AC Level 1','Fast':'DC Fast Charger'
    })
else:
    # Try to find a column that looks like charger type
    for c in s.columns:
        if 'charger' in c:
            s.rename(columns={c:'charger_type'}, inplace=True)
            s['charger_type'] = s['charger_type'].astype(str).str.strip().str.title()
            break

# Numeric capacity
if 'charging_capacity_(kw)' in s.columns or 'charging_capacity' in s.columns:
    col = 'charging_capacity_(kw)' if 'charging_capacity_(kw)' in s.columns else 'charging_capacity'
    s['charging_capacity_kw'] = s[col].apply(to_numeric_safely)
else:
    # try guessing
    for c in s.columns:
        if 'kw' in c or 'capacity' in c:
            s['charging_capacity_kw'] = s[c].apply(to_numeric_safely)
            break

# Parse latitude & longitude
for latc in ['latitude','lat','station_latitude']:
    for longc in ['longitude','lon','lng','station_longitude']:
        if latc in s.columns and longc in s.columns:
            s['latitude'] = pd.to_numeric(s[latc], errors='coerce')
            s['longitude'] = pd.to_numeric(s[longc], errors='coerce')
            break
    if 'latitude' in s.columns and 'longitude' in s.columns:
        break

# Availability: normalize to simple categories (24/7, daytime, night, scheduled)
if 'availability' in s.columns:
    s['availability_simple'] = s['availability'].astype(str).str.lower()
    s['availability_simple'] = s['availability_simple'].replace({
        '24/7':'24/7','24hours':'24/7','always':'24/7','open 24 hours':'24/7'
    })
    s.loc[s['availability_simple'].str.contains('9:00|9-18|9am', na=False), 'availability_simple'] = 'daytime'
else:
    s['availability_simple'] = np.nan

# Usage stats -> numeric
for c in s.columns:
    if 'usage' in c or 'avg users' in c or 'avg_users' in c:
        s['avg_users_per_day'] = s[c].apply(to_numeric_safely)
        break

# Station operator cleanup
if 'station_operator' in s.columns:
    s['station_operator'] = s['station_operator'].astype(str).str.strip().replace({'nan':np.nan})
else:
    # attempt to find operator-like column
    for c in s.columns:
        if 'operator' in c or 'owner' in c:
            s.rename(columns={c:'station_operator'}, inplace=True)
            break

# Fill some missing numeric values with medians (non-destructive)
if 'cost_usd_per_kwh' in s.columns:
    med_cost = s['cost_usd_per_kwh'].median(skipna=True)
    s['cost_usd_per_kwh'] = s['cost_usd_per_kwh'].fillna(med_cost)

if 'charging_capacity_kw' in s.columns:
    med_cap = s['charging_capacity_kw'].median(skipna=True)
    s['charging_capacity_kw'] = s['charging_capacity_kw'].fillna(med_cap)

# Drop obvious duplicates based on lat/lon + operator + charger_type
dedup_cols = []
for c in ['latitude','longitude','station_operator','charger_type']:
    if c in s.columns:
        dedup_cols.append(c)
if dedup_cols:
    before = s.shape[0]
    s = s.drop_duplicates(subset=dedup_cols, keep='first')
    print(f"Dropped {before - s.shape[0]} duplicate station rows based on {dedup_cols}")

# Reorder & keep useful columns
keep_cols = ['station id','latitude','longitude','address','charger_type','charging_capacity_kw',
             'cost_usd_per_kwh','availability_simple','avg_users_per_day','station_operator']
# Use intersection
keep = [c for c in keep_cols if c in s.columns]
stations_clean = s[keep].copy()
print("stations_clean shape:", stations_clean.shape)
stations_clean.head(5)
# Cell 7: Clean EV population data
e = ev.copy()

# Typical columns we expect: make, model, model_year, electric_range, fuel_type, vehicle_type, base_msrp
# Normalize some common names
colmap = {}
for col in e.columns:
    cl = col.lower()
    if 'make' in cl and 'manufacturer' not in colmap:
        colmap[col] = 'make'
    if 'model_year' in cl or ('year' in cl and 'model' in cl):
        colmap[col] = 'model_year'
    if 'model' == cl or 'model_name' in cl or 'vehicle_model' in cl:
        colmap[col] = 'model'
    if 'electric' in cl and 'range' in cl:
        colmap[col] = 'electric_range'
    if 'fuel' in cl and 'type' in cl:
        colmap[col] = 'fuel_type'
    if 'vehicle_type' in cl or 'body_type' in cl:
        colmap[col] = 'vehicle_type'
    if 'base' in cl and 'msrp' in cl:
        colmap[col] = 'base_msrp'

e = e.rename(columns=colmap)

# Make/model cleanup
if 'make' in e.columns:
    e['make'] = e['make'].astype(str).str.strip().str.title()
if 'model' in e.columns:
    e['model'] = e['model'].astype(str).str.strip()

# model_year numeric
if 'model_year' in e.columns:
    e['model_year'] = pd.to_numeric(e['model_year'], errors='coerce').astype('Int64')

# electric_range numeric
if 'electric_range' in e.columns:
    e['electric_range_km'] = e['electric_range'].apply(to_numeric_safely)
    # If ranges are in miles (detect values > 500 -> probably km?), try to guess:
    # Many EV ranges in miles are under 400. If > 600 it might be in km. Here we assume values > 400 are km; otherwise treat as miles and convert.
    mask = (e['electric_range_km'] <= 400) & (e['electric_range_km'] > 0)
    # We cannot be sure — we will create a miles column if needed. For now create electric_range_km and electric_range_miles
    e['electric_range_miles'] = e['electric_range_km'] * 0.621371
else:
    e['electric_range_km'] = np.nan
    e['electric_range_miles'] = np.nan

# MSRP cleaning
if 'base_msrp' in e.columns:
    e['base_msrp_numeric'] = e['base_msrp'].apply(to_numeric_safely)
else:
    # try finding price columns
    for c in e.columns:
        if 'msrp' in c or 'price' in c or 'base' in c:
            e['base_msrp_numeric'] = e[c].apply(to_numeric_safely)
            break

# Vehicle type classification
if 'vehicle_type' in e.columns:
    e['vehicle_type'] = e['vehicle_type'].astype(str).str.title()
else:
    # best effort: if fuel_type contains 'beV' or 'phev'
    if 'fuel_type' in e.columns:
        e['vehicle_type'] = e['fuel_type'].astype(str).str.upper().map(
            lambda x: 'BEV' if 'BEV' in x or 'BATTERY' in x else ('PHEV' if 'PHEV' in x or 'PLUG-IN' in x else x)
        )

# Drop duplicates on make+model+year
drop_cols = [c for c in ['make','model','model_year'] if c in e.columns]
if drop_cols:
    before = e.shape[0]
    e = e.drop_duplicates(subset=drop_cols, keep='first')
    print("Dropped duplicates from EV dataset:", before - e.shape[0])

# Impute missing electric_range_km with median per make
if 'electric_range_km' in e.columns and 'make' in e.columns:
    median_by_make = e.groupby('make')['electric_range_km'].median()
    def impute_range(row):
        if pd.isna(row['electric_range_km']):
            m = row.get('make', None)
            if pd.notna(m) and m in median_by_make.index:
                return median_by_make.loc[m]
            else:
                return e['electric_range_km'].median(skipna=True)
        return row['electric_range_km']
    e['electric_range_km'] = e.apply(impute_range, axis=1)

# Final trimmed EV cleaned DataFrame
ev_clean = e.copy()
keep_cols_ev = [c for c in ['make','model','model_year','vehicle_type','fuel_type',
                            'electric_range_km','electric_range_miles','base_msrp_numeric'] if c in ev_clean.columns]
ev_clean = ev_clean[keep_cols_ev]
print("ev_clean shape:", ev_clean.shape)
ev_clean.head(8)
# Cell 8: quick EDA / summaries
print("Top makes (EV dataset):")
if 'make' in ev_clean.columns:
    display(ev_clean['make'].value_counts().head(15))

print("\nCharger types (stations):")
if 'charger_type' in stations_clean.columns:
    display(stations_clean['charger_type'].value_counts().head(15))

print("\nCost per kWh distribution (stations):")
if 'cost_usd_per_kwh' in stations_clean.columns:
    display(stations_clean['cost_usd_per_kwh'].describe())

# Save cleaned CSVs
ev_clean_path = "Electric_Vehicle_Population_Data_cleaned.csv"
stations_clean_path = "detailed_ev_charging_stations_cleaned.csv"
ev_clean.to_csv(ev_clean_path, index=False)
stations_clean.to_csv(stations_clean_path, index=False)
print("Saved cleaned files:", ev_clean_path, stations_clean_path)

# If in Colab, offer download links
try:
    from google.colab import files
    print("You can download the cleaned files now:")
    files.download(ev_clean_path)
    files.download(stations_clean_path)
except Exception:
    print("Not in Colab or download not supported in this environment.")




#Week 2(Model Training)

# ======= Extended Week-2 Training: Battery SoH, EcoScore, Maintenance, Forecast, Charging Cost =======
# Paste into Google Colab and run.

# 0) Install dependencies
!pip install -q scikit-learn joblib pandas numpy prophet

# 1) Imports
import os, joblib, json, math, warnings
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
OUT_DIR = "models_extended"
os.makedirs(OUT_DIR, exist_ok=True)
RANDOM_STATE = 42

print("Setup done. Models will be saved to:", OUT_DIR)

# 2) Load cleaned EV CSV (upload if missing)
DATA_FILENAME = "Electric_Vehicle_Population_Data_cleaned.csv"
if not os.path.exists(DATA_FILENAME):
    from google.colab import files
    print(f"Please upload {DATA_FILENAME}")
    uploaded = files.upload()
if not os.path.exists(DATA_FILENAME):
    raise FileNotFoundError(f"{DATA_FILENAME} not found. Place it in working dir or upload via Colab UI.")
df = pd.read_csv(DATA_FILENAME, low_memory=False)
print("Loaded", DATA_FILENAME, "shape:", df.shape)
display(df.head(3))

# 3) Helpers: canonicalize + numeric safe parser
def canonicalize_columns(df):
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(' ', '_').replace('-', '_').replace('/','_').replace('.','') for c in df.columns]
    return df

def to_numeric_safely(v):
    try:
        if pd.isna(v): return np.nan
        s = str(v).replace(',','').replace('₹','').replace('$','').strip()
        import re
        s = re.sub(r'[^d\u002e\u002d]', '', s)
        return float(s) if s not in ('', '.', '-') else np.nan
    except:
        return np.nan

df = canonicalize_columns(df)

# 4) Create or detect features commonly used
# Ensure key derived features exist: model_year -> vehicle_age
if 'model_year' in df.columns:
    df['model_year'] = pd.to_numeric(df['model_year'], errors='coerce')
    df['vehicle_age'] = 2025 - df['model_year']
else:
    if 'vehicle_age' not in df.columns:
        df['vehicle_age'] = np.nan

# Numeric features candidates (you can expand)
NUMERIC_CANDIDATES = [c for c in ['battery_kwh','electric_range_km','efficiency','top_speed','accel_0_100','vehicle_age','number_of_seats'] if c in df.columns]
CATEGORICAL_CANDIDATES = [c for c in ['make','vehicle_type','fuel_type','model'] if c in df.columns]

print("Numeric features available:", NUMERIC_CANDIDATES)
print("Categorical features available:", CATEGORICAL_CANDIDATES)

# 5) Prepare small helper to encode categorical features (LabelEncoder)
def encode_categorical(df_sub, cat_cols):
    encoders = {}
    X_cat = pd.DataFrame(index=df_sub.index)
    for c in cat_cols:
        le = LabelEncoder()
        vals = df_sub[c].astype(str).fillna('nan')
        X_cat[c] = le.fit_transform(vals)
        encoders[c] = le
    return X_cat, encoders

# 6) Function to train a regression model and save artifacts
def train_regression(target_col, df, numeric_feats, cat_feats, out_dir=OUT_DIR, run_tuning=False):
    print("\n>>> Training target:", target_col)
    # If target not present, raise
    if target_col not in df.columns:
        raise ValueError(f"Target {target_col} not in dataframe.")
    # Drop rows without target
    dfm = df.dropna(subset=[target_col]).copy()
    print("Rows available:", dfm.shape[0])
    if dfm.shape[0] < 30:
        print("Warning: fewer than 30 rows for this target — results may be unreliable.")
    # Numeric matrix
    X_num = dfm[numeric_feats].copy() if numeric_feats else pd.DataFrame(index=dfm.index)
    if not X_num.empty:
        X_num = X_num.applymap(lambda v: to_numeric_safely(v))
        X_num = X_num.astype(float)
        X_num = X_num.fillna(X_num.median())
    # Categorical encode
    X_cat, encs = encode_categorical(dfm, cat_feats) if cat_feats else (pd.DataFrame(index=dfm.index), {})
    if not X_num.empty and not X_cat.empty:
        X = pd.concat([X_num, X_cat], axis=1)
    elif not X_num.empty:
        X = X_num.copy()
    elif not X_cat.empty:
        X = X_cat.copy()
    else:
        raise ValueError("No features to train on.")
    y = dfm[target_col].astype(float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    model = RandomForestRegressor(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(X_train_s, y_train)
    # Evaluate
    y_pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) # Corrected: Calculate RMSE from MSE
    r2 = r2_score(y_test, y_pred)
    print(f"Results for {target_col}: R2={r2:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}")
    # Save artifacts
    joblib.dump(model, os.path.join(out_dir, f"rf_{target_col}.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, f"scaler_{target_col}.pkl"))
    joblib.dump(encs, os.path.join(out_dir, f"encoders_{target_col}.pkl"))
    with open(os.path.join(out_dir, f"meta_{target_col}.json"), "w") as f:
        json.dump({"target": target_col, "numeric_feats": numeric_feats, "cat_feats": cat_feats}, f, indent=2)
    # Feature importances
    feat_names = X.columns.tolist()
    imp = model.feature_importances_
    fi = sorted(zip(feat_names, imp), key=lambda x: x[1], reverse=True)
    print("Top features:", fi[:6])
    # Example prediction using median row
    example = X.median().to_frame().T.fillna(0)
    try:
        xp = scaler.transform(example)
        ex_pred = float(model.predict(xp)[0])
    except:
        ex_pred = None
    return {"model": model, "scaler": scaler, "encoders": encs, "metrics": {"r2": r2, "mae": mae, "rmse": rmse}, "example_pred": ex_pred}

# ----------------------------------------------------------------
# 7) BATTERY HEALTH ESTIMATOR
# If 'battery_soh' exists, use it; otherwise create a heuristic proxy and train on it.
if 'battery_soh' in df.columns:
    print("\nBattery SoH labels found: training on 'battery_soh'.")
    battery_target = 'battery_soh'
    # numeric + cat features to use
    numeric_feats = [f for f in ['battery_kwh','electric_range_km','vehicle_age','efficiency'] if f in df.columns]
    cat_feats = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    res_batt = train_regression(battery_target, df, numeric_feats, cat_feats)
else:
    # create proxy SoH using heuristic:
    # Soh = max(30, 100 - (vehicle_age * 0.9) - (fast_charge_penalty) - temp_penalty)
    # if no fast_charge info, assume 0 or infer from charger_count column names
    print("\nNo 'battery_soh' in dataset. Creating heuristic proxy SoH and training on it.")
    df_proxy = df.copy()
    # attempt to find fast charge frequency columns or station usage proxies (best-effort)
    fastc = None
    for c in df_proxy.columns:
        if 'fast' in c and 'charge' in c:
            fastc = c; break
    if fastc is None:
        # no column — assume 0 fast charges (proxy)
        df_proxy['fast_charges_per_week'] = 0.0
    else:
        df_proxy['fast_charges_per_week'] = pd.to_numeric(df_proxy[fastc], errors='coerce').fillna(0)
    # temperature proxy not available -> assume 25°C ambient
    df_proxy['ambient_temp_c'] = df_proxy.get('ambient_temp_c', 25.0)
    # compute heuristic SoH
    df_proxy['battery_soh_proxy'] = df_proxy.apply(lambda r:
        max(20.0, 100.0 - ( (r.get('vehicle_age',0) if not pd.isna(r.get('vehicle_age',np.nan)) else 0)*0.9 )
            - min(5.0, 0.5 * r.get('fast_charges_per_week',0))
            - max(0, (r.get('ambient_temp_c',25.0)-25.0)*0.05)
        ), axis=1)
    # train model to predict battery_soh_proxy
    battery_target = 'battery_soh_proxy'
    numeric_feats = [f for f in ['battery_kwh','electric_range_km','vehicle_age','efficiency','fast_charges_per_week','ambient_temp_c'] if f in df_proxy.columns]
    cat_feats = [c for c in CATEGORICAL_CANDIDATES if c in df_proxy.columns]
    res_batt = train_regression(battery_target, df_proxy, numeric_feats, cat_feats)

# ----------------------------------------------------------------
# 8) ECOSCORE (formula + optional ML)
# Compute EcoScore for all rows as a normalized composite and train a model to predict it (if desired)
print("\nComputing EcoScore (formulaic) and training model to predict it.")
df_ec = df.copy()
# require range and efficiency to compute meaningful score
if 'electric_range_km' in df_ec.columns and 'efficiency' in df_ec.columns:
    # formula: raw = (range / efficiency) * 10 ; age penalty
    df_ec['eco_raw'] = (df_ec['electric_range_km'].astype(float).fillna(0) / df_ec['efficiency'].astype(float).replace(0, np.nan).fillna(df_ec['efficiency'].median()))
    df_ec['eco_raw'] = df_ec['eco_raw'] * 10.0
    df_ec['eco_age_penalty'] = df_ec.get('vehicle_age',0) * 1.5
    df_ec['eco_score'] = (df_ec['eco_raw'] - df_ec['eco_age_penalty']).clip(lower=0)
    # scale 0-100 roughly
    maxv = df_ec['eco_score'].quantile(0.98) if df_ec['eco_score'].notna().any() else 100.0
    df_ec['eco_score_norm'] = (df_ec['eco_score'] / (maxv+1e-9) * 100).clip(0,100)
    # train regressor to predict eco_score_norm
    eco_target = 'eco_score_norm'
    numeric_feats = [f for f in ['battery_kwh','electric_range_km','efficiency','vehicle_age'] if f in df_ec.columns]
    cat_feats = [c for c in CATEGORICAL_CANDIDATES if c in df_ec.columns]
    res_eco = train_regression(eco_target, df_ec, numeric_feats, cat_feats)
else:
    print("Not enough columns (range, efficiency) to compute EcoScore. Skipping Eco model.")
    res_eco = None

# ----------------------------------------------------------------
# 9) MAINTENANCE COST PREDICTOR
# If 'maintenance_cost' exists, train on it; else create heuristic target and train.
if 'maintenance_cost' in df.columns or 'expected_service_cost' in df.columns:
    maint_col = 'maintenance_cost' if 'maintenance_cost' in df.columns else 'expected_service_cost'
    print("\nFound maintenance cost label:", maint_col, "-> training.")
    numeric_feats = [f for f in ['vehicle_age','battery_kwh','electric_range_km','battery_soh'] if f in df.columns]
    cat_feats = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]
    res_maint = train_regression(maint_col, df, numeric_feats, cat_feats)
else:
    print("\nNo maintenance cost labels found. Creating heuristic maintenance target and training.")
    dfm = df.copy()
    # if battery_soh previously computed either real or proxy
    if 'battery_soh' in dfm.columns:
        soh_col = 'battery_soh'
    elif 'battery_soh_proxy' in dfm.columns:
        soh_col = 'battery_soh_proxy'
    else:
        soh_col = None
    # heuristic: base + age factor + battery penalty
    def compute_maint(row):
        age = row.get('vehicle_age', 3) if not pd.isna(row.get('vehicle_age', np.nan)) else 3
        battery_pen = 0
        if soh_col and soh_col in row.index and not pd.isna(row[soh_col]):
            battery_pen = (100 - row[soh_col]) * 50.0
        return 2000 + age * 500 + battery_pen
    dfm['maintenance_cost_proxy'] = dfm.apply(compute_maint, axis=1)
    maint_target = 'maintenance_cost_proxy'
    numeric_feats = [f for f in ['vehicle_age','battery_kwh','electric_range_km'] if f in dfm.columns]
    if soh_col: numeric_feats.append(soh_col)
    cat_feats = [c for c in CATEGORICAL_CANDIDATES if c in dfm.columns]
    res_maint = train_regression(maint_target, dfm, numeric_feats, cat_feats)

# ----------------------------------------------------------------
# 10) MARKET FORECAST
# If dataset has a date column (registration_date or similar), aggregate monthly and forecast with Prophet if available.
print("\nMarket Forecast: looking for date/time column to aggregate monthly registrations.")
date_col = None
for c in df.columns:
    if 'date' in c or 'time' in c or 'month' in c:
        date_col = c; break

if date_col:
    print("Found date-like column:", date_col)
    df_dates = df[[date_col]].copy()
    df_dates['ds'] = pd.to_datetime(df_dates[date_col], errors='coerce')
    df_dates = df_dates.dropna(subset=['ds'])
    if df_dates.shape[0] < 12:
        print("Too few date rows (less than 12) — producing simple year-count projection")
        # fall back: counts by model_year
        if 'model_year' in df.columns:
            counts = df['model_year'].value_counts().sort_index()
            Xy = counts.reset_index().rename(columns={'model_year':'year'})
            # linear regression on year->count
            lr = LinearRegression()
            X = Xy[['year']].astype(float)
            y = Xy['count'].astype(float)
            lr.fit(X, y)
            next_year = int(X['year'].max()) + 1
            pred_next = lr.predict([[next_year]])[0]
            print(f"Projected count next year ({next_year}): {int(pred_next)}")
            joblib.dump(lr, os.path.join(OUT_DIR, "market_lr_year_count.pkl"))
        else:
            print("No model_year to project from.")
    else:
        # aggregate monthly
        df_dates['month'] = df_dates['ds'].dt.to_period('M').dt.to_timestamp()
        monthly = df_dates.groupby('month').size().reset_index(name='count')
        monthly = monthly.sort_values('month')
        print("Monthly series length:", len(monthly))
        # try Prophet
        try:
            from prophet import Prophet
            m = Prophet(yearly_seasonality=True)
            mp = monthly.rename(columns={'month':'ds','count':'y'})
            m.fit(mp)
            future = m.make_future_dataframe(periods=12, freq='M')
            fc = m.predict(future)
            fc_tail = fc[['ds','yhat']].tail(12)
            display(fc_tail)
            joblib.dump(m, os.path.join(OUT_DIR, "market_prophet.pkl"))
            print("Prophet model saved.")
        except Exception as e:
            print("Prophet not available or failed:", e)
            # simple moving-average forecast
            vals = monthly['count'].values
            ma6 = float(vals[-6:].mean()) if len(vals) >= 6 else float(vals.mean())
            preds = [int(ma6*(1+0.02*i)) for i in range(12)]
            print("Simple MA-based 12-month forecast (approx):", preds)
            joblib.dump(preds, os.path.join(OUT_DIR, "market_ma_preds.pkl"))
else:
    print("No date-like column found. Using model_year counts to produce a simple projection (if available).")
    if 'model_year' in df.columns:
        counts = df['model_year'].value_counts().sort_index()
        Xy = counts.reset_index().rename(columns={'model_year':'year'})
        lr = LinearRegression()
        X = Xy[['year']].astype(float)
        y = Xy['count'].astype(float)
        lr.fit(X, y)
        next_year = int(X['year'].max()) + 1
        pred_next = lr.predict([[next_year]])[0]
        print(f"Projected models (count) next year ({next_year}): {int(pred_next)}")
        joblib.dump(lr, os.path.join(OUT_DIR, "market_lr_year_count.pkl"))
    else:
        print("No model_year or date info to forecast market. Skipping forecast.")

# ----------------------------------------------------------------
# 11) CHARGING SESSION COST & OPERATOR PRICE SUGGESTION
# Use stations CSV if available to compute cost per session or recommend operator price.
STATIONS_FILE = "detailed_ev_charging_stations_cleaned.csv"
if os.path.exists(STATIONS_FILE):
    stn = pd.read_csv(STATIONS_FILE)
    stn = stn.rename(columns={c:c.strip().lower().replace(' ', '_') for c in stn.columns})
    # find cost column
    cost_col = next((c for c in stn.columns if 'cost' in c and 'kwh' in c), None)
    cap_col = next((c for c in stn.columns if 'kw' in c or 'capacity' in c), None)
    if cost_col:
        stn['cost_usd_per_kwh'] = stn[cost_col].apply(to_numeric_safely)
    if cap_col:
        stn['charging_capacity_kw'] = stn[cap_col].apply(to_numeric_safely)
    # simple function for session cost
    def session_cost(kwh_needed, cost_per_kwh):
        return kwh_needed * cost_per_kwh
    # example: compute session cost for median station and 50%->80% charge of battery_kwh median
    median_batt = df['battery_kwh'].median() if 'battery_kwh' in df.columns else 50.0
    kwh_needed = median_batt * 0.6  # example 60% of battery
    if cost_col:
        median_cost = stn['cost_usd_per_kwh'].median(skipna=True)
        est_session_cost = session_cost(kwh_needed, median_cost)
        print("\nEstimated session cost (median station) for ~60% charge of median battery: ", est_session_cost, "USD")
        # suggest operator price with markup
        suggested_price = median_cost * 1.2  # 20% markup
        print("Suggested operator price per kWh (20% margin):", suggested_price, "USD/kWh")
    else:
        print("No cost per kWh found in stations file; cannot estimate session cost.")
else:
    print("\nNo stations cleaned CSV found in working directory. If you want session cost calculations, place 'detailed_ev_charging_stations_cleaned.csv' here.")

# ----------------------------------------------------------------
# 12) Summary of saved artifacts
print("\nSaved files in", OUT_DIR, ":")
print(os.listdir(OUT_DIR))
# Save a JSON summary of training metrics if available
summary = {}
if 'res_batt' in locals() and isinstance(res_batt, dict) and res_batt.get("metrics"):
    summary['battery'] = res_batt['metrics']
if 'res_eco' in locals() and isinstance(res_eco, dict) and res_eco.get("metrics"):
    summary['eco'] = res_eco['metrics']
if 'res_maint' in locals() and isinstance(res_maint, dict) and res_maint.get("metrics"):
    summary['maintenance'] = res_maint['metrics']
with open(os.path.join(OUT_DIR, "training_summary_extended.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("Saved training_summary_extended.json")


