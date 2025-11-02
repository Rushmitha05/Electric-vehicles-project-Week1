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
