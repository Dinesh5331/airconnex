"""
preprocessing.py
Complete 11-step preprocessing pipeline for aviation connection feasibility dataset.
Output: preprocessed_dataset.csv  +  X_train/X_test/y_train/y_test as .npy files
"""

import os, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
RAW  = os.path.join(BASE, "aviation_combined_dataset.csv")
OUT  = os.path.join(BASE, "preprocessed_dataset.csv")

SEP  = "=" * 65

# ── STEP 1 — RAW AUDIT ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 1 — RAW AUDIT")
print(SEP)

df_raw = pd.read_csv(RAW)
print(f"Raw shape : {df_raw.shape}")

null_pct = (df_raw.isnull().sum() / len(df_raw) * 100).sort_values(ascending=False)
print("\nColumn inventory (sorted by missing %):\n")
print(f"  {'Column':<45} {'Null%':>6}  {'Dtype':<12}  {'Unique':>7}")
print(f"  {'-'*45} {'-'*6}  {'-'*12}  {'-'*7}")
for col, pct in null_pct.items():
    flag = " ◀ DROP" if pct >= 90 else (" ◀ SPARSE" if pct >= 60 else "")
    print(f"  {col:<45} {pct:6.1f}%  {str(df_raw[col].dtype):<12}  {df_raw[col].nunique():>7}{flag}")

# ── STEP 2 — REMOVE SYNTHETIC ROWS ───────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 2 — REMOVE SYNTHETIC / DISTANCE-REFERENCE ROWS")
print(SEP)

df = df_raw[df_raw["flight_status"] != "distance_reference"].copy()
print(f"Before : {len(df_raw):,} rows")
print(f"Removed: {len(df_raw) - len(df):,} distance_reference rows")
print(f"After  : {len(df):,} rows (scheduled flights only)")

# ── STEP 3 — DROP COLUMNS BY 5 RULES ─────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 3 — DROP COLUMNS (5 automated rules)")
print(SEP)

def drop_reason(col, series, total):
    pct = series.isnull().sum() / total * 100
    if pct == 100:             return "100% null"
    if pct >= 90:              return ">90% null"
    if pct >= 60:              return ">60% null"
    # Post-flight leakage columns
    leakage_keywords = ["actual", "estimated_runway", "estimated"]
    if any(kw in col for kw in leakage_keywords): return "post-flight leakage"
    # ICAO duplicates (IATA already present)
    if col.endswith(".icao") or col.endswith(".icao24"): return "ICAO duplicate"
    # Codeshare sub-fields (already captured in airline.iata/name)
    if "codeshared." in col:   return "codeshare duplicate"
    # Pure identifiers / metadata with no predictive signal
    id_exact = {"flight.codeshared", "aircraft", "live", "flight_status",
                "departure.airport", "arrival.airport",
                "departure.timezone", "arrival.timezone",
                "flight.number", "flight.iata", "flight.icao",
                "airline.icao", "airline.name",
                "terminal_from", "terminal_to"}
    if col in id_exact:        return "identifier / no signal"
    return None

n_before = df.shape[1]
dropped = {}
for col in list(df.columns):
    reason = drop_reason(col, df[col], len(df))
    if reason:
        dropped[col] = reason

df.drop(columns=list(dropped.keys()), inplace=True)
print(f"Dropped {len(dropped)} columns  |  {n_before} → {df.shape[1]} remaining\n")
for col, reason in dropped.items():
    print(f"  ✗  {col:<50}  [{reason}]")

print(f"\nSurviving columns ({df.shape[1]}):")
for c in df.columns:
    print(f"  ✔  {c}")

# ── STEP 4 — PARSE DATETIMES & DERIVE TIME FEATURES ──────────────────────────
print(f"\n{SEP}")
print("STEP 4 — PARSE DATETIMES & DERIVE TIME FEATURES")
print(SEP)

df["dep_sched"] = pd.to_datetime(df["departure.scheduled"], utc=True, errors="coerce")
df["arr_sched"] = pd.to_datetime(df["arrival.scheduled"],   utc=True, errors="coerce")

df["scheduled_flight_min"] = (df["arr_sched"] - df["dep_sched"]).dt.total_seconds() / 60
df["departure_hour"]       = df["dep_sched"].dt.hour
df["day_of_week"]          = df["dep_sched"].dt.dayofweek   # 0=Mon … 6=Sun

# Drop rows with nonsensical scheduled durations
bad = df["scheduled_flight_min"] <= 0
df = df[~bad].copy()
print(f"Removed {bad.sum()} rows with scheduled_flight_min ≤ 0")
print(f"Shape after step 4: {df.shape}")
print(f"  departure_hour  range: {df['departure_hour'].min()} – {df['departure_hour'].max()}")
print(f"  day_of_week     range: {df['day_of_week'].min()} – {df['day_of_week'].max()}")
print(f"  scheduled_flight_min  mean: {df['scheduled_flight_min'].mean():.1f} min")

df.drop(columns=["departure.scheduled", "arrival.scheduled", "dep_sched", "arr_sched",
                  "flight_date"], inplace=True, errors="ignore")

# ── STEP 5 — TARGET VALIDATION & ORDINAL ENCODING ────────────────────────────
print(f"\n{SEP}")
print("STEP 5 — TARGET VALIDATION & ORDINAL ENCODING")
print(SEP)

print(f"Null in connection_risk: {df['connection_risk'].isnull().sum()}")
print("Raw class counts:")
print(df["connection_risk"].value_counts().to_string())

RISK_MAP = {"Risky": 0, "Tight": 1, "Safe": 2}
df["risk_label"] = df["connection_risk"].map(RISK_MAP)
df.drop(columns=["connection_risk"], inplace=True)

print("\nEncoded classes  →  Risky=0  |  Tight=1  |  Safe=2")
print(df["risk_label"].value_counts().sort_index().to_string())

# ── STEP 6 — FEATURE ENGINEERING ─────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 6 — FEATURE ENGINEERING")
print(SEP)

# 6a. Transport friction score (1=easiest … 5=hardest)
FRICTION = {
    "walkway"          : 1,
    "moving walkway"   : 1,
    "airside connector": 2,
    "skytrain"         : 2,
    "monorail"         : 2,
    "automated people mover": 2,
    "link train"       : 2,
    "airtrain"         : 2,
    "shuttle"          : 3,
    "shuttle bus"      : 3,
    "bus"              : 3,
    "metro"            : 3,
    "walkway/shuttle"  : 3,
    "shuttle bus / metro": 3,
}
def friction_score(val):
    if pd.isna(val): return 3
    v = str(val).lower().strip()
    for kw, score in FRICTION.items():
        if kw in v: return score
    return 3  # default mid

df["transport_friction"] = df["terminal_transport_method"].apply(friction_score)
df.drop(columns=["terminal_transport_method"], inplace=True)
print("transport_friction distribution:")
print(df["transport_friction"].value_counts().sort_index().to_string())

# 6b. International route flag (arrival IATA not same as departure IATA)
df["is_cross_airport"] = (df["departure.iata"] != df["arrival.iata"]).astype(int)

# 6c. International HUB flag
INTL_HUBS = {"DXB","LHR","SIN","AMS","FRA","CDG","JFK","NRT","IST","HKG","ORD","LAX"}
df["is_intl_hub"] = df["departure.iata"].isin(INTL_HUBS).astype(int)

# 6d. Peak hour flag (rush: 6–9 and 17–20)
df["is_peak_hour"] = df["departure_hour"].apply(
    lambda h: 1 if (6 <= h <= 9 or 17 <= h <= 20) else 0
)

# 6e. Terminal change flag (departure terminal ≠ arrival terminal at layover airport)
df["terminal_change"] = (
    df["departure.terminal"].fillna("?") != df["arrival.terminal"].fillna("??")
).astype(int)

# 6f. Has actual delay info flag (before we impute)
df["has_delay_info"] = df["departure.delay"].notna().astype(int)

# 6g. Required time components (mirrors risk label logic)
df["walking_time_min"]     = df["terminal_walk_time_min"]
df["security_time_min"]    = 20   # standard (no same-airline column available)
df["immigration_time_min"] = df["is_intl_hub"] * 30
df["congestion_time_min"]  = df["departure_hour"].apply(
    lambda h: 15 if (6 <= h <= 9 or 17 <= h <= 20) else 8 if 9 <= h <= 17 else 3
)
df["baggage_time_min"]     = 10   # default

df["required_time_min"] = (
    df["walking_time_min"] + df["security_time_min"] +
    df["immigration_time_min"] + df["congestion_time_min"] + df["baggage_time_min"]
)

# 6h. Computed time margin (key derived feature)
df["time_margin_min"] = (
    df["connection_time_min"] - df["arrival_delay_min"] - df["required_time_min"]
)

print("\nEngineered features created:")
new_feats = ["transport_friction","is_cross_airport","is_intl_hub","is_peak_hour",
             "terminal_change","has_delay_info","required_time_min","time_margin_min"]
for f in new_feats:
    print(f"  {f:30s}  mean={df[f].mean():.2f}  null={df[f].isnull().sum()}")

# ── STEP 7 — MISSING VALUES ───────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 7 — MISSING VALUES")
print(SEP)

# Categorical → "Unknown"
cat_cols = df.select_dtypes(include="object").columns.tolist()
for col in cat_cols:
    n = df[col].isnull().sum()
    if n > 0:
        df[col] = df[col].fillna("Unknown")
        print(f"  {col}: filled {n} nulls → 'Unknown'")

# Numeric → median
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    n = df[col].isnull().sum()
    if n > 0:
        med = df[col].median()
        df[col] = df[col].fillna(med)
        print(f"  {col}: filled {n} nulls → median={med:.2f}")

remaining_null = df.isnull().sum().sum()
print(f"\nTotal nulls remaining: {remaining_null}  ✓")

# ── STEP 8 — OUTLIER TREATMENT (IQR WINSORISATION) ───────────────────────────
print(f"\n{SEP}")
print("STEP 8 — OUTLIER TREATMENT (IQR WINSORISATION)")
print(SEP)

SKIP_OUTLIER = {"risk_label","is_intl_hub","is_cross_airport","is_peak_hour",
                "terminal_change","has_delay_info","transport_friction",
                "day_of_week","departure_hour","security_time_min","baggage_time_min"}

outlier_summary = []
for col in df.select_dtypes(include=[np.number]).columns:
    if col in SKIP_OUTLIER:
        continue
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out = ((df[col] < lo) | (df[col] > hi)).sum()
    if n_out > 0:
        df[col] = df[col].clip(lo, hi)
        outlier_summary.append((col, n_out, lo, hi))

for col, n, lo, hi in outlier_summary:
    print(f"  {col:<35}  {n:4d} outliers clipped to [{lo:.1f}, {hi:.1f}]")
print(f"\nTotal columns winsorised: {len(outlier_summary)}")

# ── STEP 9 — ENCODING ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("STEP 9 — ENCODING")
print(SEP)

# Identify remaining object columns (excluding the few we want to keep for reference)
encode_cols = [c for c in df.select_dtypes(include="object").columns
               if c not in ("risk_label",)]

# High-cardinality → target encoding (mean of risk_label)
HIGH_CARD_THRESH = 15
te_cols = [c for c in encode_cols if df[c].nunique() > HIGH_CARD_THRESH]
le_cols = [c for c in encode_cols if df[c].nunique() <= HIGH_CARD_THRESH]

print(f"\nTarget encoding ({len(te_cols)} columns, cardinality > {HIGH_CARD_THRESH}):")
target_encoders = {}
for col in te_cols:
    mapping = df.groupby(col)["risk_label"].mean()
    df[col + "_te"] = df[col].map(mapping).fillna(df["risk_label"].mean())
    target_encoders[col] = mapping
    print(f"  {col:<40}  {df[col].nunique()} unique → mean-target encoded")
    df.drop(columns=[col], inplace=True)

print(f"\nLabel encoding ({len(le_cols)} columns, cardinality ≤ {HIGH_CARD_THRESH}):")
label_encoders = {}
for col in le_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le
    print(f"  {col:<40}  {len(le.classes_)} classes → label encoded")

# Drop terminal columns (already captured in terminal_change)
df.drop(columns=["departure.terminal","arrival.terminal"], inplace=True, errors="ignore")

print(f"\nFinal columns ({df.shape[1]}):")
for c in df.columns:
    print(f"  {c}")

# ── STEP 10 — STRATIFIED TRAIN/TEST SPLIT ────────────────────────────────────
print(f"\n{SEP}")
print("STEP 10 — STRATIFIED TRAIN / TEST SPLIT  (80 / 20)")
print(SEP)

FEATURE_COLS = [c for c in df.columns if c != "risk_label"]
X = df[FEATURE_COLS].values
y = df["risk_label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"Training set : {X_train.shape[0]:,} rows  |  {X_train.shape[1]} features")
print(f"Test set     : {X_test.shape[0]:,}  rows  |  {X_test.shape[1]} features")
print("\nClass distribution (train):")
for k, v in zip(*np.unique(y_train, return_counts=True)):
    print(f"  {['Risky','Tight','Safe'][k]:6s} ({k})  {v:5d}  ({v/len(y_train)*100:.1f}%)")
print("Class distribution (test):")
for k, v in zip(*np.unique(y_test, return_counts=True)):
    print(f"  {['Risky','Tight','Safe'][k]:6s} ({k})  {v:5d}  ({v/len(y_test)*100:.1f}%)")

# ── STEP 11 — SCALING (MinMax, no leakage) ───────────────────────────────────
print(f"\n{SEP}")
print("STEP 11 — MINMAX SCALING  (fit on train only)")
print(SEP)

scaler = MinMaxScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print("All features scaled to [0, 1]")
print(f"X_train_scaled shape : {X_train_sc.shape}")
print(f"X_test_scaled  shape : {X_test_sc.shape}")

# ── SAVE OUTPUTS ──────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("SAVING OUTPUTS")
print(SEP)

df.to_csv(OUT, index=False)
print(f"preprocessed_dataset.csv saved  ({df.shape[0]:,} rows × {df.shape[1]} cols)")

np.save(os.path.join(BASE, "X_train.npy"), X_train_sc)
np.save(os.path.join(BASE, "X_test.npy"),  X_test_sc)
np.save(os.path.join(BASE, "y_train.npy"), y_train)
np.save(os.path.join(BASE, "y_test.npy"),  y_test)
pd.Series(FEATURE_COLS).to_csv(os.path.join(BASE, "feature_names.csv"), index=False, header=False)

print("X_train.npy  X_test.npy  y_train.npy  y_test.npy  feature_names.csv  — all saved")
print(f"\n{'='*65}")
print("PREPROCESSING COMPLETE ✓")
print(f"{'='*65}\n")
