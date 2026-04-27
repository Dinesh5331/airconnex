# 🛫 Smart Connection Feasibility Predictor
## Complete Implementation Plan — Data to Presentation

---

## Project Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    DATA LAYER                           │
│   AviationStack API  →  Raw CSV  →  Cleaned Dataset    │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│               FEATURE ENGINEERING LAYER                 │
│   Real Features + Synthetic Features → Final Dataset   │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                     ML LAYER                            │
│   Training → Model → Predict Transfer Margin (minutes) │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                  BACKEND (FastAPI)                      │
│   REST API  →  Predict Endpoint  →  Alert Logic        │
└────────────────────────┬────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│                FRONTEND (React/Streamlit)               │
│   Input Form → Result Display → Dashboard Analytics    │
└─────────────────────────────────────────────────────────┘
```

---

## Phase 1 — Data Collection 📥
**Goal:** Build a clean, complete raw dataset of real landed flights

### Step 1.1 — Re-collect from AviationStack (Landed Flights)

```python
# data_collection/collect_flights.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

API_KEY = "YOUR_AVIATIONSTACK_KEY"
BASE_URL = "http://api.aviationstack.com/v1/flights"

def collect_landed_flights(start_date, days=30, limit=100):
    all_flights = []
    
    for i in range(days):
        date = (start_date - timedelta(days=i)).strftime("%Y-%m-%d")
        offset = 0
        
        while True:
            params = {
                "access_key": API_KEY,
                "flight_date": date,
                "flight_status": "landed",   # ← KEY: only completed flights
                "limit": limit,
                "offset": offset
            }
            response = requests.get(BASE_URL, params=params)
            data = response.json()
            
            if "data" not in data or not data["data"]:
                break
            
            all_flights.extend(data["data"])
            offset += limit
            time.sleep(1)  # respect rate limit
            
            if offset >= data.get("pagination", {}).get("total", 0):
                break
    
    return pd.json_normalize(all_flights)

# Collect last 30 days of landed flights
df = collect_landed_flights(datetime.now(), days=30)
df.to_csv("data/raw_landed_flights.csv", index=False)
print(f"Collected {len(df)} landed flights")
```

### Step 1.2 — What You Will Get After This

| Column | Null % (Expected) |
|---|---|
| `arrival.actual` | ~10% (much better!) |
| `arrival.delay` | ~20% |
| `departure.delay` | ~15% |
| `arrival.terminal` | ~40% |
| `departure.terminal` | ~40% |

### Deliverable 📦
- `data/raw_landed_flights.csv` — 2000–5000 rows of real completed flights

---

## Phase 2 — Data Cleaning & Feature Engineering 🔧
**Goal:** Build the full feature set needed for ML

### Step 2.1 — Clean Raw Data

```python
# feature_engineering/clean.py
import pandas as pd

df = pd.read_csv("data/raw_landed_flights.csv")

# Drop completely empty columns
df.drop(columns=["aircraft", "live", "flight.codeshared"], inplace=True, errors="ignore")

# Parse datetime columns
time_cols = [
    "departure.scheduled", "departure.estimated", "departure.actual",
    "arrival.scheduled",   "arrival.estimated",   "arrival.actual"
]
for col in time_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)

# Fill missing delays from actual vs scheduled
df["arrival.delay"] = df["arrival.delay"].fillna(
    (df["arrival.actual"] - df["arrival.scheduled"]).dt.total_seconds() / 60
)
df["departure.delay"] = df["departure.delay"].fillna(
    (df["departure.actual"] - df["departure.scheduled"]).dt.total_seconds() / 60
)

df.to_csv("data/cleaned_flights.csv", index=False)
```

### Step 2.2 — Build Connection Pairs

> This is the most important step — pairing inbound + outbound flights at the same layover airport

```python
# feature_engineering/build_connections.py
import pandas as pd

df = pd.read_csv("data/cleaned_flights.csv", parse_dates=[...])

connections = []

# Group flights by layover airport
for airport in df["arrival.iata"].unique():
    arrivals  = df[df["arrival.iata"]  == airport].copy()
    departures = df[df["departure.iata"] == airport].copy()
    
    for _, inbound in arrivals.iterrows():
        arr_time = inbound["arrival.actual"] or inbound["arrival.scheduled"]
        
        # Find outbound flights leaving 30-240 min after this arrival
        valid_outbound = departures[
            (departures["departure.scheduled"] > arr_time + pd.Timedelta("30min")) &
            (departures["departure.scheduled"] < arr_time + pd.Timedelta("240min"))
        ]
        
        for _, outbound in valid_outbound.iterrows():
            connections.append({
                # Inbound flight info
                "inbound_flight":       inbound["flight.iata"],
                "inbound_airline":      inbound["airline.iata"],
                "inbound_arr_airport":  inbound["arrival.iata"],
                "inbound_arr_terminal": inbound["arrival.terminal"],
                "inbound_arr_gate":     inbound["arrival.gate"],
                "inbound_arr_scheduled":inbound["arrival.scheduled"],
                "inbound_arr_actual":   inbound["arrival.actual"],
                "inbound_arr_delay":    inbound["arrival.delay"],
                
                # Outbound flight info
                "outbound_flight":      outbound["flight.iata"],
                "outbound_airline":     outbound["airline.iata"],
                "outbound_dep_airport": outbound["departure.iata"],
                "outbound_dep_terminal":outbound["departure.terminal"],
                "outbound_dep_gate":    outbound["departure.gate"],
                "outbound_dep_scheduled":outbound["departure.scheduled"],
                
                # Layover airport
                "layover_airport":      airport,
                "flight_date":          inbound["flight_date"],
            })

conn_df = pd.DataFrame(connections)
conn_df.to_csv("data/connection_pairs.csv", index=False)
print(f"Built {len(conn_df)} connection pairs")
```

### Step 2.3 — Generate Synthetic Features

```python
# feature_engineering/add_features.py
import pandas as pd

df = pd.read_csv("data/connection_pairs.csv", parse_dates=[...])

# --- REAL DERIVED FEATURES ---

# F1: Available Connection Time (minutes)
df["available_time_min"] = (
    df["outbound_dep_scheduled"] - df["inbound_arr_actual"]
).dt.total_seconds() / 60

# F2: Terminal Change Flag
df["terminal_change"] = (
    df["inbound_arr_terminal"] != df["outbound_dep_terminal"]
).fillna(True).astype(int)  # assume change if null (safe assumption)

# F3: Same Airline
df["same_airline"] = (
    df["inbound_airline"] == df["outbound_airline"]
).astype(int)

# F4: Time of Day (departure hour)
df["hour_of_day"] = pd.to_datetime(df["outbound_dep_scheduled"]).dt.hour

# F5: Day of Week
df["day_of_week"] = pd.to_datetime(df["flight_date"]).dt.dayofweek

# --- SYNTHETIC FEATURES (Rule-Based) ---

# T1: Walking Time
df["walking_time_min"] = df["terminal_change"].map({0: 10, 1: 25})

# T2: Security Time (assume international if different airline, simplified)
df["security_time_min"] = df["same_airline"].map({1: 0, 0: 20})

# T3: Immigration (simplified: flag large international hubs)
INTL_HUBS = ["DXB", "LHR", "SIN", "AMS", "FRA", "CDG", "JFK", "NRT"]
df["requires_immigration"] = df["layover_airport"].isin(INTL_HUBS).astype(int)
df["immigration_time_min"] = df["requires_immigration"] * 30

# T4: Congestion (based on hour of day)
def congestion_time(hour):
    if 6 <= hour <= 9 or 17 <= hour <= 20:   return 15  # peak
    elif 9 <= hour <= 17:                      return 8   # normal
    else:                                      return 3   # off-peak

df["congestion_time_min"] = df["hour_of_day"].apply(congestion_time)

# T5: Baggage (default 10 min, adjustable)
df["baggage_time_min"] = 10  # default, user can override at runtime

# --- REQUIRED TRANSFER TIME (Total) ---
df["required_time_min"] = (
    df["walking_time_min"] +
    df["security_time_min"] +
    df["immigration_time_min"] +
    df["congestion_time_min"] +
    df["baggage_time_min"]
)

# --- TIME MARGIN (The Key Output) ---
df["time_margin_min"] = df["available_time_min"] - df["required_time_min"]

# --- TARGET LABEL ---
df["connection_feasible"] = (df["time_margin_min"] >= 0).astype(int)

# --- RISK CATEGORY ---
def risk_category(margin):
    if margin >= 30:    return "Safe"
    elif margin >= 10:  return "Tight"
    elif margin >= 0:   return "Risky"
    else:               return "Missed"

df["risk_category"] = df["time_margin_min"].apply(risk_category)

df.to_csv("data/final_dataset.csv", index=False)
print(f"Final dataset: {len(df)} connection pairs")
print(df["risk_category"].value_counts())
```

### Deliverable 📦
- `data/final_dataset.csv` — Complete feature-engineered dataset with target label

---

## Phase 3 — ML Model Building 🤖
**Goal:** Train a model to predict time margin or feasibility

### Step 3.1 — Feature Selection

```
Input Features (X):
  - inbound_arr_delay
  - available_time_min
  - terminal_change
  - same_airline
  - requires_immigration
  - hour_of_day
  - day_of_week
  - layover_airport (encoded)
  - inbound_airline (encoded)

Target (y):
  - time_margin_min       ← for Regression (predict exact minutes)
  - connection_feasible   ← for Classification (catch/miss)
```

### Step 3.2 — Train Model

```python
# ml/train.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

df = pd.read_csv("data/final_dataset.csv")

# Encode categorical columns
le = LabelEncoder()
for col in ["layover_airport", "inbound_airline", "outbound_airline"]:
    df[col] = le.fit_transform(df[col].fillna("UNKNOWN"))

features = [
    "inbound_arr_delay", "available_time_min", "terminal_change",
    "same_airline", "requires_immigration", "hour_of_day",
    "day_of_week", "layover_airport", "inbound_airline"
]

X = df[features].fillna(0)
y = df["connection_feasible"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model
joblib.dump(model, "ml/model.pkl")
print("Model saved!")
```

### Step 3.3 — Model Evaluation Metrics

| Metric | Target |
|---|---|
| Accuracy | > 80% |
| Precision (Missed class) | > 75% |
| Recall (Missed class) | > 75% |
| F1 Score | > 0.78 |

### Deliverable 📦
- `ml/model.pkl` — Trained classification model
- `ml/evaluation_report.txt` — Accuracy, precision, recall, F1

---

## Phase 4 — Backend API (FastAPI) ⚙️
**Goal:** Create a REST API that accepts flight inputs and returns prediction

### Step 4.1 — Project Structure

```
backend/
├── main.py
├── model.pkl
├── predictor.py
├── airport_rules.py
└── requirements.txt
```

### Step 4.2 — Prediction Endpoint

```python
# backend/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from predictor import predict_connection

app = FastAPI(title="Smart Connection Feasibility Predictor API")
model = joblib.load("model.pkl")

class ConnectionInput(BaseModel):
    inbound_flight: str          # e.g. "EK521"
    outbound_flight: str         # e.g. "EK204"
    layover_airport: str         # e.g. "DXB"
    inbound_arr_delay: float     # minutes
    available_time_min: float    # minutes
    terminal_change: int         # 0 or 1
    same_airline: int            # 0 or 1
    requires_immigration: int    # 0 or 1
    hour_of_day: int             # 0-23
    day_of_week: int             # 0-6
    passenger_type: str          # "fast", "average", "elderly"
    has_baggage: int             # 0 or 1

@app.post("/predict")
def predict(input: ConnectionInput):
    result = predict_connection(model, input)
    return {
        "available_time_min":  result["available_time"],
        "required_time_min":   result["required_time"],
        "time_margin_min":     result["margin"],
        "risk_category":       result["category"],
        "connection_feasible": result["feasible"],
        "breakdown": {
            "walking_time":    result["walking"],
            "security_time":   result["security"],
            "immigration_time":result["immigration"],
            "congestion_time": result["congestion"],
            "baggage_time":    result["baggage"]
        }
    }

@app.get("/health")
def health():
    return {"status": "running"}
```

### Step 4.3 — Run Backend

```bash
pip install fastapi uvicorn scikit-learn joblib pandas
uvicorn main:app --reload --port 8000
```

### Deliverable 📦
- FastAPI running at `http://localhost:8000`
- Swagger docs at `http://localhost:8000/docs`

---

## Phase 5 — Frontend Dashboard 🖥️
**Goal:** User-facing interface to input flights and see results

### Option A — Streamlit (Quick, Academic Demo)

```python
# frontend/app.py
import streamlit as st
import requests

st.set_page_config(page_title="Smart Connection Predictor", page_icon="✈️")
st.title("✈️ Smart Connection Feasibility Predictor")

col1, col2 = st.columns(2)
with col1:
    inbound_flight   = st.text_input("Inbound Flight", "EK521")
    layover_airport  = st.text_input("Layover Airport (IATA)", "DXB")
    arr_delay        = st.number_input("Inbound Arrival Delay (min)", 0, 300, 15)
    available_time   = st.number_input("Available Connection Time (min)", 0, 600, 55)

with col2:
    outbound_flight  = st.text_input("Outbound Flight", "EK204")
    terminal_change  = st.selectbox("Terminal Change?", [0, 1])
    passenger_type   = st.selectbox("Passenger Type", ["fast", "average", "elderly"])
    has_baggage      = st.selectbox("Checked Baggage?", [0, 1])

if st.button("🔍 Check Connection Feasibility"):
    response = requests.post("http://localhost:8000/predict", json={...})
    result = response.json()

    margin = result["time_margin_min"]
    category = result["risk_category"]

    # Color-coded result
    color = {"Safe": "green", "Tight": "orange", "Risky": "red", "Missed": "darkred"}
    st.markdown(f"### Result: :{color[category]}[{category}]")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Available Time",  f"{result['available_time_min']} min")
    col2.metric("Required Time",   f"{result['required_time_min']} min")
    col3.metric("⏱️ Time Margin",   f"{margin:+.0f} min",
                delta_color="normal" if margin >= 0 else "inverse")

    # Breakdown
    st.subheader("Transfer Time Breakdown")
    breakdown = result["breakdown"]
    st.bar_chart(breakdown)
```

### Option B — React Dashboard (Full Web App)
- Input form for flight details
- Real-time margin display with color coding
- Breakdown chart (walking / security / immigration / congestion)
- Airport analytics dashboard (high-risk airports, peak times)

### Deliverable 📦
- Working demo app accessible in browser
- Shows: margin in minutes + risk category + breakdown

---

## Phase 6 — Testing & Presentation 🎯

### Step 6.1 — Test with Real Scenarios

| Test Case | Input | Expected Output |
|---|---|---|
| Safe connection | 90 min available, same terminal | +35 min margin — Safe |
| Tight connection | 55 min, terminal change, delay | +8 min — Tight |
| Missed connection | 45 min, immigration, delay 20 min | -12 min — Missed |
| DXB layover, elderly | 60 min, international | Risky |

### Step 6.2 — Presentation Structure

```
Slide 1: Problem — MCT is not enough
Slide 2: Solution — Smart Feasibility Predictor
Slide 3: System Architecture (diagram)
Slide 4: Data Pipeline (collection → features → model)
Slide 5: Feature Engineering (what we derived & why)
Slide 6: ML Model — accuracy, metrics
Slide 7: Live Demo — enter a flight, see margin output
Slide 8: Sample Results & Case Studies
Slide 9: Future Scope (real-time alerts, mobile app)
Slide 10: Conclusion
```

---

## 📅 Timeline

| Phase | Task | Duration |
|---|---|---|
| Phase 1 | Data Collection (AviationStack re-fetch) | Day 1 |
| Phase 2 | Cleaning + Feature Engineering | Day 2–3 |
| Phase 3 | ML Model Training & Evaluation | Day 4–5 |
| Phase 4 | FastAPI Backend | Day 5–6 |
| Phase 5 | Frontend (Streamlit or React) | Day 6–8 |
| Phase 6 | Testing + Presentation | Day 9–10 |

---

## 📁 Final Project Structure

```
smart-connection-predictor/
├── data/
│   ├── raw_landed_flights.csv
│   ├── cleaned_flights.csv
│   ├── connection_pairs.csv
│   └── final_dataset.csv
├── data_collection/
│   └── collect_flights.py
├── feature_engineering/
│   ├── clean.py
│   ├── build_connections.py
│   └── add_features.py
├── ml/
│   ├── train.py
│   ├── evaluate.py
│   └── model.pkl
├── backend/
│   ├── main.py
│   ├── predictor.py
│   └── requirements.txt
├── frontend/
│   └── app.py  (Streamlit)
├── notebooks/
│   └── EDA.ipynb
└── README.md
```

---

## ✅ Final Output of the System

```
Input:  EK521 (arrives DXB) → EK204 (departs DXB)
        Delay: 20 min | Terminal: T1 → T3 | Baggage: Yes

Output:
  Available Time :  55 min
  Required Time  :  63 min
  ─────────────────────────
  Time Margin    : -8 min   ❌ MISSED
  Risk Category  : Missed

  Breakdown:
    Walking Time      : 25 min  (diff terminal)
    Security Time     : 20 min  (international)
    Immigration Time  : 30 min  (DXB hub)
    Congestion Time   : 15 min  (peak hour)
    Baggage Time      : 10 min
    ─────────────────────
    Total Required    : 63 min
```
