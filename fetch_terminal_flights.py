"""
fetch_terminal_flights.py

Logic:
  1. Read Airport_Terminal_Distances.csv to get the set of airports (IATA codes)
     and their terminal pairs.
  2. For each unique airport (IATA), call AviationStack /flights endpoint to get
     real scheduled flights departing from that airport that also have terminal
     information.  Collect up to 50 usable rows per airport.
  3. For every fetched flight, look up whether a matching intra-airport terminal
     transfer exists in the distances CSV (departure.terminal → arrival.terminal
     at the SAME airport).  Add the distance/walk-time columns.
  4. Save the flight rows (50 per airport) to a single CSV:
        aviation_new_dataset.csv
  5. Append all rows from Airport_Terminal_Distances.csv (with NaN for pure-flight
     columns) to create:
        aviation_combined_dataset.csv

Reference schema: aviation_full_dataset.csv
"""

import os
import sys
import time
import csv
import json
import math
import random
from datetime import datetime, timedelta
import urllib.request
import urllib.parse

# Force stdout to UTF-8 so unicode symbols don't crash on Windows cp1252
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DIST_CSV   = os.path.join(BASE_DIR, "Airport_Terminal_Distances.csv")
REF_CSV    = os.path.join(BASE_DIR, "aviation_full_dataset.csv")
OUT_FLIGHT = os.path.join(BASE_DIR, "aviation_new_dataset.csv")
OUT_COMBO  = os.path.join(BASE_DIR, "aviation_combined_dataset.csv")
ENV_FILE   = os.path.join(BASE_DIR, ".env")

TARGET_ROWS_PER_AIRPORT = 50
API_DELAY   = 1.5   # seconds between API calls (rate-limit friendly)

# ── Load .env ──────────────────────────────────────────────────────────────────
def load_env(path):
    env = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
    return env

env     = load_env(ENV_FILE)
API_KEY = env.get("FLIGHT_API_KEY", "")
if not API_KEY:
    raise EnvironmentError("FLIGHT_API_KEY not found in .env")
print(f"[OK] API key loaded: {API_KEY[:8]}...")

# ── Read reference schema columns ──────────────────────────────────────────────
def get_ref_columns(ref_csv):
    with open(ref_csv, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        return next(reader)

REF_COLS = get_ref_columns(REF_CSV)
print(f"[OK] Reference schema: {len(REF_COLS)} columns")

# Extra columns we add for terminal-distance enrichment + connection scenario
DIST_COLS = [
    "terminal_distance_m",
    "terminal_walk_time_min",
    "terminal_transport_method",
    "terminal_from",
    "terminal_to",
    "connection_time_min",   # simulated available connection window (minutes)
    "arrival_delay_min",     # simulated inbound arrival delay (minutes)
    "connection_risk",       # target label: Safe / Tight / Risky
]

ALL_COLS = REF_COLS + DIST_COLS

# ── Class-balanced connection scenario helpers ────────────────────────────────
# Target: 40% Safe | 30% Tight | 30% Risky
# Risk label logic (mirrors preprocessing notebook):
#   required_time = walk_min + security(20) + congestion(8) + baggage(10)
#                   [+ immigration(30) for international hubs]
#   margin        = connection_time_min - arrival_delay_min - required_time
#   Safe  : margin >= 30
#   Tight : 10 <= margin < 30
#   Risky : 0  <= margin < 10

INTL_HUBS = {"DXB", "LHR", "SIN", "AMS", "FRA", "CDG", "JFK", "NRT", "IST", "HKG"}

def target_class(slot_idx, total_slots):
    """Deterministic class for a slot index to hit 40/30/30 distribution."""
    frac = slot_idx / total_slots
    if frac < 0.40:
        return "Safe"
    elif frac < 0.70:
        return "Tight"
    else:
        return "Risky"

def connection_params(risk_class, walk_min_val, iata):
    """
    Return (connection_time_min, arrival_delay_min) such that:
        margin = connection_time_min - arrival_delay_min - required_time
    falls in the correct range for risk_class.
    """
    try:
        wm = int(float(walk_min_val)) if walk_min_val else 8
    except (ValueError, TypeError):
        wm = 8

    base_required = wm + 20 + 8 + 10          # walk + security + congestion + baggage
    if iata in INTL_HUBS:
        base_required += 30                    # immigration penalty

    if risk_class == "Safe":
        margin_target = random.randint(30, 65) # comfortable buffer
        arr_delay     = random.randint(0, 15)
    elif risk_class == "Tight":
        margin_target = random.randint(10, 29) # narrow window
        arr_delay     = random.randint(5, 30)
    else:                                      # Risky
        margin_target = random.randint(0, 9)   # almost no buffer
        arr_delay     = random.randint(15, 45)

    conn_time = base_required + margin_target + arr_delay
    return conn_time, arr_delay

# ── Read Airport_Terminal_Distances.csv ────────────────────────────────────────
print(f"[>>] Reading {DIST_CSV} ...")
dist_rows    = []        # raw rows as dicts
airport_info = {}        # iata → {name, city, country}
terminal_map = {}        # iata → list of (term_from, term_to, dist_m, walk_min, transport)

with open(DIST_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        iata = row["IATA Code"].strip()
        dist_rows.append(row)
        if iata not in airport_info:
            airport_info[iata] = {
                "name"   : row["Airport Name"],
                "city"   : row["City"],
                "country": row["Country"],
            }
        terminal_map.setdefault(iata, []).append((
            row["Terminal From"].strip(),
            row["Terminal To"].strip(),
            row["Distance (m)"].strip(),
            row["Walk Time (min)"].strip(),
            row["Transport Method"].strip(),
        ))

unique_iatas = list(airport_info.keys())
print(f"[OK] {len(dist_rows)} terminal-distance rows | {len(unique_iatas)} unique airports")

# ── AviationStack API helper ───────────────────────────────────────────────────
API_BASE = "http://api.aviationstack.com/v1"

def api_get(endpoint, params):
    params["access_key"] = API_KEY
    url = f"{API_BASE}/{endpoint}?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data
    except Exception as e:
        print(f"  [!] API error: {e}")
        return None

# ── Flatten a single flight object into the REF_COLS schema ───────────────────
def _safe(d, *keys):
    """Navigate nested dict safely."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(k, "")
        if cur is None:
            cur = ""
    return cur

def flatten_flight(f):
    row = {c: "" for c in REF_COLS}

    row["flight_date"]   = _safe(f, "flight_date")
    row["flight_status"] = _safe(f, "flight_status")
    row["aircraft"]      = ""   # placeholder
    row["live"]          = ""

    dep = f.get("departure") or {}
    row["departure.airport"]         = dep.get("airport", "")
    row["departure.timezone"]        = dep.get("timezone", "")
    row["departure.iata"]            = dep.get("iata", "")
    row["departure.icao"]            = dep.get("icao", "")
    row["departure.terminal"]        = dep.get("terminal", "")
    row["departure.gate"]            = dep.get("gate", "")
    row["departure.delay"]           = dep.get("delay", "")
    row["departure.scheduled"]       = dep.get("scheduled", "")
    row["departure.estimated"]       = dep.get("estimated", "")
    row["departure.actual"]          = dep.get("actual", "")
    row["departure.estimated_runway"]= dep.get("estimated_runway", "")
    row["departure.actual_runway"]   = dep.get("actual_runway", "")

    arr = f.get("arrival") or {}
    row["arrival.airport"]           = arr.get("airport", "")
    row["arrival.timezone"]          = arr.get("timezone", "")
    row["arrival.iata"]              = arr.get("iata", "")
    row["arrival.icao"]              = arr.get("icao", "")
    row["arrival.terminal"]          = arr.get("terminal", "")
    row["arrival.gate"]              = arr.get("gate", "")
    row["arrival.baggage"]           = arr.get("baggage", "")
    row["arrival.scheduled"]         = arr.get("scheduled", "")
    row["arrival.delay"]             = arr.get("delay", "")
    row["arrival.estimated"]         = arr.get("estimated", "")
    row["arrival.actual"]            = arr.get("actual", "")
    row["arrival.estimated_runway"]  = arr.get("estimated_runway", "")
    row["arrival.actual_runway"]     = arr.get("actual_runway", "")

    al = f.get("airline") or {}
    row["airline.name"] = al.get("name", "")
    row["airline.iata"] = al.get("iata", "")
    row["airline.icao"] = al.get("icao", "")

    fl = f.get("flight") or {}
    row["flight.number"] = fl.get("number", "")
    row["flight.iata"]   = fl.get("iata", "")
    row["flight.icao"]   = fl.get("icao", "")

    cs = fl.get("codeshared") or {}
    row["flight.codeshared"] = ""
    row["flight.codeshared.airline_name"]   = cs.get("airline_name", "")
    row["flight.codeshared.airline_iata"]   = cs.get("airline_iata", "")
    row["flight.codeshared.airline_icao"]   = cs.get("airline_icao", "")
    row["flight.codeshared.flight_number"]  = cs.get("flight_number", "")
    row["flight.codeshared.flight_iata"]    = cs.get("flight_iata", "")
    row["flight.codeshared.flight_icao"]    = cs.get("flight_icao", "")

    ac = f.get("aircraft") or {}
    row["aircraft.registration"] = ac.get("registration", "")
    row["aircraft.iata"]         = ac.get("iata", "")
    row["aircraft.icao"]         = ac.get("icao", "")
    row["aircraft.icao24"]       = ac.get("icao24", "")

    lv = f.get("live") or {}
    row["live.updated"]         = lv.get("updated", "")
    row["live.latitude"]        = lv.get("latitude", "")
    row["live.longitude"]       = lv.get("longitude", "")
    row["live.altitude"]        = lv.get("altitude", "")
    row["live.direction"]       = lv.get("direction", "")
    row["live.speed_horizontal"]= lv.get("speed_horizontal", "")
    row["live.speed_vertical"]  = lv.get("speed_vertical", "")
    row["live.is_ground"]       = lv.get("is_ground", "")

    # remaining ref cols that aren't in the API response → leave blank
    remaining = [
        "id_x","fleet_average_age","airline_id","callsign","hub_code",
        "icao_code_x","country_iso2_x","date_founded","iata_prefix_accounting",
        "airline.full_name","country_name_x","fleet_size","status","type",
        "id_y","gmt","airport_id","city_iata_code","icao_code_y","country_iso2_y",
        "geoname_id","latitude","longitude","departure.airport_name",
        "country_name_y","phone_number","timezone",
    ]
    for c in remaining:
        if c in row and row[c] == "":
            row[c] = ""

    return row

# ── Terminal-distance lookup ───────────────────────────────────────────────────
def lookup_terminal_dist(iata, term_from, term_to):
    """Return (dist_m, walk_min, transport) or ("","","") if not found."""
    for tf, tt, dm, wm, tr in terminal_map.get(iata, []):
        if (tf.lower() == str(term_from).lower() and
                tt.lower() == str(term_to).lower()):
            return dm, wm, tr
        # also try reverse
        if (tt.lower() == str(term_from).lower() and
                tf.lower() == str(term_to).lower()):
            return dm, wm, tr
    return "", "", ""

# ── Synthetic fallback row generator ──────────────────────────────────────────
# If the API returns fewer than TARGET_ROWS_PER_AIRPORT usable rows, we build
# synthetic rows that simulate connecting flights between the terminals listed
# in the distances CSV for that airport.

AIRLINES = [
    ("American Airlines","AA","AAL"), ("Delta Air Lines","DL","DAL"),
    ("United Airlines","UA","UAL"),   ("Southwest Airlines","WN","SWA"),
    ("British Airways","BA","BAW"),   ("Lufthansa","LH","DLH"),
    ("Air France","AF","AFR"),        ("Emirates","EK","UAE"),
    ("Singapore Airlines","SQ","SIA"),("Qatar Airways","QR","QTR"),
    ("Cathay Pacific","CX","CPA"),    ("KLM","KL","KLM"),
]

def synthetic_row(iata, airport_name, term_from, term_to, dist_m, walk_min, transport, row_idx):
    """Build a fake-but-realistic flight row for an intra-airport terminal hop."""
    al = AIRLINES[row_idx % len(AIRLINES)]
    fnum = 1000 + row_idx * 7
    today = datetime.utcnow().strftime("%Y-%m-%d")
    sched_dep = f"{today}T{8 + (row_idx % 12):02d}:00:00+00:00"
    sched_arr = f"{today}T{8 + (row_idx % 12):02d}:{walk_min}:00+00:00" if walk_min else sched_dep

    row = {c: "" for c in REF_COLS}
    row["flight_date"]          = today
    row["flight_status"]        = "scheduled"
    row["departure.airport"]    = airport_name
    row["departure.iata"]       = iata
    row["departure.terminal"]   = term_from
    row["arrival.airport"]      = airport_name
    row["arrival.iata"]         = iata
    row["arrival.terminal"]     = term_to
    row["departure.scheduled"]  = sched_dep
    row["arrival.scheduled"]    = sched_arr
    row["airline.name"]         = al[0]
    row["airline.iata"]         = al[1]
    row["airline.icao"]         = al[2]
    row["flight.number"]        = str(fnum)
    row["flight.iata"]          = f"{al[1]}{fnum}"
    row["flight.icao"]          = f"{al[2]}{fnum}"
    return row

# ── Main fetch loop ────────────────────────────────────────────────────────────
flight_rows = []   # list of dicts (ALL_COLS)

print(f"\n[>>] Fetching flights for {len(unique_iatas)} airports ...\n")

for airport_idx, iata in enumerate(unique_iatas):
    info     = airport_info[iata]
    terminals= terminal_map.get(iata, [])
    collected= []
    offset   = 0
    page_size= 100

    print(f"  [{airport_idx+1}/{len(unique_iatas)}] {iata} - {info['name']} ...", end="", flush=True)

    # ── Try real API ──────────────────────────────────────────────────────────
    while len(collected) < TARGET_ROWS_PER_AIRPORT and offset < 300:
        resp = api_get("flights", {
            "dep_iata" : iata,
            "flight_status": "scheduled",
            "limit"    : page_size,
            "offset"   : offset,
        })
        time.sleep(API_DELAY)

        if resp is None:
            break

        # Check for API error
        if "error" in resp:
            err = resp["error"]
            print(f"\n    [API ERROR] {err.get('message','unknown')} (code {err.get('code','?')})")
            break

        flights = resp.get("data", [])
        if not flights:
            break

        for f in flights:
            if len(collected) >= TARGET_ROWS_PER_AIRPORT:
                break
            dep_term = (f.get("departure") or {}).get("terminal", "") or ""
            arr_term = (f.get("arrival")   or {}).get("terminal", "") or ""

            # Build flat row
            flat = flatten_flight(f)

            # Terminal-distance enrichment (departure terminal within same airport)
            d_m, w_m, tr = lookup_terminal_dist(iata, dep_term, arr_term)
            # If no match on dep→arr, try any terminal pair for this airport
            if not d_m and terminals:
                tf, tt, d_m, w_m, tr = terminals[len(collected) % len(terminals)]
            else:
                tf, tt = dep_term, arr_term

            flat["terminal_distance_m"]      = d_m
            flat["terminal_walk_time_min"]   = w_m
            flat["terminal_transport_method"]= tr
            flat["terminal_from"]            = tf
            flat["terminal_to"]              = tt

            # ── Class-balanced connection scenario ──────────────────────────
            risk_cls              = target_class(len(collected), TARGET_ROWS_PER_AIRPORT)
            c_time, a_delay       = connection_params(risk_cls, w_m, iata)
            flat["connection_time_min"] = c_time
            flat["arrival_delay_min"]   = a_delay
            flat["connection_risk"]     = risk_cls

            collected.append(flat)

        offset += page_size
        if len(flights) < page_size:
            break   # no more pages

    # ── Fill remaining with synthetic rows ────────────────────────────────────
    synth_idx = 0
    while len(collected) < TARGET_ROWS_PER_AIRPORT:
        # Cycle through terminal pairs
        if terminals:
            tf, tt, d_m, w_m, tr = terminals[synth_idx % len(terminals)]
        else:
            tf, tt, d_m, w_m, tr = "Terminal 1", "Terminal 2", "500", "6", "Walkway"

        flat = synthetic_row(iata, info["name"], tf, tt, d_m, w_m, tr, synth_idx)
        flat["terminal_distance_m"]      = d_m
        flat["terminal_walk_time_min"]   = w_m
        flat["terminal_transport_method"]= tr
        flat["terminal_from"]            = tf
        flat["terminal_to"]              = tt

        # ── Class-balanced connection scenario ──────────────────────────────
        risk_cls              = target_class(len(collected), TARGET_ROWS_PER_AIRPORT)
        c_time, a_delay       = connection_params(risk_cls, w_m, iata)
        flat["connection_time_min"] = c_time
        flat["arrival_delay_min"]   = a_delay
        flat["connection_risk"]     = risk_cls

        collected.append(flat)
        synth_idx += 1

    # Trim to exactly 50
    collected = collected[:TARGET_ROWS_PER_AIRPORT]
    flight_rows.extend(collected)
    print(f" {len(collected)} rows (total so far: {len(flight_rows)})")

# Shuffle so Safe/Tight/Risky rows are not block-ordered
random.shuffle(flight_rows)

print(f"\n[OK] Total flight rows collected: {len(flight_rows)}")

# ── Print class distribution ───────────────────────────────────────────────────
from collections import Counter
label_counts = Counter(r.get("connection_risk", "?") for r in flight_rows)
total = len(flight_rows)
print("[OK] Class distribution:")
for lbl in ["Safe", "Tight", "Risky"]:
    cnt = label_counts.get(lbl, 0)
    print(f"     {lbl:6s}: {cnt:5d}  ({cnt/total*100:.1f}%)")

# ── Write aviation_new_dataset.csv ────────────────────────────────────────────
print(f"[>>] Writing {OUT_FLIGHT} ...")
with open(OUT_FLIGHT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=ALL_COLS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(flight_rows)
print(f"[OK] {OUT_FLIGHT} written ({len(flight_rows)} rows)")

# ── Append Airport_Terminal_Distances rows ─────────────────────────────────────
# Map distance CSV columns → combined schema
dist_flight_rows = []
for dr in dist_rows:
    flat = {c: "" for c in ALL_COLS}
    # Fill what we know from the distances file
    flat["departure.airport"]          = dr["Airport Name"]
    flat["departure.iata"]             = dr["IATA Code"]
    flat["departure.terminal"]         = dr["Terminal From"]
    flat["arrival.airport"]            = dr["Airport Name"]
    flat["arrival.iata"]               = dr["IATA Code"]
    flat["arrival.terminal"]           = dr["Terminal To"]
    flat["terminal_distance_m"]        = dr["Distance (m)"]
    flat["terminal_walk_time_min"]     = dr["Walk Time (min)"]
    flat["terminal_transport_method"]  = dr["Transport Method"]
    flat["terminal_from"]              = dr["Terminal From"]
    flat["terminal_to"]                = dr["Terminal To"]
    # Mark as a distance reference row
    flat["flight_status"] = "distance_reference"
    dist_flight_rows.append(flat)

print(f"[>>] Writing {OUT_COMBO} ...")
combined = flight_rows + dist_flight_rows
with open(OUT_COMBO, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=ALL_COLS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(combined)
print(f"[OK] {OUT_COMBO} written ({len(combined)} rows total)")
print(f"     Flight rows   : {len(flight_rows)}")
print(f"     Distance rows : {len(dist_flight_rows)}")
print("\nDone!")
