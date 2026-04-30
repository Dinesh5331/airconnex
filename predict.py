"""
predict.py
---------------------------------------------------------------------------
Manual Flight Connection Feasibility Predictor
---------------------------------------------------------------------------
Enter your connection details manually:
  - layover airport IATA code
  - arrival terminal and time
  - departure terminal and time
  - arrival delay and whether the connection is international

The script computes the transfer features and uses the Random Forest model
from random_forest.py to predict whether the connection is possible.
---------------------------------------------------------------------------
"""

import os
import re
from datetime import date, datetime, timedelta

import pandas as pd

from random_forest import train_random_forest


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SEP = "=" * 65
RISK_LABELS = {
    0: "RISKY",
    1: "TIGHT",
    2: "SAFE",
}


def resolve_existing_path(*names):
    for name in names:
        path = os.path.join(BASE_DIR, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Could not find any of: {', '.join(names)}")


TRAINING_PATH = resolve_existing_path("preprocessing1.csv")
RAW_FEATURE_PATH = resolve_existing_path("cleaned_dataset_before_encoding.csv")
TERMINAL_DISTANCE_PATH = resolve_existing_path("Airport_Terminal_Distances.csv")


def normalize_lookup_value(value):
    if value is None:
        return "UNKNOWN"
    text = re.sub(r"\s+", " ", str(value).strip().upper())
    return text or "UNKNOWN"


def compact_token(value):
    return re.sub(r"[^A-Z0-9]", "", normalize_lookup_value(value))


def build_category_map(raw_df, enc_df, column_name):
    pairs = pd.DataFrame(
        {
            "raw": raw_df[column_name].fillna("Unknown").map(normalize_lookup_value),
            "enc": enc_df[column_name],
        }
    )
    return pairs.groupby("raw")["enc"].agg(lambda series: series.mode().iat[0]).to_dict()


def load_support_data():
    raw_df = pd.read_csv(RAW_FEATURE_PATH)
    enc_df = pd.read_csv(TRAINING_PATH)

    if len(raw_df) != len(enc_df):
        raise ValueError("Support datasets do not align row-for-row.")

    category_maps = {}
    for column_name in [
        "departure.iata",
        "departure.terminal",
        "arrival.terminal",
        "terminal_transport_method",
        "security_crowd",
        "immigration_crowd",
    ]:
        category_maps[column_name] = build_category_map(raw_df, enc_df, column_name)

    airport_flags = (
        raw_df.groupby("departure.iata")[["is_intl_hub", "is_major_airport"]]
        .max()
        .reset_index()
    )
    airport_flag_map = {
        normalize_lookup_value(row["departure.iata"]): {
            "is_intl_hub": int(row["is_intl_hub"]),
            "is_major_airport": int(row["is_major_airport"]),
        }
        for _, row in airport_flags.iterrows()
    }
    return category_maps, airport_flag_map


def load_terminal_reference():
    df = pd.read_csv(TERMINAL_DISTANCE_PATH)
    df.columns = [column.strip() for column in df.columns]

    for column_name in [
        "Airport Name",
        "IATA Code",
        "City",
        "Country",
        "Terminal From",
        "Terminal To",
        "Transport Method",
    ]:
        df[column_name] = df[column_name].fillna("").astype(str).str.strip()

    df["IATA Code"] = df["IATA Code"].str.upper()
    df["Distance (m)"] = pd.to_numeric(df["Distance (m)"], errors="coerce")
    df["Walk Time (min)"] = pd.to_numeric(df["Walk Time (min)"], errors="coerce")
    return df


def terminal_aliases(name):
    aliases = {normalize_lookup_value(name), compact_token(name)}
    normalized = normalize_lookup_value(name)
    directional_tokens = {"NORTH": "N", "SOUTH": "S", "EAST": "E", "WEST": "W"}

    for prefix, pattern in [
        ("T", r"\bTERMINAL\s+([A-Z0-9]+)\b"),
        ("C", r"\bCONCOURSE\s+([A-Z0-9]+)\b"),
        ("P", r"\bPIER\s+([A-Z0-9]+)\b"),
    ]:
        match = re.search(pattern, normalized)
        if match:
            token = match.group(1)
            aliases.add(token)
            aliases.add(f"{prefix}{token}")
            aliases.add(f"{prefix}_{token}")
            if token in directional_tokens:
                aliases.add(directional_tokens[token])

    return {alias for alias in aliases if alias}


def resolve_terminal_input(user_input, available_terminals):
    normalized_input = normalize_lookup_value(user_input)

    for terminal in available_terminals:
        if normalized_input == normalize_lookup_value(terminal):
            return terminal

    input_aliases = terminal_aliases(user_input)
    alias_matches = [
        terminal
        for terminal in available_terminals
        if input_aliases.intersection(terminal_aliases(terminal))
    ]
    if len(alias_matches) == 1:
        return alias_matches[0]
    if alias_matches:
        return sorted(alias_matches, key=len)[0]

    compact_input = compact_token(user_input)
    if compact_input and len(compact_input) >= 2:
        partial_matches = [
            terminal
            for terminal in available_terminals
            if compact_input in compact_token(terminal)
        ]
        if len(partial_matches) == 1:
            return partial_matches[0]
        if partial_matches:
            return sorted(partial_matches, key=len)[0]

    return None


def extract_terminal_code(value):
    normalized = normalize_lookup_value(value)
    for pattern in [
        r"\bTERMINAL\s+([A-Z0-9]+)\b",
        r"\bCONCOURSE\s+([A-Z0-9]+)\b",
        r"\bPIER\s+([A-Z0-9]+)\b",
    ]:
        match = re.search(pattern, normalized)
        if match:
            return match.group(1)

    compact_value = compact_token(value)
    if compact_value in {"UNKNOWN", "ALL"}:
        return compact_value
    if 1 <= len(compact_value) <= 3:
        return compact_value
    return None


def default_terminal_input(available_terminals, index=0):
    if not available_terminals:
        return "T1"
    choice = available_terminals[min(index, len(available_terminals) - 1)]
    short_code = extract_terminal_code(choice)
    if not short_code:
        return choice
    if (
        normalize_lookup_value(choice).startswith("TERMINAL ")
        and short_code not in {"NORTH", "SOUTH", "UNKNOWN", "ALL"}
    ):
        return f"T{short_code}"
    return short_code


def format_terminal_options(available_terminals):
    return ", ".join(available_terminals)


def lookup_transfer_details(airport_rows, arr_terminal_name, dep_terminal_name):
    if arr_terminal_name == dep_terminal_name:
        return {
            "distance_m": 0,
            "walk_min": 0,
            "transport": "Same terminal",
            "source": "same-terminal",
        }

    pair_rows = airport_rows[
        (
            (airport_rows["Terminal From"] == arr_terminal_name)
            & (airport_rows["Terminal To"] == dep_terminal_name)
        )
        | (
            (airport_rows["Terminal From"] == dep_terminal_name)
            & (airport_rows["Terminal To"] == arr_terminal_name)
        )
    ]

    if not pair_rows.empty:
        best_row = pair_rows.sort_values(["Distance (m)", "Walk Time (min)"]).iloc[0]
        return {
            "distance_m": int(round(best_row["Distance (m)"])),
            "walk_min": int(round(best_row["Walk Time (min)"])),
            "transport": best_row["Transport Method"],
            "source": "exact-pair",
        }

    fallback_row = airport_rows.sort_values(["Distance (m)", "Walk Time (min)"]).iloc[0]
    return {
        "distance_m": int(round(airport_rows["Distance (m)"].mean())),
        "walk_min": int(round(airport_rows["Walk Time (min)"].mean())),
        "transport": fallback_row["Transport Method"],
        "source": "airport-average",
    }


def ask_str(prompt, valid=None, default=None):
    while True:
        hint = f"  (default={default})" if default not in (None, "") else ""
        raw = input(f"  {prompt}{hint}: ").strip().upper()
        if raw == "" and default is not None:
            return str(default).upper()
        if valid is None or raw in valid:
            return raw
        print(f"    [!] Valid options: {', '.join(sorted(valid))}")


def ask_time(prompt, default=None):
    while True:
        hint = f"  (default={default})" if default else ""
        raw = input(f"  {prompt} [HH:MM]{hint}: ").strip()
        if raw == "" and default is not None:
            raw = default
        try:
            return datetime.strptime(raw, "%H:%M").time()
        except ValueError:
            print("    [!] Please enter time in HH:MM format (e.g. 14:35)")


def ask_int(prompt, lo, hi, default=None):
    while True:
        hint = f" [{lo}-{hi}]"
        if default is not None:
            hint += f"  (default={default})"
        raw = input(f"  {prompt}{hint}: ").strip()
        if raw == "" and default is not None:
            return int(default)
        try:
            value = int(raw)
        except ValueError:
            print("    [!] Please enter a whole number.")
            continue
        if lo <= value <= hi:
            return value
        print(f"    [!] Enter a value between {lo} and {hi}.")


def ask_terminal(prompt, available_terminals, default=None):
    while True:
        hint = f"  (default={default})" if default else ""
        raw = input(f"  {prompt}{hint}: ").strip()
        if raw == "" and default is not None:
            raw = default

        resolved = resolve_terminal_input(raw, available_terminals)
        if resolved:
            return raw.upper(), resolved

        print("    [!] Terminal not found in the airport reference.")
        print(f"    [!] Available options: {format_terminal_options(available_terminals)}")


def lookup_encoded_value(mapping, raw_value, fallback):
    key = normalize_lookup_value(raw_value)
    return mapping.get(key, fallback)


def encode_terminal_value(mapping, raw_terminal_input, canonical_terminal_name, fallback):
    candidates = []
    short_code = extract_terminal_code(canonical_terminal_name)
    if short_code:
        candidates.append(short_code)
    candidates.append(raw_terminal_input)
    candidates.append(canonical_terminal_name)

    for candidate in candidates:
        encoded = mapping.get(normalize_lookup_value(candidate))
        if encoded is not None:
            return encoded
    return fallback


def transport_friction_score(value):
    transport_map = {
        "WALKWAY": 1,
        "WALKWAY BRIDGE": 1,
        "MOVING WALKWAY": 1,
        "AIRSIDE CONNECTOR": 2,
        "SKYTRAIN": 2,
        "MONORAIL": 2,
        "AUTOMATED PEOPLE MOVER": 2,
        "TRAIN": 2,
        "AIRTRAIN": 2,
        "LINK TRAIN": 2,
        "SHUTTLE": 3,
        "BUS": 3,
        "METRO": 3,
    }
    normalized = normalize_lookup_value(value)
    for token, score in transport_map.items():
        if token in normalized:
            return score
    return 3


def base_crowd_score(hour_value):
    if 6 <= hour_value <= 10 or 17 <= hour_value <= 21:
        return 2
    if 10 < hour_value < 17:
        return 1
    return 0


def derive_crowd_label(departure_hour, day_of_week, is_major_airport, crowd_type):
    score = base_crowd_score(departure_hour)

    if is_major_airport:
        score += 0.5
    if day_of_week in {4, 5, 6}:
        score += 0.5
    if crowd_type == "immigration":
        score += 0.3

    if score < 0.75:
        return "low"
    if score < 1.75:
        return "medium"
    return "high"


def crowd_multiplier(level):
    return {"low": 0.8, "medium": 1.0, "high": 1.5}[level]


def possible_probability(classes, probabilities):
    total = 0.0
    for class_id, probability in zip(classes, probabilities):
        if int(class_id) in {1, 2}:
            total += probability
    return total


terminal_reference_df = load_terminal_reference()
category_maps, airport_flag_map = load_support_data()


print(f"\n{SEP}")
print("  Loading data & training Random Forest from random_forest.py ...")
print(SEP)

model_result = train_random_forest(TRAINING_PATH)
rf = model_result["model"]
training_df = model_result["dataframe"]
X = training_df.drop("risk_label", axis=1)
acc = model_result["accuracy"]

print(f"  [OK] Model trained  |  Test accuracy: {acc:.4f} ({acc * 100:.2f}%)")
print(f"  [OK] Training on {len(model_result['X_train']):,} rows | {X.shape[1]} features")

MEDIANS = X.median().to_dict()
SUPPORTED_AIRPORTS = sorted(terminal_reference_df["IATA Code"].unique())


print(f"\n{SEP}")
print("  FLIGHT CONNECTION FEASIBILITY CHECKER")
print(SEP)
print(f"\n  Supported airports: {', '.join(SUPPORTED_AIRPORTS)}\n")

print("-- STEP 1: Connecting Airport --")
apt_code = ask_str(
    "  Layover airport IATA code (e.g. DXB, LHR, SIN)",
    valid=set(SUPPORTED_AIRPORTS),
    default="DXB",
)

airport_rows = terminal_reference_df[terminal_reference_df["IATA Code"] == apt_code].copy()
airport_name = airport_rows["Airport Name"].mode().iat[0]
airport_terminals = sorted(
    set(airport_rows["Terminal From"]).union(set(airport_rows["Terminal To"]))
)
airport_flags = airport_flag_map.get(
    normalize_lookup_value(apt_code),
    {
        "is_intl_hub": int(round(MEDIANS["is_intl_hub"])),
        "is_major_airport": int(round(MEDIANS["is_major_airport"])),
    },
)

print(f"    --> {apt_code}  |  {airport_name}")
print(f"    --> Terminal reference: {format_terminal_options(airport_terminals)}")

print("\n-- STEP 2: Arriving Flight Details --")
arr_default = default_terminal_input(airport_terminals, index=0)
arr_terminal_input, arr_terminal_name = ask_terminal(
    "  Arriving at terminal",
    airport_terminals,
    default=arr_default,
)
arr_time_value = ask_time("  Scheduled arrival time", default="12:00")
arr_delay = ask_int("  Actual arrival delay (minutes, 0 if on time)", 0, 45, default=0)

print("\n-- STEP 3: Connecting Flight Details --")
dep_default = default_terminal_input(
    airport_terminals,
    index=1 if len(airport_terminals) > 1 else 0,
)
dep_terminal_input, dep_terminal_name = ask_terminal(
    "  Connecting flight departs from terminal",
    airport_terminals,
    default=dep_default,
)
layover_days = ask_int(
    "  Days after arrival when the connection departs (0=same day, 1=next day)",
    0,
    14,
    default=0,
)
dep_time_value = ask_time("  Connecting flight departure time", default="14:30")

print("\n-- STEP 4: Extra Flight Info --")
sched_flight_min = ask_int("  Duration of the ARRIVING flight (minutes)", 30, 1710, default=180)
is_international = ask_str(
    "  Is this an INTERNATIONAL connection? (Y/N)",
    {"Y", "N"},
    default="N",
)


print(f"\n{SEP}")
print("  COMPUTING FEATURES ...")
print(SEP)

base_date = date.today()
arr_dt = datetime.combine(base_date, arr_time_value)
dep_dt = datetime.combine(base_date + timedelta(days=layover_days), dep_time_value)
if dep_dt < arr_dt:
    dep_dt += timedelta(days=1)

conn_min = int(round((dep_dt - arr_dt).total_seconds() / 60))
actual_layover_days = (dep_dt.date() - arr_dt.date()).days
actual_arrival_dt = arr_dt + timedelta(minutes=arr_delay)

dep_hour = dep_dt.hour
day_of_week = dep_dt.weekday()
is_peak_hour = 1 if (6 <= dep_hour <= 10 or 17 <= dep_hour <= 21) else 0
cong_time = 15 if is_peak_hour else (8 if 10 < dep_hour < 17 else 3)

transfer = lookup_transfer_details(airport_rows, arr_terminal_name, dep_terminal_name)
dist_m = transfer["distance_m"]
walk_min = transfer["walk_min"]
transport = transfer["transport"]
transport_source = transfer["source"]

terminal_change = 0 if arr_terminal_name == dep_terminal_name else 1
friction = transport_friction_score(transport)
is_intl_hub = airport_flags["is_intl_hub"]
is_major = airport_flags["is_major_airport"]

sec_crowd_label = derive_crowd_label(dep_hour, day_of_week, is_major, "security")
if is_international == "Y":
    imm_crowd_label = derive_crowd_label(dep_hour, day_of_week, is_major, "immigration")
else:
    imm_crowd_label = "low"

sec_time = int(round(20 * crowd_multiplier(sec_crowd_label)))
imm_time = int(round(30 * crowd_multiplier(imm_crowd_label))) if is_international == "Y" else 0
bag_time = 10
req_time = walk_min + sec_time + imm_time + cong_time + bag_time
time_margin = conn_min - arr_delay - req_time

print(f"\n  Connection window     : {conn_min} min")
print(f"  Layover days          : {actual_layover_days}")
print(f"  Arrival delay         : {arr_delay} min")
print(f"  Walking time          : {walk_min} min")
print(f"  Security time         : {sec_time} min  (auto crowd={sec_crowd_label})")
print(
    f"  Immigration time      : {imm_time} min  "
    f"(international={'yes' if is_international == 'Y' else 'no'}, auto crowd={imm_crowd_label})"
)
print(
    f"  Congestion time       : {cong_time} min  "
    f"({'peak' if is_peak_hour else 'off-peak'} hour {dep_hour:02d}:00)"
)
print(f"  Baggage time          : {bag_time} min")
print("  ------------------------")
print(f"  Total required time   : {req_time} min")
print(f"  Time margin left      : {time_margin} min")
print(
    f"  Terminal change       : "
    f"{'YES (different terminals)' if terminal_change else 'NO (same terminal)'}"
)
print(f"  Transfer reference    : {transport_source}")
print(f"  Transport method      : {transport}  (friction={friction})")
print(f"  International hub     : {'YES' if is_intl_hub else 'NO'}")


inp = {column: MEDIANS[column] for column in X.columns}
inp["departure.iata"] = lookup_encoded_value(
    category_maps["departure.iata"],
    apt_code,
    MEDIANS["departure.iata"],
)
inp["departure.terminal"] = encode_terminal_value(
    category_maps["departure.terminal"],
    dep_terminal_input,
    dep_terminal_name,
    MEDIANS["departure.terminal"],
)
inp["arrival.terminal"] = encode_terminal_value(
    category_maps["arrival.terminal"],
    arr_terminal_input,
    arr_terminal_name,
    MEDIANS["arrival.terminal"],
)
inp["terminal_distance_m"] = dist_m
inp["terminal_walk_time_min"] = walk_min
inp["terminal_transport_method"] = lookup_encoded_value(
    category_maps["terminal_transport_method"],
    transport,
    MEDIANS["terminal_transport_method"],
)
inp["connection_time_min"] = conn_min
inp["arrival_delay_min"] = arr_delay
inp["scheduled_flight_min"] = sched_flight_min
inp["departure_hour"] = dep_hour
inp["day_of_week"] = day_of_week
inp["transport_friction"] = friction
inp["is_cross_airport"] = 1
inp["is_intl_hub"] = is_intl_hub
inp["is_major_airport"] = is_major
inp["is_peak_hour"] = is_peak_hour
inp["terminal_change"] = terminal_change
inp["has_delay_info"] = 1
inp["security_crowd"] = lookup_encoded_value(
    category_maps["security_crowd"],
    sec_crowd_label,
    MEDIANS["security_crowd"],
)
inp["immigration_crowd"] = lookup_encoded_value(
    category_maps["immigration_crowd"],
    imm_crowd_label,
    MEDIANS["immigration_crowd"],
)
inp["walking_time_min"] = walk_min
inp["security_time_min"] = sec_time
inp["immigration_time_min"] = imm_time
inp["congestion_time_min"] = cong_time
inp["baggage_time_min"] = bag_time


row = pd.DataFrame([inp], columns=X.columns)
pred = int(rf.predict(row)[0])
prob = rf.predict_proba(row)[0]
possible_prob = possible_probability(rf.classes_, prob)
is_possible = pred in {1, 2}


print(f"\n{SEP}")
print("  PREDICTION RESULT")
print(SEP)
print(
    f"\n  Arriving  : {arr_terminal_name} at {arr_dt.strftime('%H:%M')}  "
    f"(+{arr_delay} min delay => actual {actual_arrival_dt.strftime('%H:%M')})"
)
print(
    f"  Departing : {dep_terminal_name} at {dep_dt.strftime('%H:%M')}  "
    f"({actual_layover_days} day(s) after arrival)"
)
print(f"  Airport   : {apt_code}  |  {airport_name}")
print(f"\n  Feasibility: {'POSSIBLE' if is_possible else 'NOT POSSIBLE'}")
print(f"  Risk class : {RISK_LABELS[pred]}")
print(f"  Possible probability     : {possible_prob * 100:5.1f}%")
print(f"  Not possible probability : {(1 - possible_prob) * 100:5.1f}%")

print("\n  Class breakdown:")
for class_id, probability in zip(rf.classes_, prob):
    bar = "|" * int(probability * 40)
    print(f"    {RISK_LABELS[int(class_id)]:5s}  {probability * 100:5.1f}%  {bar}")


print(f"\n{SEP}")
print("  INTERPRETATION")
print(SEP)
if pred == 2:
    print(f"\n  This connection looks possible with a healthy buffer of about {time_margin} minutes.")
elif pred == 1:
    print(f"\n  This connection is possible, but tight. Estimated remaining margin is about {time_margin} minutes.")
else:
    print(f"\n  This connection is not likely to work. Estimated remaining margin is about {time_margin} minutes.")

print(f"\n{SEP}\n")
