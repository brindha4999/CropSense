import os
import numpy as np
import pandas as pd
import calendar
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    cohen_kappa_score, confusion_matrix, classification_report,
    recall_score
)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Load crop recommendation dataset 
crop = pd.read_csv(os.path.join(BASE_DIR, 'Crop_recommendation.xls'))

#Encode labels 
le = LabelEncoder()
crop['label_encoded'] = le.fit_transform(crop['label'])

X = crop[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = crop['label_encoded']

#Train / test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Scale features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)

#Train Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#Compute all metrics 
acc   = accuracy_score(y_test, y_pred)
err   = 1 - acc
f1    = f1_score(y_test, y_pred, average='weighted')
mcc   = matthews_corrcoef(y_test, y_pred)
kappa = cohen_kappa_score(y_test, y_pred)

recalls = recall_score(y_test, y_pred, average=None)
gmean   = float(np.prod(recalls) ** (1 / len(recalls)))

cm = confusion_matrix(y_test, y_pred)
report = classification_report(
    y_test, y_pred,
    target_names=le.classes_,
    output_dict=True
)

print(f"✅ Model trained and ready!")
print(f"   Accuracy   : {acc:.4f}")
print(f"   Error Rate : {err:.4f}")
print(f"   F1 Score   : {f1:.4f}")
print(f"   MCC        : {mcc:.4f}")
print(f"   Kappa      : {kappa:.4f}")
print(f"   G-Mean     : {gmean:.4f}")
#PHASE 1 — Load Fertilizer Data Files 

#Load crop NPK requirements
crop_requirements = {}
REQ_PATH = os.path.join(BASE_DIR, 'crop_requirements.csv')
if os.path.exists(REQ_PATH):
    req_df = pd.read_csv(REQ_PATH)
    req_df['crop'] = req_df['crop'].str.lower().str.strip()
    for _, row in req_df.iterrows():
        crop_requirements[row['crop']] = {
            'N': float(row['N_ideal']),
            'P': float(row['P_ideal']),
            'K': float(row['K_ideal'])
        }
    print(f"🌱 Loaded crop requirements: {len(crop_requirements)} crops.")
else:
    print("⚠️  crop_requirements.csv not found.")

#Load fertilizer mapping
fertilizer_map = {}
FERT_PATH = os.path.join(BASE_DIR, 'fertilizer_data.csv')
if os.path.exists(FERT_PATH):
    fert_df = pd.read_csv(FERT_PATH)
    for _, row in fert_df.iterrows():
        fertilizer_map[row['nutrient']] = {
            'name':            row['fertilizer_name'],
            'nutrient_percent': float(row['nutrient_percent']),
            'price_per_kg':    float(row['price_per_kg_inr'])
        }
    print(f"💊 Loaded fertilizer data: {list(fertilizer_map.keys())}")
else:
    print("⚠️  fertilizer_data.csv not found.")

#Load crop growth stages
crop_stages = {}
STAGES_PATH = os.path.join(BASE_DIR, 'crop_stages.csv')
if os.path.exists(STAGES_PATH):
    stages_df = pd.read_csv(STAGES_PATH)
    stages_df['crop'] = stages_df['crop'].str.lower().str.strip()
    for crop_name, group in stages_df.groupby('crop'):
        crop_stages[crop_name] = group.sort_values('stage_number').to_dict('records')
    print(f"📋 Loaded crop stages: {len(crop_stages)} crops.")
else:
    print("⚠️  crop_stages.csv not found.")

#Absorption efficiency by soil type 
ABSORPTION = {
    'Sandy':     {'N': 0.55, 'P': 0.25, 'K': 0.50},
    'Loamy':     {'N': 0.70, 'P': 0.35, 'K': 0.65},
    'Clay':      {'N': 0.60, 'P': 0.30, 'K': 0.55},
    'Sandy Loam':{'N': 0.65, 'P': 0.30, 'K': 0.60},
    'Clay Loam': {'N': 0.65, 'P': 0.32, 'K': 0.60},
    'Silty Loam':{'N': 0.68, 'P': 0.33, 'K': 0.62},
}

#PHASE 2 — Fertilizer Engine Functions

def calculate_deficiency(soil, ideal):
    """
    Compare measured soil NPK against ideal crop requirements.
    Returns positive deficit (0 if soil already meets requirement).
    """
    return {
        'N': round(max(0.0, ideal['N'] - soil['N']), 2),
        'P': round(max(0.0, ideal['P'] - soil['P']), 2),
        'K': round(max(0.0, ideal['K'] - soil['K']), 2)
    }


def calculate_fertilizer_amounts(deficiency, soil_type='Loamy'):
    """
    Convert nutrient deficit (kg/ha) to fertilizer quantity (kg/ha).
    Formula: fertilizer_kg = deficit / (nutrient_percent × absorption_efficiency)
    Absorption efficiency accounts for soil type nutrient retention.
    """
    absorption = ABSORPTION.get(soil_type, ABSORPTION['Loamy'])
    plan = {}
    for nutrient, deficit in deficiency.items():
        if deficit > 0 and nutrient in fertilizer_map:
            info   = fertilizer_map[nutrient]
            eff    = absorption.get(nutrient, 0.65)
            amount = round(deficit / (info['nutrient_percent'] * eff), 2)
            plan[nutrient] = {
                'fertilizer':  info['name'],
                'amount_kg':   amount,
                'deficit_kg':  deficit,
                'efficiency':  round(eff * 100, 0)
            }
        else:
            plan[nutrient] = {
                'fertilizer':  fertilizer_map.get(nutrient, {}).get('name', '—'),
                'amount_kg':   0,
                'deficit_kg':  0,
                'efficiency':  round(ABSORPTION.get(soil_type, ABSORPTION['Loamy']).get(nutrient, 0.65) * 100, 0)
            }
    return plan


def estimate_cost(fert_plan):
    """
    Calculate total fertilizer cost in INR based on quantities.
    """
    total = 0.0
    breakdown = {}
    for nutrient, details in fert_plan.items():
        if details['amount_kg'] > 0 and nutrient in fertilizer_map:
            price    = fertilizer_map[nutrient]['price_per_kg']
            cost     = round(details['amount_kg'] * price, 2)
            total   += cost
            breakdown[details['fertilizer']] = {
                'amount_kg': details['amount_kg'],
                'cost_inr':  cost
            }
    return round(total, 2), breakdown


def soil_match_score(soil, ideal):
    """How well the soil matches the crop's ideal NPK profile (%)."""
    total_ideal   = ideal['N'] + ideal['P'] + ideal['K']
    total_deficit = (
        max(0, ideal['N'] - soil['N']) +
        max(0, ideal['P'] - soil['P']) +
        max(0, ideal['K'] - soil['K'])
    )
    if total_ideal == 0:
        return 100.0
    match = max(0, 1 - (total_deficit / total_ideal)) * 100
    return round(match, 1)


def build_fertilizer_plan(crop_name, soil, soil_type='Loamy'):
    """
    Full pipeline for one crop:
    deficiency → fertilizer amounts → cost
    """
    key = crop_name.lower().strip()
    if key not in crop_requirements:
        return None

    ideal      = crop_requirements[key]
    deficiency = calculate_deficiency(soil, ideal)
    fert_plan  = calculate_fertilizer_amounts(deficiency, soil_type)
    total_cost, breakdown = estimate_cost(fert_plan)

    summary = []
    for nutrient, details in fert_plan.items():
        if details['amount_kg'] > 0:
            summary.append(
                f"{details['fertilizer']}: {details['amount_kg']} kg/ha"
            )

    return {
        'crop':        crop_name,
        'ideal_npk':   ideal,
        'deficiency':  deficiency,
        'fert_plan':   fert_plan,
        'breakdown':   breakdown,
        'total_cost':  total_cost,
        'soil_type':   soil_type,
        'summary':     summary if summary else ['No fertilizer needed — soil nutrients are sufficient.']
    }


#Load crop stages
crop_stages = {}
STAGES_PATH = os.path.join(BASE_DIR, 'crop_stages.csv')
if os.path.exists(STAGES_PATH):
    stages_df = pd.read_csv(STAGES_PATH)
    stages_df['crop'] = stages_df['crop'].str.lower().str.strip()
    for crop_key, group in stages_df.groupby('crop'):
        crop_stages[crop_key] = group.sort_values('stage_number').to_dict('records')
    print(f"📋 Loaded crop stages: {len(crop_stages)} crops.")
else:
    print("⚠️  crop_stages.csv not found.")


def generate_schedule(crop_name, fert_plan, sow_month_num, sow_year):
    """
    Build a stage-by-stage application schedule.
    For each stage, calculates kg/ha of each fertilizer to apply.
    """
    key    = crop_name.lower().strip()
    stages = crop_stages.get(key, [])
    if not stages:
        return []

    schedule = []
    for s in stages:
        stage_entry = {
            'stage':             s['stage_name'],
            'days_after_sowing': int(s['days_from_sowing']),
            'notes':             s.get('notes', ''),
            'application':       [],
            'application_month': ''
        }

        # Calculate calendar month for this stage
        days             = int(s['days_from_sowing'])
        month_offset     = days // 30
        stage_month_num  = (sow_month_num - 1 + month_offset) % 12 + 1
        stage_year_offset = (sow_month_num - 1 + month_offset) // 12
        stage_year        = sow_year + stage_year_offset
        stage_entry['application_month'] = (
            f"{calendar.month_abbr[stage_month_num]} {stage_year}"
        )

        # Calculate amounts for each nutrient at this stage
        for nutrient in ['N', 'P', 'K']:
            pct_col  = f"{nutrient}_percent"
            pct      = float(s.get(pct_col, 0)) / 100.0
            details  = fert_plan.get(nutrient, {})
            total_kg = details.get('amount_kg', 0)
            stage_kg = round(total_kg * pct, 2)

            if stage_kg > 0:
                stage_entry['application'].append({
                    'nutrient':   nutrient,
                    'fertilizer': details.get('fertilizer', '—'),
                    'amount_kg':  stage_kg,
                })

        schedule.append(stage_entry)

    return schedule
FALLBACK_CALENDAR = {
    "rice":        {"duration": 5,  "suitable_months": [6,7,8,9,10,11]},
    "maize":       {"duration": 4,  "suitable_months": [6,7,8,9]},
    "chickpea":    {"duration": 4,  "suitable_months": [10,11,12,1,2]},
    "kidneybeans": {"duration": 4,  "suitable_months": [6,7,8,9]},
    "pigeonpeas":  {"duration": 6,  "suitable_months": [6,7,8,9,10,11,12]},
    "mothbeans":   {"duration": 4,  "suitable_months": [7,8,9,10]},
    "mungbean":    {"duration": 3,  "suitable_months": [3,4,5,6]},
    "blackgram":   {"duration": 4,  "suitable_months": [6,7,8,9]},
    "lentil":      {"duration": 4,  "suitable_months": [10,11,12,1,2]},
    "pomegranate": {"duration": 7,  "suitable_months": [7,8,9,10,11,12,1]},
    "banana":      {"duration": 12, "suitable_months": list(range(1,13))},
    "mango":       {"duration": 6,  "suitable_months": [6,7,8,9,10,11,12]},
    "grapes":      {"duration": 7,  "suitable_months": [1,2,3,4,5,6,7,8]},
    "watermelon":  {"duration": 3,  "suitable_months": [1,2,3,4]},
    "muskmelon":   {"duration": 4,  "suitable_months": [2,3,4,5]},
    "apple":       {"duration": 6,  "suitable_months": [3,4,5,6,7,8,9]},
    "orange":      {"duration": 7,  "suitable_months": [7,8,9,10,11,12,1]},
    "papaya":      {"duration": 8,  "suitable_months": list(range(1,13))},
    "coconut":     {"duration": 12, "suitable_months": list(range(1,13))},
    "cotton":      {"duration": 7,  "suitable_months": [4,5,6,7,8,9,10]},
    "jute":        {"duration": 5,  "suitable_months": [3,4,5,6,7]},
    "coffee":      {"duration": 8,  "suitable_months": [6,7,8,9,10,11,12]},
    "wheat":       {"duration": 6,  "suitable_months": [10,11,12,1,2]},
}

crop_calendar = {}
CALENDAR_PATH = os.path.join(BASE_DIR, 'crop_growth_calendar.xls')

if os.path.exists(CALENDAR_PATH):
    try:
        cal_df = pd.read_excel(CALENDAR_PATH, engine="openpyxl")
        cal_df.columns = [c.strip() for c in cal_df.columns]
        cal_df['Crop'] = cal_df['Crop'].str.lower().str.strip()
        for _, row in cal_df.iterrows():
            suitable = [int(m.strip()) for m in str(row['Suitable_Months']).split(',')]
            crop_calendar[row['Crop']] = {
                'duration': int(row['Growth_Duration_Months']),
                'suitable_months': suitable
            }
        print(f"📅 Loaded growth calendar: {len(crop_calendar)} crops.")
    except Exception as e:
        print(f"⚠️  Calendar read failed: {e} — using fallback.")
else:
    print("⚠️  crop_growth_calendar.xls not found — using fallback.")

for k, v in FALLBACK_CALENDAR.items():
    if k not in crop_calendar:
        crop_calendar[k] = v


#Helper: pick best sowing month
def get_sowing_month(current_month, suitable_months):
    for offset in range(12):
        candidate = (current_month - 1 + offset) % 12 + 1
        if candidate in suitable_months:
            crosses_year = (current_month + offset) > 12
            return candidate, offset, crosses_year
    return suitable_months[0], 0, False


def soften_proba(proba, temperature=3.0):
    """
    Temperature scaling: softens peaked GNB probabilities.
    Higher temperature → softer distribution.
    Takes raw proba array, returns softened version.
    """
    log_p = np.log(np.clip(proba, 1e-10, 1.0)) / temperature
    log_p -= np.max(log_p)  # numerical stability
    softened = np.exp(log_p)
    return softened / softened.sum()
    
#ENDPOINTS

#/predict endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        features = np.array([[
            data['N'], data['P'], data['K'],
            data['temperature'], data['humidity'],
            data['ph'], data['rainfall']
        ]])
        scaled = sc.transform(features)

        # Top-3 predictions
        proba    = soften_proba(model.predict_proba(scaled)[0])
        top3_idx = np.argsort(proba)[::-1][:3]
        top3     = [
            {
                'crop':       le.inverse_transform([i])[0],
                'confidence': round(float(proba[i]) * 100, 2)
            }
            for i in top3_idx
        ]

        crop_name     = top3[0]['crop']
        crop_key      = crop_name.lower().strip()
        current_month = datetime.now().month
        current_year  = datetime.now().year

        if crop_key in crop_calendar:
            info     = crop_calendar[crop_key]
            duration = info['duration']
            suitable = info['suitable_months']

            sow_month, wait_months, sow_crosses = get_sowing_month(current_month, suitable)
            sow_year = current_year + (1 if sow_crosses else 0)

            harvest_month_num = (sow_month - 1 + duration) % 12 + 1
            harvest_crosses   = (sow_month + duration) > 12
            harvest_year      = sow_year + (1 if harvest_crosses else 0)

            return jsonify({
                'crop':                   crop_name,
                'confidence':             top3[0]['confidence'],
                'top3':                   top3,
                'growth_duration_months': duration,
                'suitable_sowing_months': [calendar.month_abbr[m] for m in suitable],
                'recommended_sow_month':  f"{calendar.month_name[sow_month]} {sow_year}",
                'harvest_month':          f"{calendar.month_name[harvest_month_num]} {harvest_year}",
                'wait_months':            wait_months,
                'sowing_now':             wait_months == 0
            })
        else:
            return jsonify({
                'crop':       crop_name,
                'confidence': top3[0]['confidence'],
                'top3':       top3,
                'message':    f"Calendar data not available for '{crop_name}'."
            })

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


#/fertilizer endpoint
@app.route('/fertilizer', methods=['POST'])
def fertilizer():
    """
    Input:  { N, P, K, crop (optional) }
    Output: deficiency + fertilizer plan + cost for given crop
            If no crop given, uses top predicted crop.
    """
    data = request.get_json()
    try:
        soil = {
            'N': float(data['N']),
            'P': float(data['P']),
            'K': float(data['K'])
        }

        # If crop is passed directly, use it; otherwise predict
        if 'crop' in data and data['crop']:
            crop_name = data['crop'].strip()
        else:
            features = np.array([[
                data['N'], data['P'], data['K'],
                data.get('temperature', 25),
                data.get('humidity', 60),
                data.get('ph', 6.5),
                data.get('rainfall', 100)
            ]])
            scaled    = sc.transform(features)
            result    = model.predict(scaled)
            crop_name = le.inverse_transform(result)[0]

        plan = build_fertilizer_plan(crop_name, soil)

        if plan is None:
            return jsonify({
                'error': f"No requirement data found for crop: '{crop_name}'"
            }), 404

        return jsonify(plan)

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


#/fertilizer/top3 endpoint
@app.route('/fertilizer/top3', methods=['POST'])
def fertilizer_top3():
    """
    Input:  { N, P, K, temperature, humidity, ph, rainfall }
    Output: fertilizer plan + cost for all top-3 predicted crops
    """
    data = request.get_json()
    try:
        soil = {
            'N': float(data['N']),
            'P': float(data['P']),
            'K': float(data['K'])
        }

        features = np.array([[
            data['N'], data['P'], data['K'],
            data['temperature'], data['humidity'],
            data['ph'], data['rainfall']
        ]])
        scaled   = sc.transform(features)
        proba    = model.predict_proba(scaled)[0]
        top3_idx = np.argsort(proba)[::-1][:3]

        results = []
        for i in top3_idx:
            crop_name  = le.inverse_transform([i])[0]
            confidence = round(float(proba[i]) * 100, 2)
            plan       = build_fertilizer_plan(crop_name, soil)
            if plan:
                plan['confidence'] = confidence
                results.append(plan)

        return jsonify({'top3': results})

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#PHASE 3 — Multi-Crop Scoring & Ranking

def normalize(values):
    """Min-max normalize a list of values to [0, 1]."""
    mn, mx = min(values), max(values)
    if mx == mn:
        return [1.0 for _ in values]
    return [(v - mn) / (mx - mn) for v in values]


def generate_reason(crop_name, confidence, cost, norm_cost, score, deficiency):
    """
    Generate a human-readable explanation for why a crop was ranked
    at its position.
    """
    reasons = []

    # Confidence comment
    if confidence >= 80:
        reasons.append("high model confidence")
    elif confidence >= 50:
        reasons.append("moderate model confidence")
    else:
        reasons.append("lower model confidence")

    # Cost comment
    if norm_cost <= 0.33:
        reasons.append("low fertilizer cost")
    elif norm_cost <= 0.66:
        reasons.append("moderate fertilizer cost")
    else:
        reasons.append("higher fertilizer investment needed")

    # Deficiency comment
    total_def = deficiency['N'] + deficiency['P'] + deficiency['K']
    if total_def == 0:
        reasons.append("soil already meets all nutrient requirements")
    elif total_def <= 50:
        reasons.append("minimal nutrient correction needed")
    else:
        reasons.append("significant nutrient correction required")

    return f"Selected for {reasons[0]}, {reasons[1]}, and {reasons[2]}."


def rank_crops(top3_plans, confidences):
    """
    Score and rank top-3 crops using:
        score = (0.7 × confidence_norm) − (0.3 × cost_norm)

    Higher score = better recommendation.
    """
    # Pair plans with their confidence
    for i, plan in enumerate(top3_plans):
        plan['confidence'] = confidences[i]

    # Normalize cost (higher cost = worse)
    costs = [p['total_cost'] for p in top3_plans]
    norm_costs = normalize(costs)
    # Invert: lower cost gets higher normalized value
    norm_costs_inv = [1 - nc for nc in norm_costs]

    # Normalize confidence (already 0-100, convert to 0-1)
    confs      = [p['confidence'] / 100 for p in top3_plans]
    norm_confs = normalize(confs)

    # Compute final score
    W_CONF = 0.7
    W_COST = 0.3

    for i, plan in enumerate(top3_plans):
        norm_cost  = norm_costs[i]
        score      = (W_CONF * norm_confs[i]) + (W_COST * norm_costs_inv[i])
        plan['score']       = round(score, 4)
        plan['norm_cost']   = round(norm_cost, 4)
        plan['norm_conf']   = round(norm_confs[i], 4)
        plan['reason']      = generate_reason(
            plan['crop'],
            plan['confidence'],
            plan['total_cost'],
            norm_cost,
            score,
            plan['deficiency']
        )

    # Sort by score descending
    top3_plans.sort(key=lambda x: x['score'], reverse=True)

    # Add rank
    for i, plan in enumerate(top3_plans):
        plan['rank'] = i + 1

    return top3_plans


#/recommend endpoint 
@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Master endpoint — runs the full pipeline:
    Predict → Fertilizer → Score → Rank → Explain

    Input:  { N, P, K, temperature, humidity, ph, rainfall }
    Output: ranked top-3 crops with full fertilizer + cost + score + reason
    """
    data = request.get_json()
    try:
        soil = {
            'N': float(data['N']),
            'P': float(data['P']),
            'K': float(data['K'])
        }
        soil_type = data.get('soil_type', 'Loamy')

        features = np.array([[
            data['N'], data['P'], data['K'],
            data['temperature'], data['humidity'],
            data['ph'], data['rainfall']
        ]])
        scaled   = sc.transform(features)
        proba    = soften_proba(model.predict_proba(scaled)[0])
        top3_idx = np.argsort(proba)[::-1][:3]
        confidences = []
        plans       = []

        for i in top3_idx:
            crop_name = le.inverse_transform([i])[0]
            plan      = build_fertilizer_plan(crop_name, soil, soil_type)
            ideal     = crop_requirements.get(crop_name.lower().strip(), {'N':0,'P':0,'K':0})
            match     = soil_match_score(soil, ideal)

            if plan is None:
                # Crop not in requirements — include with empty plan
                plan = {
                    'crop':       crop_name,
                    'ideal_npk':  {},
                    'deficiency': {'N': 0, 'P': 0, 'K': 0},
                    'fert_plan':  {},
                    'breakdown':  {},
                    'total_cost': 0,
                    'summary':    ['No requirement data available.']
                }

            confidences.append(match)
            plans.append(plan)

        # Score and rank
        ranked = rank_crops(plans, confidences)

        # Attach calendar data for the top-ranked crop
        best      = ranked[0]
        crop_key  = best['crop'].lower().strip()
        current_month = datetime.now().month
        current_year  = datetime.now().year

        calendar_info = {}
        sow_month_num = current_month
        sow_yr        = current_year

        if crop_key in crop_calendar:
            info     = crop_calendar[crop_key]
            duration = info['duration']
            suitable = info['suitable_months']
            sow_month, wait_months, sow_crosses = get_sowing_month(current_month, suitable)
            sow_yr        = current_year + (1 if sow_crosses else 0)
            sow_month_num = sow_month
            harvest_month_num = (sow_month - 1 + duration) % 12 + 1
            harvest_crosses   = (sow_month + duration) > 12
            harvest_year      = sow_yr + (1 if harvest_crosses else 0)
            calendar_info = {
                'growth_duration_months': duration,
                'recommended_sow_month':  f"{calendar.month_name[sow_month]} {sow_yr}",
                'harvest_month':          f"{calendar.month_name[harvest_month_num]} {harvest_year}",
                'sowing_now':             wait_months == 0
            }

        # Generate stage-by-stage application schedule
        schedule = generate_schedule(
            best['crop'],
            best.get('fert_plan', {}),
            sow_month_num,
            sow_yr
        )

        return jsonify({
            'best':     best['crop'],
            'calendar': calendar_info,
            'schedule': schedule,
            'ranked':   ranked
        })

    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


#/metrics endpoint
@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        per_class = []
        for crop_name in le.classes_:
            r = report.get(crop_name, {})
            per_class.append({
                'crop':      crop_name,
                'precision': round(r.get('precision', 0), 4),
                'recall':    round(r.get('recall', 0), 4),
                'f1':        round(r.get('f1-score', 0), 4),
                'support':   int(r.get('support', 0))
            })

        return jsonify({
            'accuracy':         round(acc,   4),
            'error_rate':       round(err,   4),
            'f1_weighted':      round(f1,    4),
            'mcc':              round(mcc,   4),
            'kappa':            round(kappa, 4),
            'gmean':            round(gmean, 4),
            'confusion_matrix': cm.tolist(),
            'class_names':      le.classes_.tolist(),
            'per_class':        per_class
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)
