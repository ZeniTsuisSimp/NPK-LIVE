import streamlit as st
import numpy as np
import joblib
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import math

# ─── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="NPK Crop Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS Theme ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;500;600;700;800&display=swap');

/* ── Global Theme ── */
:root {
    --bg-primary: #0f1117;
    --bg-card: rgba(25, 28, 39, 0.85);
    --bg-card-hover: rgba(35, 40, 55, 0.95);
    --accent-green: #22c55e;
    --accent-emerald: #10b981;
    --accent-lime: #84cc16;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border-subtle: rgba(148, 163, 184, 0.12);
    --glow-green: rgba(34, 197, 94, 0.15);
}

.stApp {
    font-family: 'Inter', sans-serif !important;
}

/* ── Hero Header ── */
.hero-container {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 30%, #047857 60%, #059669 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
    border: 1px solid rgba(16, 185, 129, 0.2);
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255,255,255,0.05);
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(34, 197, 94, 0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-family: 'Outfit', sans-serif !important;
    font-size: 2.6rem;
    font-weight: 800;
    color: #ecfdf5;
    margin: 0;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
}
.hero-subtitle {
    font-size: 1.1rem;
    color: #a7f3d0;
    margin-top: 0.5rem;
    font-weight: 400;
    letter-spacing: 0.3px;
}

/* ── Metric Cards ── */
.metric-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.metric-card:hover {
    background: var(--bg-card-hover);
    border-color: var(--accent-emerald);
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(16, 185, 129, 0.15);
}
.metric-value {
    font-family: 'Outfit', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0.5rem 0 0.25rem;
}
.metric-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 500;
}
.metric-status {
    font-size: 0.8rem;
    margin-top: 0.5rem;
    padding: 0.25rem 0.75rem;
    border-radius: 100px;
    display: inline-block;
    font-weight: 600;
}

/* ── Info Cards ── */
.info-card {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.75rem;
    transition: all 0.3s ease;
    height: 100%;
}
.info-card:hover {
    border-color: rgba(148, 163, 184, 0.25);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
}
.info-card h4 {
    font-family: 'Outfit', sans-serif;
    color: var(--text-primary);
    font-size: 1.15rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}
.info-card p, .info-card li {
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.7;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Outfit', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 1.5rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border-subtle);
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

/* ── Score Display ── */
.score-container {
    text-align: center;
    padding: 2rem;
}
.score-circle {
    width: 180px;
    height: 180px;
    border-radius: 50%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1rem;
    border: 4px solid;
    box-shadow: 0 0 40px rgba(0,0,0,0.2);
}
.score-number {
    font-family: 'Outfit', sans-serif;
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
}
.score-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 0.25rem;
}

/* ── Rotation Card ── */
.rotation-step {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
}
.rotation-step:hover {
    border-color: var(--accent-emerald);
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.1);
}
.rotation-crop {
    font-family: 'Outfit', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
}
.rotation-reason {
    font-size: 0.85rem;
    color: var(--text-secondary);
    margin-top: 0.5rem;
    line-height: 1.5;
}

/* ── Season Badge ── */
.season-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
}
.season-kharif { background: rgba(34, 197, 94, 0.15); color: #22c55e; border: 1px solid rgba(34, 197, 94, 0.3); }
.season-rabi { background: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
.season-zaid { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
.season-active { box-shadow: 0 0 15px rgba(16, 185, 129, 0.3); }

/* ── Sidebar Styling ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1821 0%, #0f2027 50%, #112e3a 100%) !important;
    border-right: 1px solid var(--border-subtle);
}
section[data-testid="stSidebar"] .stRadio > label {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    color: var(--text-primary);
    font-size: 1rem;
}

/* ── Hide Streamlit defaults ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* ── Divider ── */
.custom-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border-subtle), transparent);
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)


# ─── Crop Knowledge Base ─────────────────────────────────────────────────────
# Nutrient impact: how much a crop depletes (negative) or adds (positive) each nutrient
CROP_NUTRIENT_IMPACT = {
    'Rice':      {'N': -40, 'P': -15, 'K': -20, 'family': 'cereal',    'emoji': '🌾'},
    'Wheat':     {'N': -35, 'P': -18, 'K': -15, 'family': 'cereal',    'emoji': '🌿'},
    'Corn':      {'N': -50, 'P': -20, 'K': -30, 'family': 'cereal',    'emoji': '🌽'},
    'Barley':    {'N': -25, 'P': -10, 'K': -12, 'family': 'cereal',    'emoji': '🌾'},
    'Soybean':   {'N': +20, 'P': -10, 'K': -15, 'family': 'legume',    'emoji': '🫘'},
    'Cotton':    {'N': -30, 'P': -25, 'K': -35, 'family': 'cash_crop', 'emoji': '🧶'},
    'Sugarcane': {'N': -60, 'P': -25, 'K': -40, 'family': 'grass',     'emoji': '🎋'},
    'Tomato':    {'N': -35, 'P': -30, 'K': -45, 'family': 'solanaceae', 'emoji': '🍅'},
    'Potato':    {'N': -25, 'P': -20, 'K': -40, 'family': 'solanaceae', 'emoji': '🥔'},
    'Onion':     {'N': -20, 'P': -15, 'K': -30, 'family': 'allium',    'emoji': '🧅'},
}

# Crop requirements for reference (optimal NPK ranges in mg/kg)
CROP_REQUIREMENTS = {
    'Rice':      {'N': (80, 120),  'P': (40, 60),   'K': (40, 60)},
    'Wheat':     {'N': (100, 140), 'P': (50, 70),   'K': (50, 70)},
    'Corn':      {'N': (120, 180), 'P': (60, 90),   'K': (60, 100)},
    'Barley':    {'N': (60, 100),  'P': (30, 50),   'K': (40, 60)},
    'Soybean':   {'N': (40, 80),   'P': (40, 80),   'K': (80, 120)},
    'Cotton':    {'N': (100, 150), 'P': (50, 80),   'K': (80, 120)},
    'Sugarcane': {'N': (150, 200), 'P': (60, 100),  'K': (100, 150)},
    'Tomato':    {'N': (120, 160), 'P': (80, 120),  'K': (150, 200)},
    'Potato':    {'N': (100, 140), 'P': (60, 100),  'K': (120, 180)},
    'Onion':     {'N': (80, 120),  'P': (50, 80),   'K': (100, 140)},
}

# Seasonal calendar for Indian agriculture
CROP_SEASONS = {
    'Kharif':  {'months': 'Jun – Oct', 'crops': ['Rice', 'Corn', 'Cotton', 'Soybean', 'Sugarcane'], 'color': 'season-kharif'},
    'Rabi':    {'months': 'Nov – Mar', 'crops': ['Wheat', 'Barley', 'Potato', 'Onion'], 'color': 'season-rabi'},
    'Zaid':    {'months': 'Mar – Jun', 'crops': ['Tomato', 'Onion'], 'color': 'season-zaid'},
}

# Rotation companion rules
ROTATION_RULES = {
    'cereal':     ['legume', 'allium', 'solanaceae'],
    'legume':     ['cereal', 'solanaceae', 'grass'],
    'cash_crop':  ['legume', 'cereal', 'allium'],
    'grass':      ['legume', 'allium', 'cereal'],
    'solanaceae': ['legume', 'cereal', 'allium'],
    'allium':     ['cereal', 'legume', 'solanaceae'],
}


# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_data
def load_model():
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.normpath(os.path.join(script_dir, '..'))
        candidate_root = os.path.join(project_root, 'models', 'npk_crop_model.pkl')
        candidate_local = os.path.join(script_dir, 'models', 'npk_crop_model.pkl')
        candidate_parent = os.path.normpath(os.path.join(script_dir, '..', 'ml', 'models', 'npk_crop_model.pkl'))

        for p in [candidate_root, candidate_local, candidate_parent]:
            if os.path.exists(p):
                return joblib.load(p)

        raise FileNotFoundError(f"Model file not found. Tried: {[candidate_root, candidate_local, candidate_parent]}")
    except FileNotFoundError:
        st.error("⚠️ Model file not found. Please ensure the model is trained and saved in the 'models' folder.")
        return None
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        return None


# ─── Helper Functions ─────────────────────────────────────────────────────────
def get_nutrient_status(value, nutrient):
    """Return status label and color for a nutrient value."""
    ranges = {
        'N': {'low': 60, 'high': 150},
        'P': {'low': 30, 'high': 90},
        'K': {'low': 50, 'high': 120}
    }
    if value < ranges[nutrient]['low']:
        return "Low", "#ef4444"
    elif value > ranges[nutrient]['high']:
        return "High", "#f59e0b"
    else:
        return "Optimal", "#22c55e"


def predict_crop(n, p, k, model_data):
    """Run the ML model and return (predicted_crop, probability_dict)."""
    input_data = np.array([[n, p, k]])
    input_scaled = model_data['scaler'].transform(input_data)
    prediction = model_data['model'].predict(input_scaled)
    predicted_crop = model_data['label_encoder'].inverse_transform(prediction)[0]
    probabilities = model_data['model'].predict_proba(input_scaled)[0]
    prob_dict = {crop: prob for crop, prob in zip(model_data['target_names'], probabilities)}
    return predicted_crop, prob_dict


def recommend_additions(cur_n, cur_p, cur_k, target_crop, strategy='mid'):
    """Compute required NPK additions (mg/kg) to reach target crop ranges."""
    if target_crop not in CROP_REQUIREMENTS:
        raise ValueError(f"Unknown crop: {target_crop}")
    req = CROP_REQUIREMENTS[target_crop]
    if strategy == 'mid':
        target_n = (req['N'][0] + req['N'][1]) / 2
        target_p = (req['P'][0] + req['P'][1]) / 2
        target_k = (req['K'][0] + req['K'][1]) / 2
    else:
        target_n, target_p, target_k = req['N'][0], req['P'][0], req['K'][0]
    # Raw difference: positive = need to add, negative = excess in soil
    diffs = {
        'N': round(target_n - cur_n, 2),
        'P': round(target_p - cur_p, 2),
        'K': round(target_k - cur_k, 2),
    }
    targets = {'N': round(target_n, 2), 'P': round(target_p, 2), 'K': round(target_k, 2)}
    return diffs, targets


def compute_soil_health(n, p, k):
    """Compute a 0–100 soil health score based on N, P, K values."""
    # Optimal midpoints
    opt_n, opt_p, opt_k = 105, 60, 100
    # Bell-curve scoring for each nutrient (Gaussian with std = half of typical range)
    def bell(value, optimal, sigma=50):
        return math.exp(-0.5 * ((value - optimal) / sigma) ** 2) * 100

    score_n = bell(n, opt_n, 55)
    score_p = bell(p, opt_p, 35)
    score_k = bell(k, opt_k, 50)

    # Weighted average
    base = 0.35 * score_n + 0.30 * score_p + 0.35 * score_k

    # Balance bonus: if nutrients are within 30% ratio of each other
    values = [n, p, k]
    if min(values) > 0:
        ratio = max(values) / min(values)
        if ratio < 2.5:
            balance_bonus = max(0, 10 * (1 - (ratio - 1) / 1.5))
        else:
            balance_bonus = 0
    else:
        balance_bonus = 0

    return min(100, round(base + balance_bonus, 1))


def get_health_grade(score):
    """Return grade label and color for a health score."""
    if score >= 80:
        return "Excellent", "#22c55e"
    elif score >= 60:
        return "Good", "#84cc16"
    elif score >= 40:
        return "Fair", "#f59e0b"
    else:
        return "Poor", "#ef4444"


def suggest_rotation(previous_crop):
    """Suggest the best next crops based on rotation science."""
    if previous_crop not in CROP_NUTRIENT_IMPACT:
        return []

    prev = CROP_NUTRIENT_IMPACT[previous_crop]
    prev_family = prev['family']
    preferred_families = ROTATION_RULES.get(prev_family, [])

    suggestions = []
    for crop, info in CROP_NUTRIENT_IMPACT.items():
        if crop == previous_crop:
            continue

        score = 0
        reasons = []

        # Family compatibility
        if info['family'] in preferred_families:
            rank = preferred_families.index(info['family'])
            score += (3 - rank) * 25  # 75, 50, 25 based on rank
            reasons.append(f"Good rotation after {prev_family}")

        # Same family penalty
        if info['family'] == prev_family:
            score -= 40
            reasons.append("⚠️ Same crop family — risk of disease buildup")

        # Solanaceae back-to-back penalty
        if prev_family == 'solanaceae' and info['family'] == 'solanaceae':
            score -= 30
            reasons.append("⚠️ Avoid Solanaceae back-to-back (blight risk)")

        # N-fixer bonus after heavy N consumer
        if prev['N'] < -30 and info.get('N', 0) > 0:
            score += 40
            reasons.append("🌱 Legume fixes nitrogen depleted by previous crop")

        # Light feeder after heavy K consumer
        if prev['K'] < -30 and info.get('K', 0) > -20:
            score += 20
            reasons.append("Gentle on potassium — lets soil recover")

        # Nutrient balance: prefer crops that don't deplete the same nutrient heavily
        if prev['N'] < -30 and info.get('N', 0) > -20:
            score += 15
            reasons.append("Low nitrogen demand")
        if prev['P'] < -20 and info.get('P', 0) > -15:
            score += 10
            reasons.append("Low phosphorus demand")

        suggestions.append({
            'crop': crop,
            'emoji': info['emoji'],
            'score': score,
            'reasons': reasons,
            'family': info['family'],
            'n_impact': info['N'],
            'p_impact': info['P'],
            'k_impact': info['K'],
        })

    suggestions.sort(key=lambda x: x['score'], reverse=True)
    return suggestions


def get_current_season():
    """Return current Indian farming season based on month."""
    month = datetime.now().month
    if 6 <= month <= 10:
        return 'Kharif'
    elif month >= 11 or month <= 2:
        return 'Rabi'
    else:
        return 'Zaid'


# ─── Hero Header ──────────────────────────────────────────────────────────────
def render_hero():
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">🌾 NPK Crop Intelligence</div>
        <div class="hero-subtitle">Smart soil analysis • Crop rotation planning • Nutrient management</div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Crop Prediction
# ═══════════════════════════════════════════════════════════════════════════════
def page_crop_prediction(model_data):
    st.markdown('<div class="section-header">🌾 Crop Prediction from Soil NPK</div>', unsafe_allow_html=True)

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown("##### 📊 Enter Soil NPK Values")
        with st.form("prediction_form"):
            nitrogen = st.number_input("Nitrogen (N) — mg/kg", value=100.0, min_value=0.0, max_value=300.0, step=1.0, help="Typical: 40–200 mg/kg")
            phosphorus = st.number_input("Phosphorus (P) — mg/kg", value=50.0, min_value=0.0, max_value=200.0, step=1.0, help="Typical: 20–120 mg/kg")
            potassium = st.number_input("Potassium (K) — mg/kg", value=80.0, min_value=0.0, max_value=250.0, step=1.0, help="Typical: 40–200 mg/kg")
            predict_btn = st.form_submit_button("🔍 Analyze & Recommend", use_container_width=True)

        # Nutrient status cards
        n_status, n_color = get_nutrient_status(nitrogen, 'N')
        p_status, p_color = get_nutrient_status(phosphorus, 'P')
        k_status, k_color = get_nutrient_status(potassium, 'K')

        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Nitrogen (N)</div>
                <div class="metric-value" style="color: {n_color}">{nitrogen:.0f}</div>
                <div class="metric-status" style="background: {n_color}22; color: {n_color}">{n_status}</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Phosphorus (P)</div>
                <div class="metric-value" style="color: {p_color}">{phosphorus:.0f}</div>
                <div class="metric-status" style="background: {p_color}22; color: {p_color}">{p_status}</div>
            </div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Potassium (K)</div>
                <div class="metric-value" style="color: {k_color}">{potassium:.0f}</div>
                <div class="metric-status" style="background: {k_color}22; color: {k_color}">{k_status}</div>
            </div>""", unsafe_allow_html=True)

    with col_result:
        st.markdown("##### 🌱 Prediction Results")
        if predict_btn:
            if nitrogen <= 0 or phosphorus <= 0 or potassium <= 0:
                st.error("Please enter positive values for all NPK nutrients.")
            else:
                with st.spinner("Analyzing soil composition..."):
                    predicted_crop, probabilities = predict_crop(nitrogen, phosphorus, potassium, model_data)

                # Save to history
                if 'npk_history' not in st.session_state:
                    st.session_state.npk_history = []
                st.session_state.npk_history.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'N': nitrogen, 'P': phosphorus, 'K': potassium,
                    'crop': predicted_crop
                })

                emoji = CROP_NUTRIENT_IMPACT.get(predicted_crop, {}).get('emoji', '🌱')
                st.success(f"### {emoji} Recommended: **{predicted_crop}**")

                # Top 3 with progress bars
                sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:5]
                medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
                for i, (crop, prob) in enumerate(sorted_probs):
                    conf = prob * 100
                    st.markdown(f"{medals[i]} **{crop}** — {conf:.1f}%")
                    st.progress(prob)

                # Probability chart
                prob_df = pd.DataFrame(list(probabilities.items()), columns=['Crop', 'Probability'])
                prob_df['Probability'] = prob_df['Probability'] * 100
                prob_df = prob_df.sort_values('Probability', ascending=True)
                fig = px.bar(prob_df, x='Probability', y='Crop', orientation='h',
                             color='Probability', color_continuous_scale='Emrld',
                             title="All Crop Confidence Scores (%)")
                fig.update_layout(
                    height=350, showlegend=False,
                    xaxis_title="Confidence (%)", yaxis_title="",
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#94a3b8')
                )
                st.plotly_chart(fig, use_container_width=True)

                # Comparison with optimal
                if predicted_crop in CROP_REQUIREMENTS:
                    req = CROP_REQUIREMENTS[predicted_crop]
                    st.markdown("##### Soil vs Optimal Requirements")
                    comp = pd.DataFrame({
                        'Nutrient': ['N', 'P', 'K'],
                        'Your Soil': [nitrogen, phosphorus, potassium],
                        'Optimal Min': [req['N'][0], req['P'][0], req['K'][0]],
                        'Optimal Max': [req['N'][1], req['P'][1], req['K'][1]],
                    })
                    st.dataframe(comp, use_container_width=True, hide_index=True)
        else:
            st.info("👆 Enter your soil NPK values and click **Analyze & Recommend** to get started.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: NPK Additions
# ═══════════════════════════════════════════════════════════════════════════════
def page_npk_additions(model_data):
    st.markdown('<div class="section-header">🧪 NPK Addition Recommendations</div>', unsafe_allow_html=True)
    st.markdown("Find out how much **N, P, K (mg/kg)** to add to your soil for a target crop.")

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("##### 🎯 Target Crop & Current Soil")

        # Quick lookup
        with st.expander("🔎 View crop NPK requirements"):
            lk = st.selectbox("Crop:", list(CROP_REQUIREMENTS.keys()), key="lk_crop")
            req = CROP_REQUIREMENTS[lk]
            mid_n = (req['N'][0] + req['N'][1]) / 2
            mid_p = (req['P'][0] + req['P'][1]) / 2
            mid_k = (req['K'][0] + req['K'][1]) / 2
            st.dataframe(pd.DataFrame({
                'Nutrient': ['N', 'P', 'K'],
                'Min': [req['N'][0], req['P'][0], req['K'][0]],
                'Max': [req['N'][1], req['P'][1], req['K'][1]],
                'Midpoint': [mid_n, mid_p, mid_k]
            }), use_container_width=True, hide_index=True)

        with st.form("add_form"):
            target_crop = st.selectbox("Target crop:", model_data.get('target_names', []))
            cur_n = st.number_input("Current N (mg/kg)", value=100.0, min_value=0.0, max_value=1000.0, step=1.0)
            cur_p = st.number_input("Current P (mg/kg)", value=50.0, min_value=0.0, max_value=1000.0, step=1.0)
            cur_k = st.number_input("Current K (mg/kg)", value=80.0, min_value=0.0, max_value=1000.0, step=1.0)
            method = st.radio("Strategy:", ["Midpoint (recommended)", "Minimum optimal"])
            submit_add = st.form_submit_button("✅ Calculate Additions", use_container_width=True)

    with col2:
        st.markdown("##### 📋 Required Additions")
        if submit_add:
            if target_crop not in CROP_REQUIREMENTS:
                st.error(f"Unknown crop: {target_crop}")
            else:
                strat = 'mid' if method.startswith('Mid') else 'min'
                diffs, targets = recommend_additions(cur_n, cur_p, cur_k, target_crop, strat)

                emoji = CROP_NUTRIENT_IMPACT.get(target_crop, {}).get('emoji', '🌱')
                st.markdown(f"### {emoji} Target: **{target_crop}**")

                # Visual cards for additions
                ac1, ac2, ac3 = st.columns(3)
                for col, (nut, diff, target_val) in zip([ac1, ac2, ac3],
                    [('N', diffs['N'], targets['N']),
                     ('P', diffs['P'], targets['P']),
                     ('K', diffs['K'], targets['K'])]):
                    if diff < -5:
                        # Excess — current is much higher than target
                        color = "#ef4444"
                        label = f"⬆️ Excess: {abs(diff)} mg/kg"
                    elif diff < 0:
                        # Slightly over but close enough
                        color = "#f59e0b"
                        label = f"⬆️ Slight excess"
                    elif diff == 0:
                        color = "#22c55e"
                        label = "✅ Optimal"
                    elif diff <= 10:
                        color = "#84cc16"
                        label = f"✅ Near optimal"
                    elif diff <= 30:
                        color = "#f59e0b"
                        label = f"+{diff} mg/kg needed"
                    else:
                        color = "#ef4444"
                        label = f"+{diff} mg/kg needed"
                    with col:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{nut}</div>
                            <div class="metric-value" style="color: {color}">{label}</div>
                            <div class="metric-status" style="background: {color}22; color: {color}">Target: {target_val}</div>
                        </div>""", unsafe_allow_html=True)

                st.markdown("")
                table = pd.DataFrame({
                    'Nutrient': ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)'],
                    'Current': [cur_n, cur_p, cur_k],
                    'Target': [targets['N'], targets['P'], targets['K']],
                    'Difference (mg/kg)': [diffs['N'], diffs['P'], diffs['K']],
                    'Status': [
                        '⬆️ Excess' if diffs['N'] < -5 else ('✅ OK' if diffs['N'] <= 10 else '⬇️ Low'),
                        '⬆️ Excess' if diffs['P'] < -5 else ('✅ OK' if diffs['P'] <= 10 else '⬇️ Low'),
                        '⬆️ Excess' if diffs['K'] < -5 else ('✅ OK' if diffs['K'] <= 10 else '⬇️ Low'),
                    ]
                })
                st.dataframe(table, use_container_width=True, hide_index=True)

                # If any nutrient is in excess, suggest a better-suited crop
                has_excess = any(d < -5 for d in diffs.values())
                if has_excess and model_data:
                    st.markdown("---")
                    st.markdown("##### 💡 Your soil doesn't match this crop — here's what fits better")
                    st.caption("Since you can't easily reduce soil nutrients, consider growing a crop that thrives with your current NPK levels:")
                    best_crop, probs = predict_crop(cur_n, cur_p, cur_k, model_data)
                    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
                    sc1, sc2, sc3 = st.columns(3)
                    for i, (col, (crop, prob)) in enumerate(zip([sc1, sc2, sc3], sorted_probs)):
                        medal = ["🥇", "🥈", "🥉"][i]
                        crop_emoji = CROP_NUTRIENT_IMPACT.get(crop, {}).get('emoji', '🌱')
                        conf = prob * 100
                        border = "#22c55e" if i == 0 else "#84cc16" if i == 1 else "#64748b"
                        with col:
                            st.markdown(f"""
                            <div class="rotation-step" style="border-color: {border}">
                                <div style="font-size: 1.8rem;">{medal} {crop_emoji}</div>
                                <div class="rotation-crop">{crop}</div>
                                <div style="color: {border}; font-weight: 700; font-size: 1.1rem; margin-top: 0.5rem;">{conf:.1f}% match</div>
                            </div>""", unsafe_allow_html=True)
                    st.info(f"🌱 **Best match for your soil (N={cur_n}, P={cur_p}, K={cur_k}): {best_crop}** — consider growing this instead!")

                st.markdown("---")
                st.caption("💡 Values are approximate mg/kg adjustments. Actual application rates depend on soil depth, bulk density, and local conditions. Consult an agronomist for precise guidance.")
        else:
            st.info("👆 Select a target crop, enter your current soil values, and click **Calculate Additions**.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Crop Rotation Advisor
# ═══════════════════════════════════════════════════════════════════════════════
def page_crop_rotation():
    st.markdown('<div class="section-header">🔄 Smart Crop Rotation Advisor</div>', unsafe_allow_html=True)
    st.markdown("Maintain soil fertility by choosing the right **next crop** based on what you just harvested.")

    col_input, col_result = st.columns([1, 2], gap="large")

    with col_input:
        st.markdown("##### 🌾 Previous Crop")
        previous_crop = st.selectbox("What did you grow last?", list(CROP_NUTRIENT_IMPACT.keys()), key="rot_prev")
        prev = CROP_NUTRIENT_IMPACT[previous_crop]

        st.markdown(f"**{prev['emoji']} {previous_crop}** — Nutrient Impact:")
        impact_df = pd.DataFrame({
            'Nutrient': ['N', 'P', 'K'],
            'Impact (mg/kg)': [prev['N'], prev['P'], prev['K']],
            'Effect': [
                '🌱 Adds nitrogen' if prev['N'] > 0 else '⬇️ Depletes nitrogen',
                '🌱 Adds phosphorus' if prev['P'] > 0 else '⬇️ Depletes phosphorus',
                '🌱 Adds potassium' if prev['K'] > 0 else '⬇️ Depletes potassium',
            ]
        })
        st.dataframe(impact_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("##### 📐 How Rotation Works")
        st.markdown("""
        - **Legumes** (Soybean) fix atmospheric N into soil
        - **Cereals** after legumes use the extra nitrogen
        - **Avoid same-family** crops back-to-back (disease risk)
        - **Alternate** heavy and light feeders
        """)

    with col_result:
        st.markdown("##### 🏆 Recommended Next Crops")
        suggestions = suggest_rotation(previous_crop)

        if not suggestions:
            st.warning("No rotation data available for this crop.")
            return

        # Top 3 as highlighted cards
        top3 = suggestions[:3]
        cols = st.columns(3)
        for i, (col, sug) in enumerate(zip(cols, top3)):
            medal = ["🥇", "🥈", "🥉"][i]
            border_color = ["#22c55e", "#84cc16", "#f59e0b"][i]
            with col:
                reasons_html = "<br>".join(f"• {r}" for r in sug['reasons'][:3]) if sug['reasons'] else "General rotation benefit"
                st.markdown(f"""
                <div class="rotation-step" style="border-color: {border_color}">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">{medal} {sug['emoji']}</div>
                    <div class="rotation-crop">{sug['crop']}</div>
                    <div class="rotation-reason">{reasons_html}</div>
                    <div style="margin-top: 0.75rem; font-size: 0.75rem; color: #64748b;">
                        N: {sug['n_impact']:+d} &nbsp; P: {sug['p_impact']:+d} &nbsp; K: {sug['k_impact']:+d}
                    </div>
                </div>""", unsafe_allow_html=True)

        # Rotation chain visualization
        st.markdown("")
        st.markdown("##### 🔄 Suggested 3-Season Rotation Chain")
        chain = [previous_crop]
        current = previous_crop
        for _ in range(3):
            next_suggestions = suggest_rotation(current)
            if next_suggestions:
                next_crop = next_suggestions[0]['crop']
                chain.append(next_crop)
                current = next_crop
            else:
                break

        chain_parts = []
        for j, c in enumerate(chain):
            em = CROP_NUTRIENT_IMPACT.get(c, {}).get('emoji', '🌱')
            label = "Previous" if j == 0 else f"Season {j}"
            chain_parts.append(f"**{em} {c}** ({label})")
        st.markdown(" → ".join(chain_parts))

        # Full ranking
        st.markdown("")
        st.markdown("##### 📋 Full Ranking")
        rank_df = pd.DataFrame([{
            'Rank': i + 1,
            'Crop': f"{s['emoji']} {s['crop']}",
            'Family': s['family'].replace('_', ' ').title(),
            'N Impact': f"{s['n_impact']:+d}",
            'P Impact': f"{s['p_impact']:+d}",
            'K Impact': f"{s['k_impact']:+d}",
            'Score': s['score'],
        } for i, s in enumerate(suggestions)])
        st.dataframe(rank_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Soil Health Score
# ═══════════════════════════════════════════════════════════════════════════════
def page_soil_health():
    st.markdown('<div class="section-header">💚 Soil Health Score</div>', unsafe_allow_html=True)
    st.markdown("Get a composite **0–100 health score** for your soil based on NPK balance.")

    col_input, col_result = st.columns([1, 1.5], gap="large")

    with col_input:
        st.markdown("##### 📊 Enter Soil Values")
        n_val = st.number_input("Nitrogen (N) — mg/kg", value=100.0, min_value=0.0, max_value=300.0, step=1.0, key="health_n")
        p_val = st.number_input("Phosphorus (P) — mg/kg", value=50.0, min_value=0.0, max_value=200.0, step=1.0, key="health_p")
        k_val = st.number_input("Potassium (K) — mg/kg", value=80.0, min_value=0.0, max_value=250.0, step=1.0, key="health_k")
        analyze_btn = st.button("🩺 Analyze Soil Health", use_container_width=True)

    with col_result:
        if analyze_btn:
            score = compute_soil_health(n_val, p_val, k_val)
            grade, color = get_health_grade(score)

            # Score display
            st.markdown(f"""
            <div class="score-container">
                <div class="score-circle" style="border-color: {color}; background: {color}15;">
                    <div class="score-number" style="color: {color}">{score:.0f}</div>
                    <div class="score-label" style="color: {color}">{grade}</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Radar chart
            categories = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)']
            # Normalize to 0-100 scale based on typical ranges
            norm_n = min(100, (n_val / 150) * 100)
            norm_p = min(100, (p_val / 100) * 100)
            norm_k = min(100, (k_val / 150) * 100)

            fig = go.Figure()
            # Ideal profile
            fig.add_trace(go.Scatterpolar(
                r=[70, 60, 67, 70],
                theta=categories + [categories[0]],
                fill='toself',
                name='Ideal Range',
                line=dict(color='rgba(34, 197, 94, 0.5)', width=2),
                fillcolor='rgba(34, 197, 94, 0.08)'
            ))
            # User profile
            fig.add_trace(go.Scatterpolar(
                r=[norm_n, norm_p, norm_k, norm_n],
                theta=categories + [categories[0]],
                fill='toself',
                name='Your Soil',
                line=dict(color='#60a5fa', width=3),
                fillcolor='rgba(96, 165, 250, 0.15)'
            ))
            fig.update_layout(
                polar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor='rgba(148,163,184,0.1)'),
                    angularaxis=dict(gridcolor='rgba(148,163,184,0.1)')
                ),
                showlegend=True,
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                margin=dict(t=30, b=30)
            )
            st.plotly_chart(fig, use_container_width=True)

            # Interpretation
            st.markdown("##### 📝 Interpretation")
            if score >= 80:
                st.success("🌟 Your soil is in excellent condition! Nutrient levels are well-balanced and within optimal ranges.")
            elif score >= 60:
                st.info("👍 Your soil is in good health. Minor adjustments could optimize it further for specific crops.")
            elif score >= 40:
                st.warning("⚠️ Your soil health is fair. Consider targeted nutrient amendments to improve balance.")
            else:
                st.error("🚨 Your soil needs attention. One or more nutrients are significantly outside optimal ranges.")

            # Individual status
            n_st, n_cl = get_nutrient_status(n_val, 'N')
            p_st, p_cl = get_nutrient_status(p_val, 'P')
            k_st, k_cl = get_nutrient_status(k_val, 'K')
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Nitrogen</div>
                    <div class="metric-value" style="color: {n_cl}">{n_val:.0f}</div>
                    <div class="metric-status" style="background: {n_cl}22; color: {n_cl}">{n_st}</div>
                </div>""", unsafe_allow_html=True)
            with sc2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Phosphorus</div>
                    <div class="metric-value" style="color: {p_cl}">{p_val:.0f}</div>
                    <div class="metric-status" style="background: {p_cl}22; color: {p_cl}">{p_st}</div>
                </div>""", unsafe_allow_html=True)
            with sc3:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">Potassium</div>
                    <div class="metric-value" style="color: {k_cl}">{k_val:.0f}</div>
                    <div class="metric-status" style="background: {k_cl}22; color: {k_cl}">{k_st}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.info("👆 Enter your soil NPK values and click **Analyze Soil Health** to see your report.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: Seasonal Calendar
# ═══════════════════════════════════════════════════════════════════════════════
def page_seasonal_calendar():
    st.markdown('<div class="section-header">📅 Indian Crop Calendar</div>', unsafe_allow_html=True)

    current_season = get_current_season()
    st.markdown(f"Current season: **{current_season}** (as of {datetime.now().strftime('%B %Y')})")

    for season_name, info in CROP_SEASONS.items():
        is_active = season_name == current_season
        active_class = "season-active" if is_active else ""
        active_marker = " — 📍 CURRENT" if is_active else ""

        st.markdown(f"""
        <div class="info-card" style="margin-bottom: 1rem; {'border-color: #22c55e; box-shadow: 0 0 20px rgba(34,197,94,0.1);' if is_active else ''}">
            <h4>
                <span class="season-badge {info['color']} {active_class}">{season_name}</span>
                &nbsp; {info['months']}{active_marker}
            </h4>
            <p style="margin-top: 0.75rem;">
                {"&nbsp;&nbsp;".join(f"{CROP_NUTRIENT_IMPACT.get(c, {}).get('emoji', '🌱')} <strong>{c}</strong>" for c in info['crops'])}
            </p>
        </div>""", unsafe_allow_html=True)

    # Interactive season timeline
    st.markdown("")
    st.markdown("##### 🗓️ Season × Crop Matrix")
    matrix_data = []
    for crop in CROP_NUTRIENT_IMPACT:
        row = {'Crop': f"{CROP_NUTRIENT_IMPACT[crop]['emoji']} {crop}"}
        for sn, si in CROP_SEASONS.items():
            row[sn] = "✅" if crop in si['crops'] else "—"
        matrix_data.append(row)
    st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)

    # Seasonal recommendations
    st.markdown("")
    st.markdown(f"##### 🌟 Best Crops for This Season ({current_season})")
    current_crops = CROP_SEASONS[current_season]['crops']
    cols = st.columns(min(len(current_crops), 5))
    for i, crop in enumerate(current_crops):
        info = CROP_NUTRIENT_IMPACT.get(crop, {})
        req = CROP_REQUIREMENTS.get(crop, {})
        with cols[i]:
            n_range = f"{req.get('N', (0,0))[0]}–{req.get('N', (0,0))[1]}"
            p_range = f"{req.get('P', (0,0))[0]}–{req.get('P', (0,0))[1]}"
            k_range = f"{req.get('K', (0,0))[0]}–{req.get('K', (0,0))[1]}"
            st.markdown(f"""
            <div class="info-card" style="text-align: center;">
                <div style="font-size: 2.5rem;">{info.get('emoji', '🌱')}</div>
                <h4 style="margin: 0.5rem 0 0.25rem;">{crop}</h4>
                <p style="font-size: 0.8rem; margin: 0;">N: {n_range}<br>P: {p_range}<br>K: {k_range}</p>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: NPK History Tracker
# ═══════════════════════════════════════════════════════════════════════════════
def page_npk_history():
    st.markdown('<div class="section-header">📈 NPK Reading History</div>', unsafe_allow_html=True)
    st.markdown("Track your soil readings across this session to observe trends.")

    # Initialize history
    if 'npk_history' not in st.session_state:
        st.session_state.npk_history = []

    col_input, col_chart = st.columns([1, 2], gap="large")

    with col_input:
        st.markdown("##### ➕ Add New Reading")
        with st.form("history_form"):
            h_n = st.number_input("N (mg/kg)", value=100.0, min_value=0.0, max_value=300.0, step=1.0, key="h_n")
            h_p = st.number_input("P (mg/kg)", value=50.0, min_value=0.0, max_value=200.0, step=1.0, key="h_p")
            h_k = st.number_input("K (mg/kg)", value=80.0, min_value=0.0, max_value=250.0, step=1.0, key="h_k")
            h_note = st.text_input("Note (optional)", placeholder="e.g., Field A, post-harvest")
            add_btn = st.form_submit_button("📌 Add Reading", use_container_width=True)

        if add_btn:
            st.session_state.npk_history.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'N': h_n, 'P': h_p, 'K': h_k,
                'note': h_note or f"Reading #{len(st.session_state.npk_history) + 1}"
            })
            st.success("Reading added!")

        if st.button("🗑️ Clear All History", use_container_width=True):
            st.session_state.npk_history = []
            st.rerun()

    with col_chart:
        history = st.session_state.npk_history
        if len(history) == 0:
            st.info("No readings yet. Add soil readings from the form or from the **Crop Prediction** page.")
        else:
            st.markdown(f"##### 📊 {len(history)} Reading(s)")

            # Trend chart
            hist_df = pd.DataFrame(history)
            if 'note' not in hist_df.columns:
                hist_df['note'] = [f"Reading #{i+1}" for i in range(len(hist_df))]
            hist_df['Reading'] = range(1, len(hist_df) + 1)

            fig = go.Figure()
            colors = {'N': '#22c55e', 'P': '#f59e0b', 'K': '#60a5fa'}
            for nut in ['N', 'P', 'K']:
                fig.add_trace(go.Scatter(
                    x=hist_df['Reading'], y=hist_df[nut],
                    mode='lines+markers',
                    name=nut,
                    line=dict(color=colors[nut], width=3),
                    marker=dict(size=8)
                ))
            fig.update_layout(
                title="NPK Trends Across Readings",
                xaxis_title="Reading #",
                yaxis_title="mg/kg",
                height=350,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#94a3b8'),
                legend=dict(orientation='h', y=1.15)
            )
            st.plotly_chart(fig, use_container_width=True)

            # History table
            display_df = hist_df[['Reading', 'time', 'N', 'P', 'K']].copy()
            if 'crop' in hist_df.columns:
                display_df['Predicted Crop'] = hist_df['crop']
            if 'note' in hist_df.columns:
                display_df['Note'] = hist_df['note']
            st.dataframe(display_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: About NPK
# ═══════════════════════════════════════════════════════════════════════════════
def page_about():
    st.markdown('<div class="section-header">📚 About NPK & Soil Science</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🔵 Nitrogen (N)</h4>
            <ul>
                <li>Essential for leaf & stem growth</li>
                <li>Promotes lush green foliage</li>
                <li>Key for protein synthesis</li>
                <li>Typical range: 40–200 mg/kg</li>
                <li>Deficiency → yellowing leaves</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🟠 Phosphorus (P)</h4>
            <ul>
                <li>Critical for root development</li>
                <li>Aids flower & fruit formation</li>
                <li>Energy transfer in plants</li>
                <li>Typical range: 20–120 mg/kg</li>
                <li>Deficiency → stunted growth</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="info-card">
            <h4>🟡 Potassium (K)</h4>
            <ul>
                <li>Regulates water uptake</li>
                <li>Boosts disease resistance</li>
                <li>Improves fruit quality</li>
                <li>Typical range: 40–200 mg/kg</li>
                <li>Deficiency → weak stems</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

    st.markdown("##### 🤖 Model Information")
    model_data = load_model()
    if model_data:
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f"""
            <div class="info-card">
                <h4>Model Details</h4>
                <ul>
                    <li><strong>Algorithm:</strong> Random Forest Classifier</li>
                    <li><strong>Accuracy:</strong> {model_data.get('accuracy', 0):.2%}</li>
                    <li><strong>Features:</strong> {', '.join(model_data.get('feature_names', []))}</li>
                </ul>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""
            <div class="info-card">
                <h4>Supported Crops</h4>
                <p>{', '.join(model_data.get('target_names', []))}</p>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    render_hero()

    # Load model
    model_data = load_model()

    # Sidebar navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")
        page = st.radio(
            "Choose a tool:",
            [
                "🌾 Crop Prediction",
                "🧪 NPK Additions",
                "🔄 Crop Rotation Advisor",
                "💚 Soil Health Score",
                "📅 Seasonal Calendar",
                "📈 NPK History",
                "📚 About NPK",
            ],
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown("##### 📊 Quick Stats")
        history = st.session_state.get('npk_history', [])
        st.metric("Readings This Session", len(history))
        current_season = get_current_season()
        st.metric("Current Season", current_season)
        st.metric("Crops Supported", len(CROP_NUTRIENT_IMPACT))

    # Page routing
    if model_data is None and page in ["🌾 Crop Prediction", "🧪 NPK Additions"]:
        st.error("Model not loaded. Please ensure the trained model file exists in the `models/` directory.")
        return

    if page == "🌾 Crop Prediction":
        page_crop_prediction(model_data)
    elif page == "🧪 NPK Additions":
        page_npk_additions(model_data)
    elif page == "🔄 Crop Rotation Advisor":
        page_crop_rotation()
    elif page == "💚 Soil Health Score":
        page_soil_health()
    elif page == "📅 Seasonal Calendar":
        page_seasonal_calendar()
    elif page == "📈 NPK History":
        page_npk_history()
    elif page == "📚 About NPK":
        page_about()


if __name__ == '__main__':
    main()
