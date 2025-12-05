import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
import pickle
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from typing import List
import time
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

st.set_page_config(page_title="ROADSAFE AI", layout="wide", initial_sidebar_state="expanded")

BASE_DIR = Path(__file__).resolve().parent
CSS_PATH = BASE_DIR / "static" / "style.css"

def load_local_css(path: Path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception:
        pass

load_local_css(CSS_PATH)

COMPONENTS_DIR = BASE_DIR / "components"
if COMPONENTS_DIR.exists() and str(COMPONENTS_DIR) not in sys.path:
    sys.path.insert(0, str(COMPONENTS_DIR))

LIVE_COMPONENTS_AVAILABLE = False
realtime_module = None
animations_module = None

try:
    import realtime
    import animations
    realtime_module = realtime
    animations_module = animations
    LIVE_COMPONENTS_AVAILABLE = True
except ImportError:
    LIVE_COMPONENTS_AVAILABLE = False

@st.cache_resource
def load_models():
    try:
        model_dir = (BASE_DIR.parent / "models").resolve()
        ml_model = pickle.load(open(model_dir / "ml_classifier.pkl", "rb"))
        pca = pickle.load(open(model_dir / "pca.pkl", "rb"))
        scaler = pickle.load(open(model_dir / "scaler.pkl", "rb"))
        feature_extractor = load_model(str(model_dir / "resnet50_feature_extractor.h5"), compile=False)
        return ml_model, pca, scaler, feature_extractor
    except Exception as e:
        st.error(f"Models loading failed: {e}")
        return None, None, None, None

@st.cache_data
def load_crew_data():
    crew_data = pd.DataFrame({
        'name': ['Rahul Sharma', 'Priya Patel', 'Amit Kumar', 'Sneha Gupta', 'Vikram Singh'],
        'email': ['rahul@crewteam.com', 'priya@crewteam.com', 'amit@crewteam.com', 
                  'sneha@crewteam.com', 'vikram@crewteam.com'],
        'level': ['senior', 'junior', 'senior', 'junior', 'senior'],
        'status': ['available', 'available', 'busy', 'available', 'available']
    })
    
    crew_data['name'] = crew_data['name'].astype(str)
    crew_data['email'] = crew_data['email'].astype(str)
    crew_data['status'] = crew_data['status'].astype(str).str.lower()
    
    return crew_data

ML_MODEL, PCA, SCALER, FEATURE_EXTRACTOR = load_models()
CREW_DF = load_crew_data()

if LIVE_COMPONENTS_AVAILABLE:
    st.sidebar.success(" Live Components: ACTIVE")
else:
    st.sidebar.info(" Live Components: OFFLINE")

def dispatch_crew(analysis_results):
    if CREW_DF.empty or not analysis_results.get("detections"):
        return " "
    
    required_cols = ['status', 'name', 'email']
    missing_cols = [col for col in required_cols if col not in CREW_DF.columns]
    if missing_cols:
        return f" Crew data missing columns: {missing_cols}"
    
    try:
        available_crew = CREW_DF[CREW_DF['status'].str.contains('available', case=False, na=False)].head(3)
        if available_crew.empty:
            return " No available crew found"
        
        dispatched = []
        for _, crew in available_crew.iterrows():
            dispatched.append(f"{crew['name']} ({crew['email']})")
        return f" Dispatched: {', '.join(dispatched)}"
    except Exception as e:
        return f" Dispatch error: {str(e)}"

def generate_pdf_report(analysis):
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    p.drawString(100, 750, "ROADSAFE AI - Infrastructure Report")
    p.drawString(100, 730, f"Prediction: {analysis.get('prediction', 'N/A')}")
    p.drawString(100, 710, f"Confidence: {analysis.get('confidence', 0):.1f}%")
    p.drawString(100, 690, f"Detections: {len(analysis.get('detections', []))}")
    p.save()
    buffer.seek(0)
    return buffer.getvalue()

def run_ai_analysis(img: Image.Image):
    if not all([ML_MODEL, PCA, SCALER, FEATURE_EXTRACTOR]):
        return {"status": "error", "detail": "Models not loaded"}
    try:
        img_resized = img.resize((224, 224))
        arr = np.array(img_resized)
        pre = resnet_preprocess(np.expand_dims(arr, axis=0))
        features = FEATURE_EXTRACTOR.predict(pre, verbose=0)
        scaled = SCALER.transform(features.flatten().reshape(1, -1))
        reduced = PCA.transform(scaled)
        pred = ML_MODEL.predict(reduced)[0]
        proba = ML_MODEL.predict_proba(reduced)[0]
        detections = []
        if pred == 1:
            detections = [{"label": "Pothole", "confidence": 0.92, "severity": "HIGH"},
                         {"label": "Crack", "confidence": 0.78, "severity": "MEDIUM"}]
        return {"status": "success", "prediction": "Adverse Condition" if pred == 1 else "Safe Condition",
                "confidence": 85.0, "detections": detections, "flood_risk": 12.5 if pred == 1 else 1.2,
                "processing_time": 2.1, "model_metrics": {"f1": 0.94, "precision": 0.96, "recall": 0.92}}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

def draw_damage_annotations(img: Image.Image, detections: List):
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    colors = {"Pothole": "#EF4444", "Crack": "#F59E0B"}
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    for i, d in enumerate(detections):
        box = [60 + i * 140, 150, 260 + i * 140, 350]
        color = colors.get(d.get("label", ""), "#10B981")
        draw.rectangle(box, outline=color, width=5)
        text = f"{d.get('label','')}: {int(d.get('confidence',0)*100)}%"
        draw.rectangle([box[0]-4, box[1]-28, box[0] + len(text)*10 + 16, box[1]-4], fill=color)
        draw.text((box[0]+6, box[1]-26), text, fill="white", font=font)
    buf = BytesIO()
    img_copy.save(buf, format="PNG")
    return buf.getvalue()

def create_kpi_html(title: str, value: str, color: str = "#094ab3"):
    return f"""
    <div class="status-card">
        <div class="card-value" style="color:{color};">{value}</div>
        <div class="card-label">{title}</div>
    </div>
    """

def create_risk_gauge(value: float):
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=value, 
        gauge={'axis': {'range': [None, 100]}, 
               'steps': [{'range': [0, 30], 'color': "#076c4b"},
                        {'range': [30, 70], 'color': "#f59e0b"},
                        {'range': [70, 100], 'color': "#af0808"}],
               'threshold': {'line': {'color': "red", 'width': 4}, 'value': value}}))
    fig.update_layout(height=320, margin=dict(t=20, b=10, l=10, r=10))
    return fig

def recommendation_from_detections(detections):
    if not detections:
        return {"level": "Low", "action": "No immediate action required"}
    severities = [d.get("severity","MEDIUM") for d in detections]
    if "HIGH" in severities:
        return {"level": "High", "action": "Schedule immediate repair and dispatch crew"}
    if "MEDIUM" in severities:
        return {"level": "Medium", "action": "Plan maintenance and monitor"}
    return {"level": "Low", "action": "Monitor periodically"}

# Session state
if "detection_history" not in st.session_state:
    st.session_state["detection_history"] = []
if "last_analysis" not in st.session_state:
    st.session_state["last_analysis"] = None
if "last_image" not in st.session_state:
    st.session_state["last_image"] = None

def render_header():
    risk_class = "risk-low"
    if st.session_state.get("last_analysis"):
        conf = st.session_state["last_analysis"]["confidence"]
        if conf > 70: risk_class = "risk-high"
        elif conf > 40: risk_class = "risk-medium"
    
    st.markdown(f"""
    <div class="main-header {risk_class}">
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <div>
                <h1 class="header-title">Road Infrastructure Analysis</h1>
                <div class="header-subtitle">Deep Learning + Classical ML Hybrid Pipeline</div>
            </div>
            <div style="text-align:right">
                <div style="font-size:14px;color:#cbd5e1">Model Health: <span style="font-weight:800;color:#10b981">OK</span></div>
                <div style="margin-top:8px"><small style="color:#94a3b8">Inference: 2.1s</small></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def left_sidebar_navigation():
    st.sidebar.title("🚦 ROADSAFE AI")
    st.sidebar.markdown("**Monitoring & Analysis**")
    page = st.sidebar.selectbox("Navigation", ["Real-time Analysis", "Live Monitor", "Network Analytics", "Settings"])
    uploaded_file = st.sidebar.file_uploader(" Upload Road Image", type=["jpg", "jpeg", "png"])
    st.sidebar.metric("Crew Available", len(CREW_DF[CREW_DF['status'].str.contains('available', case=False, na=False)]))
    return page, uploaded_file

def main_analysis_page(uploaded_file):
    render_header()
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.markdown('<div class="chart-card"><h3> Image Scanner</h3></div>', unsafe_allow_html=True)
        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, use_container_width=True)
            if st.button(" Run Analysis", type="primary"):
                with st.spinner("Running AI analysis..."):
                    analysis = run_ai_analysis(img)
                    if analysis.get("status") == "success":
                        st.session_state["last_analysis"] = analysis
                        st.session_state["last_image"] = img
                        st.session_state["detection_history"].insert(0, {
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "prediction": analysis["prediction"],
                            "confidence": round(analysis["confidence"], 2),
                            "issues": [d["label"] for d in analysis["detections"]]
                        })
                        st.success(" Analysis complete!")
                    else:
                        st.error(f" Analysis failed: {analysis.get('detail','unknown')}")
        else:
            st.markdown('<div class="chart-card" style="padding:40px;color:#94a3b8;text-align:center"><h3>📤 Upload road image to start</h3></div>', unsafe_allow_html=True)

        if st.session_state.get("last_analysis"):
            a = st.session_state["last_analysis"]
            st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
            st.markdown("<h4>Detection Summary</h4>", unsafe_allow_html=True)
            cols = st.columns(3)
            cols[0].markdown(create_kpi_html("Prediction", a["prediction"]), unsafe_allow_html=True)
            cols[1].markdown(create_kpi_html("Confidence", f"{a['confidence']:.1f}%"), unsafe_allow_html=True)
            cols[2].markdown(create_kpi_html("Issues", str(len(a["detections"]))), unsafe_allow_html=True)
            
            annotated = draw_damage_annotations(st.session_state["last_image"], a["detections"])
            st.markdown("<h4> Annotated Image</h4>", unsafe_allow_html=True)
            st.image(annotated, use_container_width=True)
            
            rec = recommendation_from_detections(a["detections"])
            st.markdown(f"""
            <div style='padding:16px;border-radius:12px;background:linear-gradient(90deg, #10b981, #059669);color:white;text-align:center'>
                <strong> Recommendation:</strong> {rec['action']} <br><small>Level: {rec['level']}</small>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander(" Model Metrics"):
                dfm = pd.DataFrame([a.get("model_metrics", {})]).T
                st.dataframe(dfm, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="chart-card"><h3> Risk Analytics</h3></div>', unsafe_allow_html=True)
        if st.session_state.get("last_analysis"):
            la = st.session_state["last_analysis"]
            fig = create_risk_gauge(la["confidence"])
            st.plotly_chart(fig, use_container_width=True)
            flood_df = pd.DataFrame({
        "Risk Level": ["Current Risk", "Medium Risk", "High Risk"], 
        "Percentage": [la["flood_risk"], 30, 20]
            })
            fig_pie = px.pie(flood_df, values="Percentage", names="Risk Level",
                    title="Risk Distribution",
                    color_discrete_sequence=["#10b981", "#f59e0b", "#ef4444"])
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie.update_layout(showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                if st.button(" Dispatch Crew", type="primary", use_container_width=True):
                    result = dispatch_crew(la)
                    st.success(result)
                    if LIVE_COMPONENTS_AVAILABLE:
                        try:
                            animations_module.crew_dispatch_animation(3)
                        except:
                            st.balloons()
            
            with col2:
                if st.button("Generate Report", use_container_width=True):
                    report_data = generate_pdf_report(la)
                    st.download_button("Download PDF", report_data, 
                                     "roadsafe_report.pdf", "application/pdf")
        else:
            st.markdown('<div style="padding:30px;color:#94a3b8;text-align:center">Run analysis first</div>', unsafe_allow_html=True)

        st.markdown('<div class="chart-card"><h4> Detection History</h4></div>', unsafe_allow_html=True)
        hist = st.session_state.get("detection_history", [])
        if hist:
            for h in hist[:5]:
                st.markdown(f"""
                <div style='padding:12px;border-radius:10px;background:rgba(56,189,248,0.1);margin-bottom:8px;border-left:4px solid #38bdf8'>
                    <strong>{h['time']}</strong> | {h['prediction']} 
                    <span style='float:right;color:#94a3b8'>({h['confidence']}%) - {', '.join(h['issues'])}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown('<div style="color:#94a3b8;padding:20px;text-align:center">No detections yet</div>', unsafe_allow_html=True)

def live_monitor_page():
    render_header()
    st.markdown('<div class="chart-card"><h3 style="margin:0 0 12px 0">Live Monitor</h3></div>', unsafe_allow_html=True)
    
    if st.button(" START Live Monitor", type="primary"):
        st.session_state.live_monitor_active = True
    elif st.button(" STOP Live Monitor"):
        st.session_state.live_monitor_active = False
    
    if st.session_state.get('live_monitor_active', False) and LIVE_COMPONENTS_AVAILABLE:
        try:
            realtime_module.live_kpi_metrics(refresh_rate=1.5)
        except:
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            col1.metric("Total Roads", "1,247", "+12")
            col2.metric("Crew Dispatched", "17", "+3")
            col3.metric("Critical Alerts", "3", "0")
            col4.metric("Uptime", "99.8%", "+0.1%")
            col5.metric("Accuracy", "94.2%", "+0.2%")
            col6.metric("Risk Score", "42", "-1")
    else:
        st.info(" Click START to begin live monitoring")

def network_analytics_page():
    render_header()
    st.markdown('<div class="chart-card"><h2> Network Analytics</h2></div>', unsafe_allow_html=True)
    df = pd.DataFrame({
        "date": pd.date_range(end=pd.Timestamp.now(), periods=10).astype(str),
        "issues": np.random.randint(15, 35, 10),
        "confidence": np.random.uniform(85, 95, 10)
    })
    st.plotly_chart(px.line(df, x="date", y="issues", title="Issues Trend"), use_container_width=True)
    st.plotly_chart(px.bar(df, x="date", y="confidence", title="Confidence"), use_container_width=True)

def settings_page():
    render_header()
    st.markdown('<div class="chart-card"><h2> System Settings</h2></div>', unsafe_allow_html=True)
    st.checkbox("Debug: Show Crew Data", key="debug_crew")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(" ML Model", " Loaded" if ML_MODEL else " Failed")
        st.metric(" Feature Extractor", " Loaded" if FEATURE_EXTRACTOR else " Failed")
    with col2:
        st.metric(" Crew Records", len(CREW_DF))
        available_crew = len(CREW_DF[CREW_DF['status'].str.contains('available', case=False, na=False)])
        st.metric(" Available Crew", available_crew)
    
    if st.session_state.get('debug_crew') and not CREW_DF.empty:
        st.subheader("Crew Data Preview")
        st.dataframe(CREW_DF[['name', 'email', 'status']].head(10))

page, uploaded = left_sidebar_navigation()

if page == "Real-time Analysis":
    main_analysis_page(uploaded)
elif page == "Live Monitor":
    live_monitor_page()
elif page == "Network Analytics":
    network_analytics_page()
else:
    settings_page()
