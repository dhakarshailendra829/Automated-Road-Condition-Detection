import streamlit as st
import plotly.graph_objects as go
import time
import numpy as np

def animated_gauge(title, target_value=85, duration=2.5):
    """Enhanced animated gauge with ROADSAFE AI colors and smooth transitions"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=0,
            title={'text': f"<b>{title}</b>", 'font': {'size': 18, 'color': '#e2e8f0'}},
            number={'font': {'size': 34, 'color': '#38bdf8'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#64748b'},
                'bar': {'color': '#38bdf8'},
                'bgcolor': "#0f172a",
                'borderwidth': 2,
                'bordercolor': "#1e293b",
                'steps': [
                    {'range': [0, 30], 'color': '#10b981'},  
                    {'range': [30, 70], 'color': '#f59e0b'},  
                    {'range': [70, 100], 'color': '#ef4444'}  
                ],
                'threshold': {
                    'line': {'color': "#ef4444", 'width': 4},
                    'thickness': 0.75,
                    'value': target_value
                }
            }
        )
    )
    
    placeholder = st.empty()
    
    for i in range(101):
        value = target_value * (i / 100)
        fig.update_traces(value=value)
        placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(duration / 100)
    
    st.plotly_chart(fig, use_container_width=True)

def crew_dispatch_animation(crew_count=3):
    """Animated crew dispatch with truck movement and success indicators"""
    st.success("CREW DISPATCHED SUCCESSFULLY!")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    trucks = ["🚛", "🚚", "🚜"]
    for i in range(crew_count):
        status_text.markdown(f"Dispatching {trucks[i]} - Crew #{i+1}")
        for j in range(101):
            progress_bar.progress(j)
            time.sleep(0.02)
        st.balloons()
    
    status_text.markdown("✅ All crews dispatched and en route!")
    st.rerun()

def risk_alert_animation(severity="HIGH", confidence=92):
    """Pulsing risk alert with severity-based colors"""
    colors = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#10b981"}
    color = colors.get(severity, "#10b981")
    
    st.markdown(f"""
    <div style="text-align: center; padding: 20px; border-radius: 12px; 
                background: linear-gradient(45deg, {color}, rgba(255,255,255,0.1)); 
                animation: pulse 1.5s infinite;">
        <div style="font-size: 36px; font-weight: 800; color: {color}; margin-bottom: 10px;">
             {severity} RISK DETECTED
        </div>
        <div style="font-size: 24px; color: rgba(255,255,255,0.9);">
            Confidence: <strong>{confidence}%</strong>
        </div>
    </div>
    <style>
    @keyframes pulse {{
        0% {{ transform: scale(1); opacity: 1; box-shadow: 0 0 0 0 rgba(239,68,68,0.7); }}
        50% {{ transform: scale(1.05); opacity: 0.9; box-shadow: 0 0 20px 0 rgba(239,68,68,0.5); }}
        100% {{ transform: scale(1); opacity: 1; box-shadow: 0 0 0 0 rgba(239,68,68,0); }}
    }}
    </style>
    """, unsafe_allow_html=True)

def loading_scanner(duration=3.0):
    """Road infrastructure scanner loading animation"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    phases = [
        "Initializing Deep Learning Pipeline...",
        "Loading ResNet50 Feature Extractor...",
        "Applying PCA Dimensionality Reduction...",
        "Running ML Classification...",
        "Analysis Complete!"
    ]
    
    for i, phase in enumerate(phases):
        status_text.markdown(f"**{phase}**")
        for j in range(101):
            progress = (i * 100 + j) / (len(phases) * 100)
            progress_bar.progress(progress)
            time.sleep(duration / (len(phases) * 100))
    
    st.success("Road Infrastructure Analysis Ready!")

def metrics_reveal(metrics):
    """
    Animated metrics reveal for analysis results
    metrics: dict like {"Confidence": 92, "Issues": 2, "Risk": 75}
    """
    cols = st.columns(len(metrics))
    for i, (key, value) in enumerate(metrics.items()):
        with cols[i]:
            color = "#10b981" if value < 30 else "#f59e0b" if value < 70 else "#ef4444"
            
            placeholder = st.empty()
            for j in range(101):
                reveal_val = value * (j / 100)
                placeholder.metric(
                    label=key,
                    value=f"{reveal_val:.0f}{'#' if 'Issue' in key else '%'}",
                    delta=f"+{reveal_val:.0f}" if j == 100 else None,
                    delta_color="normal"
                )
                time.sleep(0.03)
