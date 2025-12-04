import streamlit as st
import numpy as np
import time

def live_kpi_metrics():
    """🔥 LIVE updating KPI metrics - Auto refreshes every 2s"""
    col1, col2, col3, col4, col5 = st.columns(5)
    placeholders = [col1.empty(), col2.empty(), col3.empty(), col4.empty(), col5.empty()]
    
    for _ in range(50):  # Runs for ~100 seconds
        metrics = ["99.8%", "94.2%", "2.1s", "47", "99.99%"]
        trends = ["+0.2%", "+1.3%", "-0.3s", "+12", "+0.01%"]
        colors = ["#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#10b981"]
        
        for i, (ph, metric, trend, color) in enumerate(zip(placeholders, metrics, trends, colors)):
            # Add live variation
            if i == 2: metric = f"{2.1+np.random.uniform(-0.2,0.2):.1f}s"
            elif i == 4: metric = f"{99.99+np.random.uniform(-0.05,0.05):.2f}%"
            else: metric = f"{float(metric[:-1])+np.random.uniform(-1,2):.1f}%"
            
            with ph.container():
                st.markdown(f"""
                <div style='background:linear-gradient(145deg,#0f172a,#1e293b);border-radius:20px;padding:30px;text-align:center;border:2px solid {color};'>
                    <div style='font-size:2.2rem;font-weight:900;color:{color};'>{metric}</div>
                    <div style='font-size:1.1rem;color:#64748b;font-weight:600;margin-top:8px;'>Metric {i+1}</div>
                    <div style='font-size:0.95rem;color:{color};font-weight:700;margin-top:5px;'>{trend}</div>
                </div>
                """, unsafe_allow_html=True)
        
        time.sleep(2)
        st.experimental_rerun()
