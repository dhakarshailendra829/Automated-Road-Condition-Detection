import streamlit as st
import plotly.graph_objects as go
import numpy as np
import time

def animated_gauge(target_value=85, duration=3):
    """🎯 Smooth animated gauge from 0 to target_value"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=0, number={'font': {'size': 36, 'color': '#10b981'}},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': '#10b981'},
               'steps': [{'range': [0, 50], 'color': '#059669'}, {'range': [50, 80], 'color': '#f59e0b'}, {'range': [80, 100], 'color': '#ef4444'}]}
    ))
    
    placeholder = st.empty()
    
    for i in range(101):
        value = target_value * (i/100)
        fig.update_traces(value=value)
        with placeholder.container():
            st.plotly_chart(fig, use_container_width=True)
        time.sleep(duration/100)
