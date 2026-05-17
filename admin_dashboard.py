"""
Results Analytics Dashboard - View aggregated quiz results
Run: streamlit run admin_dashboard.py
"""

import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime

st.set_page_config(
    page_title="Quiz Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Quiz Results Analytics")
st.caption("View aggregated results and user performance")

results_file = "results/quiz_results.csv"

if not os.path.exists(results_file):
    st.warning("No results yet. Run the quiz and publish results to populate this dashboard.")
    st.stop()

# Load results
try:
    df = pd.read_csv(results_file)
    
    st.subheader(f"📈 Summary ({len(df)} submissions)")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Submissions", len(df))
    col2.metric("Avg Score %", f"{df['Percentage'].mean():.1f}%")
    col3.metric("Highest Score %", f"{df['Percentage'].max():.1f}%")
    col4.metric("Unique Users", df['Email'].nunique())
    
    st.divider()
    
    # Score distribution
    st.subheader("Score Distribution")
    score_ranges = {
        "Excellent (80+%)": len(df[df['Percentage'] >= 80]),
        "Good (60-79%)": len(df[(df['Percentage'] >= 60) & (df['Percentage'] < 80)]),
        "Fair (40-59%)": len(df[(df['Percentage'] >= 40) & (df['Percentage'] < 60)]),
        "Needs Work (<40%)": len(df[df['Percentage'] < 40]),
    }
    st.bar_chart(score_ranges)
    
    # Recent submissions
    st.divider()
    st.subheader("Recent Submissions")
    df_recent = df.sort_values('Timestamp', ascending=False).head(10)
    st.dataframe(df_recent, use_container_width=True)
    
    # Download all results
    st.divider()
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download All Results as CSV",
        data=csv,
        file_name=f"all_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    
except Exception as e:
    st.error(f"Error loading results: {e}")
