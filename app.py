import streamlit as st
import pandas as pd
import json
import time
import os
import subprocess

st.set_page_config(page_title="AI Recruiter Dashboard | Redrob", layout="wide", page_icon="🤖")

st.markdown("""
<style>
.stApp { background-color: #f8f9fa; }
.metric-card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 5px solid #6c63ff; }
.cand-name { font-size: 24px; font-weight: bold; color: #1e1e24; margin-bottom: 0;}
.cand-title { font-size: 16px; color: #6c757d; margin-top: 0;}
.reasoning-box { background-color: #eef2ff; border-left: 3px solid #6c63ff; padding: 10px; border-radius: 5px; font-size: 14px; font-style: italic;}
.badge { background-color: #6c63ff; color: white; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 Redrob AI Recruiter Dashboard")
st.markdown("**Welcome to the Future of Hiring.** Upload `small_candidates.jsonl` to run the state-of-the-art Two-Stage NLP pipeline.")

uploaded_file = st.file_uploader("Upload Candidates (JSONL)", type="jsonl")

if uploaded_file is not None:
    with open("temp_candidates.jsonl", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    if st.button("🚀 Run AI Ranking Pipeline", type="primary"):
        with st.spinner("Initializing Pipeline (BM25 -> Bi-Encoder -> Cross-Encoder -> RRF)..."):
            start_time = time.time()
            result = subprocess.run(
                ["python", "ranker.py", "--candidates", "temp_candidates.jsonl", "--out", "temp_submission.csv"],
                capture_output=True, text=True
            )
            elapsed = time.time() - start_time
            
        if result.returncode == 0:
            st.success(f"Pipeline finished in {elapsed:.2f} seconds!")
            
            df = pd.read_csv("temp_submission.csv")
            
            # Load candidate details
            cand_dict = {}
            with open("temp_candidates.jsonl", "r") as f:
                for line in f:
                    if line.strip():
                        c = json.loads(line)
                        cand_dict[c["candidate_id"]] = c
                        
            st.markdown("### Top AI Matches")
            
            for index, row in df.head(10).iterrows():
                cid = row["candidate_id"]
                cand = cand_dict.get(cid, {})
                fname = cand.get("first_name", "Unknown")
                lname = cand.get("last_name", "")
                profile = cand.get("profile", {})
                title = profile.get("current_title", "Engineer")
                company = profile.get("current_company", "")
                
                with st.container():
                    st.markdown(f'''
                    <div class="metric-card">
                        <div style="display:flex; justify-content:space-between;">
                            <div>
                                <p class="cand-name">{fname} {lname} <span class="badge">Rank #{int(row['rank'])}</span></p>
                                <p class="cand-title">{title} @ {company}</p>
                            </div>
                            <div style="text-align:right;">
                                <h2>{row['score']:.1f}</h2>
                                <p style="color:gray; font-size:12px; margin-top:-15px;">RRF Score</p>
                            </div>
                        </div>
                        <div class="reasoning-box">
                            💡 <b>AI Reasoning:</b> {row['reasoning']}
                        </div>
                    </div>
                    <br>
                    ''', unsafe_allow_html=True)
                    
            with open("temp_submission.csv", "rb") as file:
                st.download_button("Download Full CSV Submission", data=file, file_name="team_PJ2001_IND.csv", mime="text/csv")
        else:
            st.error("Error running ranker pipeline.")
            st.code(result.stderr)
