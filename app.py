import streamlit as st
import pandas as pd
import json
import time
import os
import subprocess
import plotly.graph_objects as go

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

st.title("🤖 Redrob AI Recruiter Dashboard (The Masterpiece)")
st.markdown("**Welcome to the Future of Hiring.** Upload `small_candidates.jsonl` to run the state-of-the-art Psychological AI pipeline.")

blind_mode = st.toggle("🎭 Enable Blind Audition Mode (Anti-Bias)", value=False)

uploaded_file = st.file_uploader("Upload Candidates (JSONL)", type="jsonl")

if uploaded_file is not None:
    with open("temp_candidates.jsonl", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    if st.button("🚀 Run AI Ranking Pipeline", type="primary"):
        with st.spinner("Initializing Pipeline (BM25 -> Bi-Encoder -> Cross-Encoder -> RRF -> Psych Profiling -> Graphing)..."):
            start_time = time.time()
            result = subprocess.run(
                ["python", "ranker.py", "--candidates", "temp_candidates.jsonl", "--out", "temp_submission.csv"],
                capture_output=True, text=True
            )
            elapsed = time.time() - start_time
            
        if result.returncode == 0:
            st.success(f"Pipeline finished in {elapsed:.2f} seconds!")
            
            df = pd.read_csv("temp_submission.csv")
            
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
                
                reasoning_full = str(row['reasoning'])
                xray_data = {"inflation": "Unknown", "velocity": "Unknown", "elite": "Unknown", "leadership": "Unknown"}
                reasoning_text = reasoning_full
                
                if " | XRAY:" in reasoning_full:
                    parts = reasoning_full.split(" | XRAY:")
                    reasoning_text = parts[0]
                    try:
                        xray_data = json.loads(parts[1])
                    except: pass
                
                if blind_mode:
                    display_fname = f"Candidate"
                    display_lname = f"Alpha-{str(cid)[-4:]}"
                    display_company = "Confidential Company"
                else:
                    display_fname = fname
                    display_lname = lname
                    display_company = company
                
                badges_html = "".join([f'<span class="badge" style="background-color:#28a745; margin-left:5px;">{b}</span>' for b in xray_data.get('percentile_badges', [])])
                
                with st.container():
                    st.markdown(f'''
                    <div class="metric-card">
                        <div style="display:flex; justify-content:space-between;">
                            <div>
                                <p class="cand-name">{display_fname} {display_lname} <span class="badge">Rank #{int(row['rank'])}</span>{badges_html}</p>
                                <p class="cand-title">{title} @ {display_company}</p>
                            </div>
                            <div style="text-align:right;">
                                <h2>{row['score']:.1f}</h2>
                                <p style="color:gray; font-size:12px; margin-top:-15px;">RRF Score</p>
                            </div>
                        </div>
                        <div class="reasoning-box">
                            💡 <b>AI Reasoning:</b> {reasoning_text}
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    with st.expander("🔍 Open AI X-Ray Vision & Knowledge Graph"):
                        col1, col2, col3 = st.columns([1, 1, 1])
                        
                        with col1:
                            st.markdown("**Psychological Metrics:**")
                            st.write(f"- **Resume Inflation Risk:** {xray_data.get('inflation', 'N/A')}")
                            st.write(f"- **Career Velocity:** {xray_data.get('velocity', 'N/A')}")
                            st.write(f"- **Company Tier:** {xray_data.get('elite', 'N/A')}")
                            st.write(f"- **Leadership Bias:** {xray_data.get('leadership', 'N/A')}")
                            st.write(f"- **Flight Risk (Poachability):** {xray_data.get('flight_risk', 'N/A')}")
                            
                            st.markdown("**Omega HR Predictors:**")
                            st.write(f"- **Est. Salary:** {xray_data.get('salary', 'N/A')}")
                            st.write(f"- **Career Trajectory:** {xray_data.get('trajectory', 'N/A')}")
                            st.write(f"- **Competitor Poach Value:** {xray_data.get('poach_value', 'N/A')}")
                            st.write(f"- **Time-to-Hire:** {xray_data.get('time_to_hire', 'N/A')}")
                            st.write(f"- **Retention Prob:** {xray_data.get('retention', 'N/A')}")
                            
                            st.markdown("**AI Red Flag Scanner:**")
                            flags = xray_data.get('red_flags', [])
                            for f in flags:
                                if "Clean" in f:
                                    st.write(f"✅ {f}")
                                else:
                                    st.write(f"🚩 {f}")
                            st.write(f"📝 **Resume Language:** {xray_data.get('fluff', 'N/A')}")
                            
                            st.markdown("**Generative AI Summary:**")
                            st.info(xray_data.get('exec_summary', 'N/A'))
                            
                            st.markdown("**Automated Draft Offer Letter:**")
                            st.success(xray_data.get('offer_letter', 'N/A'))
                            
                            st.markdown("**Targeted Interview Questions:**")
                            for q in xray_data.get('interview_qs', []):
                                st.write(f"🎙️ {q}")
                            
                        with col2:
                            categories = ['Semantic Match', 'Career Stability', 'Action Bias', 'Modesty', 'Velocity']
                            sem = min(100, max(20, xray_data.get('semantic_score', 50) * 10))
                            stab = min(100, (xray_data.get('stability', 1.0) / 1.2) * 100)
                            lead_val = 100 if "High" in xray_data.get('leadership', '') else (50 if "Balanced" in xray_data.get('leadership', '') else 20)
                            mod_val = min(100, xray_data.get('modesty_score', 1.0) * 100)
                            vel_val = 100 if "Top" in xray_data.get('velocity', '') else (80 if "High" in xray_data.get('velocity', '') else 50)
                            values = [sem, stab, lead_val, mod_val, vel_val]
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatterpolar(
                                r=values, theta=categories, fill='toself', name=f"{display_fname}'s Profile", line_color='#6c63ff'
                            ))
                            fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False, margin=dict(l=20, r=20, t=20, b=20), height=250)
                            st.plotly_chart(fig, use_container_width=True)
                            
                        with col3:
                            st.markdown("**Skill Knowledge Graph:**")
                            skills = xray_data.get('skills', [])
                            graph_code = f"""
                            digraph G {{
                              bgcolor="transparent"
                              node [shape=box, style=filled, fillcolor="#eef2ff", color="#6c63ff", fontname="Helvetica"]
                              edge [color="#6c63ff"]
                              "JD: AI Engineer" -> "{display_fname}"
                            """
                            for sk in skills:
                                if sk:
                                    sk_clean = str(sk).replace('"', '')
                                    graph_code += f'  "{display_fname}" -> "{sk_clean}"\n'
                            graph_code += "}"
                            st.graphviz_chart(graph_code)

                    st.write("")
                    
            with open("temp_submission.csv", "rb") as file:
                st.download_button("Download Full CSV Submission", data=file, file_name="team_PJ2001_IND.csv", mime="text/csv")
        else:
            st.error("Error running ranker pipeline.")
            st.code(result.stderr)
