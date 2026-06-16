import streamlit as st
import pandas as pd
import json
import time
import os
import subprocess

st.set_page_config(page_title="Redrob Ranker Sandbox", layout="wide")
st.title("Redrob Ranker Sandbox")
st.write("Upload a small `candidates.jsonl` sample to test the ranker pipeline.")

uploaded_file = st.file_uploader("Choose a candidates.jsonl file", type="jsonl")

if uploaded_file is not None:
    # Save uploaded file
    with open("temp_candidates.jsonl", "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.write("File uploaded successfully. Running ranking pipeline...")
    
    start_time = time.time()
    
    # Run the ranker pipeline via subprocess
    try:
        # Check if ranker.py exists
        if not os.path.exists("ranker.py"):
            st.error("ranker.py not found in the current directory.")
        else:
            result = subprocess.run(
                ["python", "ranker.py", "--candidates", "temp_candidates.jsonl", "--out", "temp_submission.csv"],
                capture_output=True,
                text=True
            )
            
            elapsed = time.time() - start_time
            
            if result.returncode == 0:
                st.success(f"Ranking completed successfully in {elapsed:.2f} seconds!")
                
                # Show CSV
                df = pd.read_csv("temp_submission.csv")
                st.dataframe(df)
                
                # Provide download button
                with open("temp_submission.csv", "rb") as file:
                    st.download_button(
                        label="Download Submission CSV",
                        data=file,
                        file_name="submission.csv",
                        mime="text/csv",
                    )
            else:
                st.error("Error running ranker pipeline.")
                st.code(result.stderr)
                
    except Exception as e:
        st.error(f"An exception occurred: {e}")
