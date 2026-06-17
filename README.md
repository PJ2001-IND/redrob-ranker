# 🚀 Redrob Candidate Ranker: The Omega Architecture
**Team:** Data Enthusiast  
**Leader:** Praasuk Jain  

This repository contains the official codebase for the **India Runs Data & AI Challenge**. It features a 9-stage Omni-Architecture that evaluates candidates using deep semantic NLP, psychological profiling, and behavioral heuristics.

## 📥 Setup Instructions
To safely reproduce our environment, you will need a machine with **Python 3.11+**, at least **16GB RAM**, and a **CPU**.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PJ2001-IND/redrob-ranker.git
   cd redrob-ranker
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the exact dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## ⚙️ How to Reproduce Our Submission
Place your `candidates.jsonl` file in the root directory.

To run the pipeline and generate our final CSV output, execute this **exact single command**:
```bash
python ranker.py --candidates ./candidates.jsonl --out ./team_data_enthusiast.csv
```
*Note: This command runs entirely locally on CPU, with the network off, and natively aggregates 100,000 candidates in under 30 seconds.*

## 🧠 System Architecture Overview
Our pipeline processes candidates through 9 stages:
1. **Hard Filtering:** Drops honeypots and unviable candidates.
2. **Behavioral Heuristics:** Calculates recruiter response rates and notice periods.
3. **Bi-Encoder Search:** Uses `all-MiniLM-L6-v2` for rapid semantic matching against the JD.
4. **Cross-Encoder Ranking:** Employs `ms-marco` for deep text interaction.
5. **Reciprocal Rank Fusion (RRF):** Mathematically sorts discrete ranks to output a monotonically decreasing score.
6. **Psychological Profiling:** Infers Big Five personality traits, Dunning-Kruger modesty, and Leadership bias.
7. **Omega Predictors:** Calculates Poach Value, Flight-Risk, Salary, and Time-to-Hire.
8. **UI & Visualization:** Live Streamlit dashboard featuring Interactive Graphviz Knowledge Graphs and Plotly Radar Charts.
9. **Generative Summarization:** Deterministically outputs factual Executive Summaries and Custom Interview Questions.

## 📊 Live Sandbox Demo
Interact with our live Hugging Face Streamlit Dashboard here:
[https://huggingface.co/spaces/praasukjain2001/redrob-ranker](https://huggingface.co/spaces/praasukjain2001/redrob-ranker)
