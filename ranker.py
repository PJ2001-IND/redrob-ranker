import json
import csv
import argparse
import time
from datetime import datetime
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# 1. Dynamic Knowledge Graph (Query Expansion)
KNOWLEDGE_GRAPH = {
    "llm": ["llm", "gpt", "llama", "langchain", "claude", "transformers", "generative ai"],
    "vector database": ["vector database", "pinecone", "milvus", "qdrant", "weaviate", "chroma"],
    "machine learning": ["machine learning", "ml", "deep learning", "neural networks"],
    "python": ["python", "pandas", "numpy", "scikit", "flask", "fastapi"]
}

BASE_KEYWORDS = [
    "embedding", "retrieval", "ranking",
    "fine-tuning", "a/b test", "evaluation", "ndcg", "mrr",
    "product", "search", "recommendation"
]

def get_expanded_query():
    query = BASE_KEYWORDS.copy()
    for k, v in KNOWLEDGE_GRAPH.items():
        query.extend(v)
    return " ".join(query).split()

SERVICE_COMPANIES = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}

def is_honeypot(cand):
    title = cand.get("profile", {}).get("current_title", "").lower()
    if title and not any(kw in title for kw in ["engineer", "developer", "scientist", "ml", "ai", "data", "software"]):
        return True
    
    skills = cand.get("skills", [])
    expert_zero_duration = 0
    for s in skills:
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0:
            expert_zero_duration += 1
    if expert_zero_duration >= 3:
        return True
    return False

def compute_career_stability(cand):
    exp_list = cand.get("career_history", [])
    if not exp_list or len(exp_list) < 2: 
        return 1.0
    
    total_months = 0
    valid_roles = 0
    
    for exp in exp_list:
        start = exp.get("start_date")
        end = exp.get("end_date")
        if start and end and end != 'Present':
            try:
                sy, sm, _ = map(int, start.split('-'))
                ey, em, _ = map(int, end.split('-'))
                total_months += (ey - sy) * 12 + (em - sm)
                valid_roles += 1
            except:
                pass
    
    if valid_roles == 0: return 1.0
    avg_tenure = total_months / valid_roles
    
    if avg_tenure < 12: return 0.8  # Penalty for extreme job-hopping
    if avg_tenure > 24: return 1.2  # Boost for stability
    return 1.0

def compute_behavioral_score(cand):
    signals = cand.get("redrob_signals", {})
    score = 0.0
    score += signals.get("recruiter_response_rate", 0.0) * 10
    
    notice = signals.get("notice_period_days", 90)
    if notice <= 30: score += 5
    elif notice <= 60: score += 2
    else: score -= 5
        
    score += signals.get("interview_completion_rate", 0.0) * 5
    score += (signals.get("profile_completeness_score", 0.0) / 100.0) * 5
    
    gh = signals.get("github_activity_score", -1)
    if gh > 0: score += (gh / 100.0) * 10
        
    last_active = signals.get("last_active_date", "2020-01-01")
    if last_active < "2025-01-01": score -= 10
        
    stability = compute_career_stability(cand)
    # Multiply behavioral score by career stability index
    return score * stability

def extract_text_for_bm25(cand):
    text = []
    profile = cand.get("profile", {})
    text.append(profile.get("headline", ""))
    text.append(profile.get("summary", ""))
    
    history = cand.get("career_history", [])
    for i, job in enumerate(history):
        title = job.get("title", "")
        desc = job.get("description", "")
        # Recency Weighting: Double the weight of the most recent job
        if i == 0:
            text.append(title)
            text.append(desc)
        text.append(title)
        text.append(desc)
        
    for skill in cand.get("skills", []):
        text.append(skill.get("name", ""))
        
    return " ".join(text).lower()

def has_product_experience(cand):
    for job in cand.get("career_history", []):
        comp = job.get("company", "").lower()
        if comp and not any(sc in comp for sc in SERVICE_COMPANIES):
            return True
    return False

def generate_reasoning(cand, rrf_score, stability):
    profile = cand.get("profile", {})
    title = profile.get("current_title", "Engineer")
    yoe = profile.get("years_of_experience", 0)
    signals = cand.get("redrob_signals", {})
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    
    stab_str = "Stable tenure" if stability >= 1.0 else "High mobility"
    return f"{yoe} years exp as {title} ({stab_str}). High hybrid RRF score. Highly responsive ({int(resp_rate*100)}%)."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("Pass 1: Streaming & Filtering...")
    candidates = []
    texts_for_bm25 = []
    
    with open(args.candidates, 'r') as f:
        for line in f:
            if not line.strip(): continue
            cand = json.loads(line)
            
            yoe = cand.get("profile", {}).get("years_of_experience", 0)
            if yoe < 3: continue
            if is_honeypot(cand): continue
            if not has_product_experience(cand): continue
                
            candidates.append(cand)
            texts_for_bm25.append(extract_text_for_bm25(cand))
            
    print(f"Filtered down to {len(candidates)} candidates. Time elapsed: {time.time() - start_time:.2f}s")
    
    print("Pass 2: BM25 & Behavioral Scoring...")
    tokenized_corpus = [doc.split() for doc in texts_for_bm25]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = get_expanded_query()
    bm25_scores = np.array(bm25.get_scores(tokenized_query))
    
    behav_scores = np.array([compute_behavioral_score(c) for c in candidates])
    
    # Normalize
    if bm25_scores.max() > 0: bm25_scores = bm25_scores / bm25_scores.max()
    if behav_scores.max() > 0: behav_scores = behav_scores / behav_scores.max()
    
    # Combined Pass 2 Score
    pass2_scores = bm25_scores * 0.6 + behav_scores * 0.4
    
    # Top 1000 for Semantic
    top_1000_idx = np.argsort(pass2_scores)[::-1][:1000]
    top_candidates = [candidates[i] for i in top_1000_idx]
    
    # Keep their BM25 ranks within this top 1000 subset
    bm25_subset_scores = pass2_scores[top_1000_idx]
    bm25_ranks = np.argsort(np.argsort(bm25_subset_scores)[::-1]) # Rank 0 is best
    
    print(f"Top 1000 selected. Time elapsed: {time.time() - start_time:.2f}s")
    
    print("Pass 3: Semantic Re-Ranking...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    jd_summary = "Senior AI Engineer. Product company. Embeddings, retrieval, vector database, ranking, evaluation frameworks, NDCG, MRR, python."
    jd_emb = model.encode(jd_summary)
    
    cand_summaries = []
    for c in top_candidates:
        prof = c.get("profile", {})
        cand_summaries.append(f"{prof.get('current_title', '')}. {prof.get('summary', '')}")
        
    cand_embs = model.encode(cand_summaries)
    cos_sims = np.dot(cand_embs, jd_emb) / (np.linalg.norm(cand_embs, axis=1) * np.linalg.norm(jd_emb))
    
    semantic_ranks = np.argsort(np.argsort(cos_sims)[::-1]) # Rank 0 is best
    
    print("Pass 4: Reciprocal Rank Fusion (RRF)...")
    k = 60
    rrf_scores = []
    for i in range(1000):
        rrf = 1.0 / (k + bm25_ranks[i] + 1) + 1.0 / (k + semantic_ranks[i] + 1)
        rrf_scores.append((rrf, top_candidates[i]))
        
    # Sort top 100
    rrf_scores.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    
    final_100_candidates = [x[1] for x in rrf_scores[:100]]
    final_100_scores = [x[0] for x in rrf_scores[:100]]
    
    print(f"Generating reasoning and saving to {args.out}...")
    with open(args.out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cand, score) in enumerate(zip(final_100_candidates, final_100_scores), 1):
            cid = cand.get("candidate_id")
            stab = compute_career_stability(cand)
            reasoning = generate_reasoning(cand, score, stab)
            # Scale RRF score slightly for presentation, RRF is usually very small (e.g. 0.03)
            # Multiplying by 1000 makes it readable.
            display_score = round(float(score * 1000), 4)
            writer.writerow([cid, rank, display_score, reasoning])
            
    print(f"Done! Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
