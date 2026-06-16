import json
import csv
import argparse
import time
from datetime import datetime
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

JD_KEYWORDS = [
    "embedding", "retrieval", "vector database", "ranking", "llm",
    "fine-tuning", "a/b test", "evaluation", "ndcg", "mrr",
    "pinecone", "milvus", "qdrant", "weaviate", "sentence-transformers",
    "machine learning", "product", "python", "search", "recommendation"
]

SERVICE_COMPANIES = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}

def is_honeypot(cand):
    # 1. "Marketing Manager" with all AI keywords
    title = cand.get("profile", {}).get("current_title", "").lower()
    if title and not any(kw in title for kw in ["engineer", "developer", "scientist", "ml", "ai", "data", "software"]):
        return True
    
    # 2. Expert proficiency with 0 duration
    skills = cand.get("skills", [])
    expert_zero_duration = 0
    for s in skills:
        if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0:
            expert_zero_duration += 1
    if expert_zero_duration >= 3:
        return True
    
    return False

def compute_behavioral_score(cand):
    signals = cand.get("redrob_signals", {})
    score = 0.0
    
    # Recruiter response rate (0 to 1) -> 0 to 10 points
    score += signals.get("recruiter_response_rate", 0.0) * 10
    
    # Notice period (lower is better, max 180) -> up to 5 points
    notice = signals.get("notice_period_days", 90)
    if notice <= 30:
        score += 5
    elif notice <= 60:
        score += 2
    else:
        score -= 5 # Penalty for long notice
        
    # Interview completion rate
    score += signals.get("interview_completion_rate", 0.0) * 5
    
    # Profile completeness
    score += (signals.get("profile_completeness_score", 0.0) / 100.0) * 5
    
    # GitHub activity
    gh = signals.get("github_activity_score", -1)
    if gh > 0:
        score += (gh / 100.0) * 10
        
    # Inactive penalty
    last_active = signals.get("last_active_date", "2020-01-01")
    if last_active < "2025-01-01":
        score -= 10
        
    return score

def extract_text_for_bm25(cand):
    text = []
    profile = cand.get("profile", {})
    text.append(profile.get("headline", ""))
    text.append(profile.get("summary", ""))
    
    for job in cand.get("career_history", []):
        text.append(job.get("title", ""))
        text.append(job.get("description", ""))
        
    for skill in cand.get("skills", []):
        text.append(skill.get("name", ""))
        
    return " ".join(text).lower()

def has_product_experience(cand):
    for job in cand.get("career_history", []):
        comp = job.get("company", "").lower()
        if comp and not any(sc in comp for sc in SERVICE_COMPANIES):
            return True
    return False

def generate_reasoning(cand, score):
    profile = cand.get("profile", {})
    title = profile.get("current_title", "Engineer")
    yoe = profile.get("years_of_experience", 0)
    signals = cand.get("redrob_signals", {})
    resp_rate = signals.get("recruiter_response_rate", 0.0)
    notice = signals.get("notice_period_days", 90)
    
    return f"{yoe} years exp as {title}. Strong semantic fit for the JD requirements. High behavioral signal (responds to {int(resp_rate*100)}% of recruiters, {notice}d notice)."

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    start_time = time.time()
    
    print("Pass 1: Streaming and filtering candidates...")
    candidates = []
    texts_for_bm25 = []
    
    # Pre-filter logic
    with open(args.candidates, 'r') as f:
        for line in f:
            if not line.strip(): continue
            cand = json.loads(line)
            
            # Hard filters based on JD
            yoe = cand.get("profile", {}).get("years_of_experience", 0)
            if yoe < 3: # Keep >3 to be safe
                continue
                
            if is_honeypot(cand):
                continue
                
            if not has_product_experience(cand):
                continue
                
            candidates.append(cand)
            texts_for_bm25.append(extract_text_for_bm25(cand))
            
    print(f"Filtered down to {len(candidates)} candidates. Time elapsed: {time.time() - start_time:.2f}s")
    
    print("Pass 2: BM25 Scoring...")
    tokenized_corpus = [doc.split() for doc in texts_for_bm25]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = " ".join(JD_KEYWORDS).split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    print("Pass 2: Behavioral Scoring...")
    behav_scores = [compute_behavioral_score(c) for c in candidates]
    
    # Normalize BM25 and Behavioral scores to sum them
    bm25_norm = np.array(bm25_scores)
    if bm25_norm.max() > 0:
        bm25_norm = bm25_norm / bm25_norm.max()
        
    behav_norm = np.array(behav_scores)
    if behav_norm.max() > 0:
        behav_norm = behav_norm / behav_norm.max()
        
    pass2_scores = bm25_norm * 0.6 + behav_norm * 0.4
    
    # Get top 1000 for Pass 3
    top_1000_idx = np.argsort(pass2_scores)[::-1][:1000]
    top_candidates = [candidates[i] for i in top_1000_idx]
    top_pass2_scores = pass2_scores[top_1000_idx]
    
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
    
    # Cosine similarity
    cos_sims = np.dot(cand_embs, jd_emb) / (np.linalg.norm(cand_embs, axis=1) * np.linalg.norm(jd_emb))
    
    # Final Score
    final_scores = top_pass2_scores * 0.3 + cos_sims * 0.7
    
    # Sort top 100 handling ties by candidate_id ascending
    scored_candidates = []
    for i in range(len(final_scores)):
        scored_candidates.append((round(float(final_scores[i]), 4), top_candidates[i]))
        
    scored_candidates.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    
    final_100_candidates = [x[1] for x in scored_candidates[:100]]
    final_100_scores = [x[0] for x in scored_candidates[:100]]
    
    print(f"Generating reasoning and saving to {args.out}...")
    
    with open(args.out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        
        for rank, (cand, score) in enumerate(zip(final_100_candidates, final_100_scores), 1):
            cid = cand.get("candidate_id")
            reasoning = generate_reasoning(cand, score)
            writer.writerow([cid, rank, round(float(score), 4), reasoning])
            
    print(f"Done! Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
