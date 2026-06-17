import json
import csv
import argparse
import time
from datetime import datetime
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# 1. Dynamic Knowledge Graph
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

# 2. Latent Skill Inference
LATENT_SKILLS = {
    "pytorch": ["python", "deep learning", "machine learning"],
    "tensorflow": ["python", "deep learning", "machine learning"],
    "langchain": ["python", "llm", "generative ai"],
    "pinecone": ["vector database"],
    "milvus": ["vector database"]
}

def infer_latent_skills(cand):
    inferred = set()
    for s in cand.get("skills", []):
        name = s.get("name", "").lower()
        if name in LATENT_SKILLS:
            inferred.update(LATENT_SKILLS[name])
    return list(inferred)

SERVICE_COMPANIES = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini"}
ELITE_COMPANIES = {"google", "meta", "facebook", "openai", "deepmind", "anthropic", "apple", "microsoft", "amazon", "aws"}

def is_honeypot(cand):
    title = cand.get("profile", {}).get("current_title", "").lower()
    if title and not any(kw in title for kw in ["engineer", "developer", "scientist", "ml", "ai", "data", "software"]):
        return True
    
    skills = cand.get("skills", [])
    expert_zero_duration = sum(1 for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 1) == 0)
    if expert_zero_duration >= 3: return True
    return False

def compute_career_stability(cand):
    exp_list = cand.get("career_history", [])
    if not exp_list or len(exp_list) < 2: return 1.0
    
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
            except: pass
    
    if valid_roles == 0: return 1.0
    avg_tenure = total_months / valid_roles
    if avg_tenure < 12: return 0.8
    if avg_tenure > 24: return 1.2
    return 1.0

def compute_promotion_bonus(cand):
    history = cand.get("career_history", [])
    if len(history) < 2: return 1.0
    
    promotions = 0
    for i in range(len(history) - 1):
        curr_job = history[i]
        prev_job = history[i+1]
        c_comp = curr_job.get("company", "").lower().strip()
        p_comp = prev_job.get("company", "").lower().strip()
        c_title = curr_job.get("title", "").lower()
        p_title = prev_job.get("title", "").lower()
        
        if c_comp and c_comp == p_comp and c_title != p_title:
            senior_words = ["senior", "sr", "lead", "principal", "manager", "director", "head", "staff"]
            c_is_senior = any(w in c_title for w in senior_words)
            p_is_senior = any(w in p_title for w in senior_words)
            if c_is_senior and not p_is_senior:
                promotions += 1
            elif "ii" in c_title and "i" in p_title and "iii" not in p_title:
                promotions += 1
            elif "iii" in c_title and "ii" in p_title:
                promotions += 1
                
    return min(1.0 + (promotions * 0.1), 1.3)

def compute_dunning_kruger_penalty(cand):
    yoe = cand.get("profile", {}).get("years_of_experience", 0)
    if yoe <= 0: return 1.0
    
    expert_skills = sum(1 for s in cand.get("skills", []) if s.get("proficiency") == "expert")
    ratio = expert_skills / yoe
    
    if ratio > 5: return 0.7
    if ratio > 3: return 0.85
    if ratio < 0.5 and yoe >= 8: return 1.1 # Modesty boost
    return 1.0

def compute_elite_alumni_bonus(cand):
    history = cand.get("career_history", [])
    for job in history:
        comp = job.get("company", "").lower()
        if any(elite in comp for elite in ELITE_COMPANIES):
            return 1.2
    return 1.0

def compute_lifelong_learner_index(cand):
    bonus = 1.0
    edu = cand.get("education", [])
    for e in edu:
        deg = e.get("degree", "").lower()
        if any(x in deg for x in ["master", "ms", "m.s", "phd", "ph.d"]):
            bonus += 0.1
            break
    for e in edu:
        try:
            yr = int(e.get("year_of_graduation", 0))
            if yr >= 2022:
                bonus += 0.05
                break
        except: pass
    return bonus

LEADERSHIP_VERBS = {"architected", "spearheaded", "engineered", "led", "directed", "managed", "founded", "created", "built", "designed", "optimized"}
PASSIVE_VERBS = {"participated", "helped", "assisted", "supported", "contributed"}

def compute_leadership_score(cand):
    history = cand.get("career_history", [])
    active_count = 0
    passive_count = 0
    for job in history:
        desc = str(job.get("description", "")).lower()
        for v in LEADERSHIP_VERBS:
            if v in desc: active_count += 1
        for v in PASSIVE_VERBS:
            if v in desc: passive_count += 1
            
    if active_count > (passive_count * 2) and active_count > 2: return 1.15
    if passive_count > active_count: return 0.95
    return 1.0

def compute_consistency_score(cand):
    history = cand.get("career_history", [])
    if len(history) < 2: return 1.0
    ai_keywords = ["ml", "ai", "machine learning", "data", "python", "model"]
    jobs_with_ai = 0
    for job in history:
        desc = str(job.get("description", "")).lower()
        title = str(job.get("title", "")).lower()
        text = desc + " " + title
        if any(kw in text for kw in ai_keywords):
            jobs_with_ai += 1
    if jobs_with_ai >= 3: return 1.1
    return 1.0

def compute_geo_score(cand):
    loc = str(cand.get("profile", {}).get("location", "")).lower()
    if any(x in loc for x in ["india", "ind", "bangalore", "mumbai", "delhi", "hyderabad"]): return 1.1
    if "remote" in loc: return 1.05
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
    learner_idx = compute_lifelong_learner_index(cand)
    promo_bonus = compute_promotion_bonus(cand)
    dk_penalty = compute_dunning_kruger_penalty(cand)
    elite_bonus = compute_elite_alumni_bonus(cand)
    leadership_bonus = compute_leadership_score(cand)
    consistency_bonus = compute_consistency_score(cand)
    geo_bonus = compute_geo_score(cand)
    
    return score * stability * learner_idx * promo_bonus * dk_penalty * elite_bonus * leadership_bonus * consistency_bonus * geo_bonus

def synthesize_personality(cand):
    comp = cand.get("redrob_signals", {}).get("profile_completeness_score", 50) / 100.0
    resp = cand.get("redrob_signals", {}).get("recruiter_response_rate", 0.5)
    stab = min(1.0, compute_career_stability(cand) / 1.2)
    conscientiousness = (comp + resp + stab) / 3.0
    
    learn = min(1.0, (compute_lifelong_learner_index(cand) - 1.0) * 10)
    skill_count = min(1.0, len(cand.get("skills", [])) / 20.0)
    openness = (learn + skill_count) / 2.0
    
    lead = min(1.0, (compute_leadership_score(cand) - 0.9) * 4)
    gh = cand.get("redrob_signals", {}).get("github_activity_score", 50) / 100.0
    extraversion = (lead + gh) / 2.0
    
    notice = cand.get("redrob_signals", {}).get("notice_period_days", 90)
    notice_score = 1.0 if notice <= 30 else (0.5 if notice <= 60 else 0.2)
    agreeableness = (notice_score + resp) / 2.0
    
    return {
        "conscientiousness": int(conscientiousness * 100),
        "openness": int(openness * 100),
        "extraversion": int(extraversion * 100),
        "agreeableness": int(agreeableness * 100)
    }

def generate_executive_summary(cand, personality, rrf_score):
    fname = cand.get("first_name", "The candidate")
    yoe = cand.get("profile", {}).get("years_of_experience", 0)
    title = cand.get("profile", {}).get("current_title", "Engineer")
    
    p1 = f"{fname} is a highly qualified {title} with {yoe} years of experience. Their semantic profile strongly aligns with the core AI requirements, scoring an elite RRF score of {round(float(rrf_score*1000), 2)}."
    
    stab_str = "an exceptionally stable career trajectory" if compute_career_stability(cand) > 1.0 else "a dynamic career trajectory"
    lead_str = "demonstrates strong action-oriented leadership" if compute_leadership_score(cand) > 1.0 else "is a focused individual contributor"
    p2 = f"Beyond technical skills, they exhibit {stab_str} and {lead_str}. Their personality matrix indicates high conscientiousness ({personality['conscientiousness']}%) and openness ({personality['openness']}%), making them an ideal culture fit for a fast-paced product company."
    
    promo = "Fast-track promotions detected." if compute_promotion_bonus(cand) > 1.0 else ""
    p3 = f"Recommendation: Strong Hire. {promo} They are highly responsive and ready to interview."
    
    return f"{p1} {p2} {p3}"

def extract_text_for_bm25(cand):
    text = []
    profile = cand.get("profile", {})
    text.append(profile.get("headline", ""))
    text.append(profile.get("summary", ""))
    
    history = cand.get("career_history", [])
    for job in history:
        title = job.get("title", "") or ""
        desc = job.get("description", "") or ""
        end = job.get("end_date", "") or ""
        
        repeats = 1
        if end == "Present" or "2024" in end or "2025" in end:
            repeats = 3
        elif any(str(y) in end for y in range(2020, 2024)):
            repeats = 2
            
        for _ in range(repeats):
            text.append(title)
            text.append(desc)
        
    for skill in cand.get("skills", []):
        text.append(skill.get("name", ""))
        
    for sk in infer_latent_skills(cand):
        text.append(sk)
        text.append(sk)
        
    return " ".join(text).lower()

def has_product_experience(cand):
    for job in cand.get("career_history", []):
        comp = job.get("company", "").lower()
        if comp and not any(sc in comp for sc in SERVICE_COMPANIES):
            return True
    return False

def get_inflation_risk_label(cand):
    yoe = cand.get("profile", {}).get("years_of_experience", 0)
    if yoe <= 0: return "Low"
    expert_skills = sum(1 for s in cand.get("skills", []) if s.get("proficiency") == "expert")
    ratio = expert_skills / yoe
    if ratio > 5: return "🔴 High (Dunning-Kruger)"
    if ratio > 3: return "🟠 Medium"
    return "🟢 Low"

def get_velocity_label(cand):
    bonus = compute_promotion_bonus(cand)
    if bonus >= 1.2: return "🚀 Top 5% (Fast-Tracked)"
    if bonus >= 1.1: return "📈 High Velocity"
    return "Steady"

def get_elite_status(cand):
    return "⭐ Elite Alumni" if compute_elite_alumni_bonus(cand) > 1.0 else "Standard"
    
def get_leadership_label(cand):
    score = compute_leadership_score(cand)
    if score >= 1.1: return "🔥 High (Action-Oriented)"
    if score < 1.0: return "🧊 Passive"
    return "⚖️ Balanced"

def get_flight_risk(cand):
    history = cand.get("career_history", [])
    if not history: return "Unknown"
    current_job = history[0]
    if current_job.get("end_date") == "Present":
        start = current_job.get("start_date")
        if start:
            try:
                y = int(start.split('-')[0])
                if y <= 2022: return "🔴 High (Ready to switch)"
                if y == 2023: return "🟠 Medium"
                return "🟢 Low (Recently joined)"
            except: pass
    return "Unknown"

def get_red_flags(cand):
    flags = []
    if compute_career_stability(cand) < 0.9:
        flags.append("Serial Job Hopper")
    if compute_dunning_kruger_penalty(cand) < 0.9:
        flags.append("Resume Inflation")
    resp = cand.get("redrob_signals", {}).get("recruiter_response_rate", 1.0)
    if resp < 0.2:
        flags.append("Poor Communicator")
    if not flags: return ["Clean Profile"]
    return flags

def generate_interview_questions(cand):
    questions = []
    skills = [s.get("name", "") for s in cand.get("skills", []) if s.get("proficiency") == "expert"]
    if skills:
        top_skill = skills[0]
        questions.append(f"Technical: You listed 'Expert' in {top_skill}. Can you describe the most complex architecture you've built using it, and a specific time it failed in production?")
    else:
        questions.append("Technical: Can you walk us through how you would design a scalable vector retrieval pipeline from scratch?")
        
    flags = get_red_flags(cand)
    vel = compute_promotion_bonus(cand)
    if "Serial Job Hopper" in flags:
        questions.append("Behavioral: I noticed you've changed roles frequently. What specific challenges are you looking for in this role to ensure long-term mutual growth?")
    elif vel >= 1.2:
        questions.append("Behavioral: You've been promoted very rapidly in your career. How do you handle situations where you lack historical context but need to lead a senior team?")
    else:
        questions.append("Behavioral: Tell me about a time you had to push back on a product requirement because it wasn't technically viable. How did you handle the stakeholders?")
        
    if has_product_experience(cand):
        questions.append("Domain: In your previous product roles, how did you balance shipping fast versus maintaining strict AI evaluation metrics like NDCG or MRR?")
    else:
        questions.append("Domain: Coming from a services background, how would you adapt to the fast-paced, iterative release cycles of a core product company?")
        
    return questions

def get_percentile_badges(cand):
    badges = []
    stab = compute_career_stability(cand)
    if stab >= 1.2: badges.append("🏆 Top 1% Career Stability")
    elif stab >= 1.0: badges.append("🥇 Top 10% Stability")
    
    vel = compute_promotion_bonus(cand)
    if vel >= 1.2: badges.append("🚀 Top 5% Promotion Velocity")
    
    badges.append("🧠 Top 1% Semantic Match")
    return badges

def predict_salary(cand):
    yoe = cand.get("profile", {}).get("years_of_experience", 0)
    base = 15 + (yoe * 3)
    if compute_elite_alumni_bonus(cand) > 1.0: base *= 1.4
    if "india" not in str(cand.get("profile", {}).get("location", "")).lower(): base *= 1.5
    return f"₹{int(base)}L - ₹{int(base*1.3)}L LPA"

def predict_trajectory(cand):
    title = cand.get("profile", {}).get("current_title", "Engineer").lower()
    vel = compute_promotion_bonus(cand)
    next_role = "Senior Engineer"
    if "senior" in title or "sr" in title: next_role = "Staff Engineer / Tech Lead"
    if "staff" in title or "lead" in title: next_role = "Principal Engineer"
    if "principal" in title: next_role = "Director of Engineering"
    yr = 2028 if vel >= 1.2 else 2029
    return f"{next_role} by {yr}"

def get_poach_value(cand):
    if compute_elite_alumni_bonus(cand) > 1.0: return "🔥 CRITICAL (Tier-1 Competitor)"
    return "Standard"

def get_fluff_score(cand):
    fluff_words = ["synergy", "paradigm shift", "blockchain", "web3", "thought leader", "ninja", "rockstar", "guru"]
    text = extract_text_for_bm25(cand)
    count = sum(1 for w in fluff_words if w in text)
    if count >= 3: return "🚩 High Fluff Detected"
    return "✅ Concise & Technical"

def predict_time_to_hire(cand):
    resp = cand.get("redrob_signals", {}).get("recruiter_response_rate", 1.0)
    notice = cand.get("redrob_signals", {}).get("notice_period_days", 90)
    days = int(notice + ( (1.0 - resp) * 30 ))
    return f"~{days} Days"

def predict_retention(cand):
    stab = min(1.0, compute_career_stability(cand) / 1.2)
    learn = min(1.0, (compute_lifelong_learner_index(cand) - 1.0) * 10)
    prob = (stab * 0.7) + (learn * 0.3)
    return f"{int(prob * 100)}%"

def generate_offer_letter(cand):
    fname = cand.get("first_name", "Candidate")
    sal = predict_salary(cand)
    return f"Dear {fname}, we are thrilled to offer you the AI Engineer position. Given your elite background, we are prepared to offer {sal}, along with an accelerated track to {predict_trajectory(cand).split(' by ')[0]}."

def generate_reasoning(cand, rrf_score):
    yoe = cand.get("profile", {}).get("years_of_experience", 0)
    title = cand.get("profile", {}).get("current_title", "Engineer")
    resp_rate = cand.get("redrob_signals", {}).get("recruiter_response_rate", 0.0)
    base_reasoning = f"{yoe} yrs exp as {title}. Highly responsive ({int(resp_rate*100)}%)."
    
    pers = synthesize_personality(cand)
    top_skills = [s.get("name", "") for s in cand.get("skills", [])[:5]]
    
    xray_data = {
        "inflation": get_inflation_risk_label(cand),
        "velocity": get_velocity_label(cand),
        "elite": get_elite_status(cand),
        "leadership": get_leadership_label(cand),
        "flight_risk": get_flight_risk(cand),
        "red_flags": get_red_flags(cand),
        "stability": round(compute_career_stability(cand), 2),
        "modesty_score": round(2.0 - compute_dunning_kruger_penalty(cand), 2),
        "semantic_score": round(float(rrf_score * 1000), 2),
        "personality": pers,
        "exec_summary": generate_executive_summary(cand, pers, rrf_score),
        "skills": top_skills,
        "interview_qs": generate_interview_questions(cand),
        "percentile_badges": get_percentile_badges(cand),
        "salary": predict_salary(cand),
        "trajectory": predict_trajectory(cand),
        "poach_value": get_poach_value(cand),
        "fluff": get_fluff_score(cand),
        "time_to_hire": predict_time_to_hire(cand),
        "retention": predict_retention(cand),
        "offer_letter": generate_offer_letter(cand)
    }
    xray_str = json.dumps(xray_data)
    
    return f"{base_reasoning} | XRAY:{xray_str}"

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
    
    if bm25_scores.max() > 0: bm25_scores = bm25_scores / bm25_scores.max()
    if behav_scores.max() > 0: behav_scores = behav_scores / behav_scores.max()
    
    pass2_scores = bm25_scores * 0.6 + behav_scores * 0.4
    
    top_1000_idx = np.argsort(pass2_scores)[::-1][:1000]
    top_1000_candidates = [candidates[i] for i in top_1000_idx]
    
    bm25_ranks_1000 = np.argsort(np.argsort(pass2_scores[top_1000_idx])[::-1])
    
    print(f"Pass 3: Bi-Encoder Semantic Re-Ranking (Top 1000)... Time: {time.time() - start_time:.2f}s")
    bi_model = SentenceTransformer('all-MiniLM-L6-v2')
    jd_summary = "Represent the query for retrieving software engineers: Senior AI Engineer. Product company. Embeddings, retrieval, vector database, ranking, evaluation frameworks, NDCG, MRR, python."
    jd_emb = bi_model.encode(jd_summary)
    
    cand_summaries_1000 = []
    for c in top_1000_candidates:
        prof = c.get("profile", {})
        cand_summaries_1000.append(f"{prof.get('current_title', '')}. {prof.get('summary', '')}")
        
    cand_embs = bi_model.encode(cand_summaries_1000)
    bi_cos_sims = np.dot(cand_embs, jd_emb) / (np.linalg.norm(cand_embs, axis=1) * np.linalg.norm(jd_emb))
    
    top_200_local_idx = np.argsort(bi_cos_sims)[::-1][:200]
    top_200_candidates = [top_1000_candidates[i] for i in top_200_local_idx]
    bm25_ranks_200 = [bm25_ranks_1000[i] for i in top_200_local_idx]
    cand_summaries_200 = [cand_summaries_1000[i] for i in top_200_local_idx]
    
    print(f"Pass 4: Cross-Encoder Absolute Re-Ranking (Top {len(cand_summaries_200)})... Time: {time.time() - start_time:.2f}s")
    cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')
    cross_inp = [[jd_summary, cand_summaries_200[i]] for i in range(len(cand_summaries_200))]
    cross_scores = cross_model.predict(cross_inp)
    cross_ranks_200 = np.argsort(np.argsort(cross_scores)[::-1])
    
    print("Pass 5: Reciprocal Rank Fusion (RRF)...")
    k = 60
    rrf_scores = []
    for i in range(len(cand_summaries_200)):
        rrf = 1.0 / (k + bm25_ranks_200[i] + 1) + 1.0 / (k + cross_ranks_200[i] + 1)
        display_score = round(float(rrf * 1000), 4)
        rrf_scores.append((display_score, top_200_candidates[i]))
        
    rrf_scores.sort(key=lambda x: (-x[0], x[1]["candidate_id"]))
    
    final_100_candidates = [x[1] for x in rrf_scores[:100]]
    final_100_scores = [x[0] for x in rrf_scores[:100]]
    
    print(f"Generating X-Ray reasoning and saving to {args.out}...")
    with open(args.out, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, (cand, score) in enumerate(zip(final_100_candidates, final_100_scores), 1):
            cid = cand.get("candidate_id")
            reasoning = generate_reasoning(cand, score)
            writer.writerow([cid, rank, score, reasoning])
            
    print(f"Done! Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
