import numpy as np
from langchain_ollama import OllamaEmbeddings
 
from models.jd_parsor import parse_jd_with_llm
from models.resume_parsor import data_of_resume
 
# ------------------------- Config ------------------------- #
# Must match a model name shown by `ollama list`
# e.g. "nomic-embed-text", "mxbai-embed-large", "all-minilm"
EMBED_MODEL = "nomic-embed-text"
 
# Minimum cosine similarity to count a JD skill as "present" in the resume.
# Tune this per model (see the printed similarity table when you run the file).
SIM_THRESHOLD = 0.70
 
# How much each JD category is worth
CATEGORY_WEIGHT = {"must_have": 1.0, "intermediate": 0.7, "good_to_have": 0.5}
 
# How much credit a match earns depending on which resume tier it was found in
TIER_CREDIT = {"primary": 1.0, "secondary": 0.7, "tertiary": 0.5}
TIER_ORDER = ("primary", "secondary", "tertiary")
 
_embedder = OllamaEmbeddings(model=EMBED_MODEL)
 
 
# ------------------------- Helpers ------------------------- #
def _clean_list(value):
    """Coerce LLM output to a deduplicated list of lowercase strings."""
    if not value:
        return []
    if isinstance(value, str):
        value = value.split(",")
    seen, out = set(), []
    for item in value:
        s = str(item).strip().lower()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out
 
 
def embed(texts):
    """Embed a list of strings in one batch and L2-normalize the vectors,
    so a dot product equals cosine similarity."""
    if not texts:
        return np.empty((0, 0))
    vecs = np.array(_embedder.embed_documents(texts), dtype=float)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms
 
 
def _best_match(skill, skill_vec, pool_texts, pool_vecs):
    """Return (similarity, matched_text) for the closest skill in a pool."""
    if not pool_texts:
        return 0.0, None
    if skill in pool_texts:  # exact hit always wins
        return 1.0, skill
    sims = pool_vecs @ skill_vec
    i = int(np.argmax(sims))
    return float(sims[i]), pool_texts[i]
 
 
# ------------------------- Main scoring ------------------------- #
def calculate_match_score(JD):
    jd_skills = parse_jd_with_llm(JD)
    resume_skills = data_of_resume()
 
    # JD skills by category
    jd = {cat: _clean_list(jd_skills.get(cat)) for cat in CATEGORY_WEIGHT}
 
    # Resume skills by tier. Tools are added to the secondary tier only,
    # so a tool mention doesn't automatically count as a primary skill.
    tools = _clean_list(resume_skills.get("Tools"))
    resume = {
        "primary": _clean_list(resume_skills.get("Primary Skills")),
        "secondary": _clean_list(
            _clean_list(resume_skills.get("Secondary Skills")) + tools
        ),
        "tertiary": _clean_list(resume_skills.get("Tertiary Skills")),
    }
 
    # Embed everything once
    jd_vecs = {cat: embed(skills) for cat, skills in jd.items()}
    resume_vecs = {tier: embed(skills) for tier, skills in resume.items()}
 
    expected_score = 0.0
    score = 0.0
    details = []
 
    for cat, weight in CATEGORY_WEIGHT.items():
        for skill, vec in zip(jd[cat], jd_vecs[cat]):
            expected_score += weight
 
            credit, hit_tier, hit_skill = 0.0, None, None
            best_sim_any, best_skill_any = 0.0, None
 
            for tier in TIER_ORDER:
                sim, match = _best_match(skill, vec, resume[tier], resume_vecs[tier])
 
                if sim > best_sim_any:
                    best_sim_any, best_skill_any = sim, match
 
                if sim >= SIM_THRESHOLD:
                    credit = TIER_CREDIT[tier]
                    hit_tier, hit_skill = tier, match
                    break  # take the highest tier that matches
 
            score += weight * credit
            details.append(
                {
                    "jd_skill": skill,
                    "category": cat,
                    "matched_tier": hit_tier,
                    "matched_resume_skill": hit_skill,
                    "closest_resume_skill": best_skill_any,
                    "closest_similarity": round(best_sim_any, 3),
                    "points": round(weight * credit, 2),
                }
            )
 
    return expected_score, jd_skills, resume_skills, score, details
 
 
if __name__ == "__main__":
    job_desc = input("Enter the Job Description: ")
    expected_score, jd_skills, resume_skills, score, details = calculate_match_score(job_desc)
 
    print("\n-----Skills from Job Description-----\n")
    print(jd_skills)
 
    print("\n-----Skills from Resume -----\n")
    print(resume_skills)
 
    print("\n-----Per-skill matches (use this to tune SIM_THRESHOLD)-----\n")
    for d in details:
        status = d["matched_tier"] or "NO MATCH"
        print(
            f"[{d['category']:<12}] {d['jd_skill']:<25} -> {status:<10} "
            f"closest: {d['closest_resume_skill']} ({d['closest_similarity']})"
        )
 
    print("\n-----Expected Score-----\n")
    print(expected_score)
 
    print("\n-----Matching Score-----\n")
    print(round(score, 2))
 
    print("\n-----Match Percentage-----\n")
    percent = (score / expected_score * 100) if expected_score else 0.0
    print(np.round(percent, 2))



# from models.resume_parsor import resume_parsor, data_of_resume
# from models.jd_parsor import parse_jd_with_llm
# import numpy as np
# from langchain_ollama import OllamaEmbeddings


# def calculate_match_score(JD):

#     jd_skills = parse_jd_with_llm(JD)
    
#     expected_score = 0
#     must_have = [s.lower() for s in jd_skills["must_have"]]
#     intermediate =[s.lower() for s in jd_skills["intermediate"]]
#     good_to_have = [s.lower() for s in jd_skills["good_to_have"]]

#     expected_score = len(must_have) * 1 + len(intermediate) * 0.7 + len(good_to_have) * 0.5

#     #--------------------------------------------------#
#     # Extracting the Skills from Resume

#     resume_skills = data_of_resume()
#     tools = [s.lower() for s in resume_skills["Tools"]]
 
#     primary_skills = [s.lower() for s in resume_skills["Primary Skills"]] + tools
#     secondary_skills = [s.lower() for s in resume_skills["Secondary Skills"]] + tools
#     tertiary_skills = [s.lower() for s in resume_skills["Tertiary Skills"]] + tools

#     #--------------------------------------------------#

#     score = 0

#     for skill in must_have:
#         if skill in primary_skills:
#             score += 1
#         elif skill in secondary_skills:
#             score += 0.7
#         elif skill in tertiary_skills:
#             score += 0.5

#     for skill_i in intermediate:
#         if skill_i in primary_skills or skill_i in secondary_skills:
#             score += 1
#         elif skill_i in tertiary_skills:
#             score += 0.7

#     for skill_j in good_to_have:
#         if skill_j in primary_skills or skill_j in secondary_skills or skill_j in tertiary_skills:
#             score += 1

#     return expected_score, jd_skills, resume_skills, score
        
            




#     #--------------------------------------------------#

#     # def score_category(skills, weights):

#     #     if not skills:

#     #         return 0, []
        
#     #     matched = resume_parsor(resume_test, skills)

#     #     score = len(matched)/ len(skills)

#     #     return score * weights, matched
    
#     # must_score, must_matched = score_category(must_have, 0.5)
#     # int_score, int_matched = score_category(intermediate, 0.3)
#     # good_score, good_matched = score_category(good_to_have, 0.2)

#     # final_score = (must_score + int_score + good_score) * 100

#     # return {
#     #     "final_score": round(final_score, 2),
#     #     "must_have_matched": must_matched,
#     #     "intermediate_matched": int_matched,
#     #     "good_to_have_matched": good_matched
#     # }

# if __name__ == "__main__":
#     job_desc = input("Enter the Job Description: ")
#     expected_score, jd_skills, resume_skills, score = calculate_match_score(job_desc)

#     print("\n-----Skills from Job Description-----\n")
#     print(jd_skills)

#     print("\n-----Skills from Resume -----\n")
#     print(resume_skills)
    
#     print("\n-----Expected Score-----\n")
#     print(expected_score)    
    
#     print("\n-----Matching Score-----\n")
#     print(score)

#     print("\n-----Match Percentage-----\n")
#     print(np.round((score/expected_score)*100,2))