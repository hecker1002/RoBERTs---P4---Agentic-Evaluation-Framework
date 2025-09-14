from __future__ import annotations
import re
import json
import csv
import statistics
from typing import Dict, Any, List, Tuple
from collections import Counter

# ------- Optional monkeypatch for older dependency quirks -------
try:
    import numpy as np  # noqa: F401
    if not hasattr(np, "ComplexWarning"):
        class ComplexWarning(RuntimeWarning): ...
        np.ComplexWarning = ComplexWarning  # type: ignore
except Exception:
    pass

# ------- Semantic model (MiniLM) -------
try:
    from sentence_transformers import SentenceTransformer, util
    SEM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    SEM_MODEL, util = None, None
    print("Warning: sentence-transformers failed. Falling back to lexical features. Error:", e)

# ------- Constants -------
STOP_WORDS = {
    'a','an','and','the','is','in','at','of','on','by','for','with',
    'about','as','to','from','it','this','that','i','you'
}
HEDGING_WORDS = {'maybe','might','could','possibly','probably','suggest','likely','may','seems'}
ASSUMPTION_PHRASES = {'assume','assuming',"let's assume",'suppose','in case','given that','presume','presuming'}
RECOMMENDATION_VERBS = {'recommend','suggest','should','advise','you should','i suggest','i recommend'}
IMPERATIVE_KEYWORDS = {'list','summarize','explain','compare','describe','provide','give','write','outline'}
PRONOUNS = {'he','she','they','it','them','his','her','their','this','that','those','these'}

# ------- Basic text utils -------
def preprocess_text(text: str) -> set:
    if not text: return set()
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    return {t for t in tokens if t not in STOP_WORDS}

def split_sentences(text: str) -> List[str]:
    if not text.strip(): return []
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

# ------- Prompt chunking & emphasis -------
def chunk_prompt(prompt: str) -> List[str]:
    if not prompt: return []
    pieces = re.split(r'[\n;]+', prompt)
    chunks = []
    for p in pieces:
        p = p.strip()
        if not p: continue
        if re.search(r'\b(?:'+'|'.join(IMPERATIVE_KEYWORDS)+r')\b', p, re.IGNORECASE) or re.search(r'\d+\.', p):
            subs = re.split(r',\s*', p)
            chunks.extend([s.strip() for s in subs if s.strip()])
        else:
            chunks.append(p)
    seen, out = set(), []
    for c in chunks:
        if c not in seen:
            seen.add(c); out.append(c)
    return out

def prompt_chunk_weights(chunks: List[str], prompt: str) -> List[float]:
    prompt_tokens = re.findall(r'\b[a-zA-Z0-9]+\b', prompt.lower())
    freq = Counter(prompt_tokens)
    weights = []
    for ch in chunks:
        w = 1.0
        if re.findall(r'\b[A-Z]{2,}\b', ch): w += 1.0
        if re.search(r'["\'].+?["\']', ch): w += 1.0
        ch_tokens = re.findall(r'\b[a-zA-Z0-9]+\b', ch.lower())
        rep_boost = sum((freq[t]-1) for t in ch_tokens if freq[t] > 1)
        if rep_boost > 0: w += min(2.0, rep_boost*0.25)
        weights.append(round(w,3))
    s = sum(weights) or 1.0
    return [round(w/s,4) for w in weights]

# ------- Semantic helpers -------
def semantic_sim(a: str, b: str) -> float:
    if not a or not b: return 0.0
    if SEM_MODEL is None:
        return round(calculate_jaccard_similarity(preprocess_text(a), preprocess_text(b)),3)
    emb = SEM_MODEL.encode([a,b], convert_to_tensor=True)
    return round(float(util.cos_sim(emb[0],emb[1]).item()),3)

def batch_similarity_matrix(a_list: List[str], b_list: List[str]) -> List[List[float]]:
    if not a_list or not b_list: return []
    if SEM_MODEL is None:
        return [[round(calculate_jaccard_similarity(preprocess_text(a),preprocess_text(b)),3) for b in b_list] for a in a_list]
    a_emb = SEM_MODEL.encode(a_list, convert_to_tensor=True)
    b_emb = SEM_MODEL.encode(b_list, convert_to_tensor=True)
    sim = util.cos_sim(a_emb,b_emb)
    return [[round(float(sim[i][j].item()),3) for j in range(sim.shape[1])] for i in range(sim.shape[0])]

# ------- Heuristic detectors -------
def extract_numbers(text: str) -> List[str]: return re.findall(r'\b\d+(?:\.\d+)?\b', text)
def detect_entity_candidates(text: str) -> List[str]:
    return [m.strip() for m in re.findall(r'\b[A-Z][a-zA-Z0-9\-]*(?:\s+[A-Z][a-zA-Z0-9\-]*)*\b', text)]
def contains_assumption(text: str) -> bool: return any(p in text.lower() for p in ASSUMPTION_PHRASES)
def contains_recommendation(resp: str) -> bool: return any(rv in resp.lower() for rv in RECOMMENDATION_VERBS)
def hedging_ratio(resp: str) -> float:
    sents = split_sentences(resp)
    if not sents: return 0.0
    return round(sum(1 for s in sents if any(h in s.lower() for h in HEDGING_WORDS))/len(sents),3)

# ------- Backtracking -------
def map_response_to_prompt_chunks(prompt_chunks: List[str], response_sentences: List[str],
                                   sim_threshold: float=0.45) -> Tuple[List[Dict[str, Any]],float,float]:
    mapping=[];
    if not response_sentences: return mapping,0.0,0.0
    if not prompt_chunks:
        for s in response_sentences:
            mapping.append({"sentence":s,"best_chunk_idx":None,"best_chunk":None,"sim":0.0,"is_orphan":True})
        return mapping,0.0,0.0
    sim_matrix = batch_similarity_matrix(response_sentences,prompt_chunks)
    mapped_flags=[False]*len(response_sentences); chunk_covered=[False]*len(prompt_chunks)
    for i,row in enumerate(sim_matrix):
        best_j = max(range(len(row)), key=lambda j: row[j]); best_sim=row[best_j]
        is_mapped = best_sim>=sim_threshold; mapped_flags[i]=is_mapped
        if is_mapped: chunk_covered[best_j]=True
        mapping.append({"sentence":response_sentences[i],"best_chunk_idx":int(best_j),
                         "best_chunk":prompt_chunks[best_j],"sim":round(best_sim,3),"is_orphan":not is_mapped})
    backtrackability=round(sum(mapped_flags)/len(mapped_flags),3)
    prompt_cov=round(sum(chunk_covered)/len(chunk_covered),3) if chunk_covered else 0.0
    return mapping, backtrackability, prompt_cov

# ------- Simple metrics -------
def calculate_jaccard_similarity(s1:set,s2:set)->float:
    if not s1 or not s2: return 0.0
    return len(s1&s2)/len(s1|s2)

def calculate_keyword_recall(prompt:str,response:str)->Dict[str,Any]:
    kws={t for t in re.findall(r'\b[a-zA-Z]{3,}\b',prompt.lower()) if t not in STOP_WORDS}
    rw=preprocess_text(response); found=sorted(list(kws&rw))
    return {"recall_percentage": round((len(found)/max(1,len(kws)))*100,2) if kws else 0.0,
             "prompt_keywords":sorted(list(kws)),"found_keywords":found}

def calculate_extraneous_content_ratio(prompt:str,response:str)->float:
    pw,rw=preprocess_text(prompt),preprocess_text(response)
    return round(len(rw-pw)/len(rw),3) if rw else 0.0

def detect_entity_mismatch(prompt:str,response:str)->Dict[str,Any]:
    return {"mismatch_count":len(set(detect_entity_candidates(response))-set(detect_entity_candidates(prompt))),
             "mismatched_entities":sorted(list(set(detect_entity_candidates(response))-set(detect_entity_candidates(prompt))))}

def detect_number_mismatch(prompt:str,response:str)->Dict[str,Any]:
    pnums,rnums=set(extract_numbers(prompt)),set(extract_numbers(response))
    return {"missing_numbers":sorted(list(pnums-rnums)),"extra_numbers":sorted(list(rnums-pnums)),
             "extra_numbers_count":len(rnums-pnums)}

def detect_internal_contradictions_lexical(resp:str)->Dict[str,Any]:
    sents=split_sentences(resp)
    if len(sents)<2: return {"contradiction_detected":False,"pair":None}
    neg_kw={"no","not","never","without","impossible","n't"}
    for i in range(len(sents)-1):
        a,b=sents[i],sents[i+1]
        if not (preprocess_text(a)&preprocess_text(b)): continue
        if any(n in a.lower() for n in neg_kw)!=any(n in b.lower() for n in neg_kw):
            return {"contradiction_detected":True,"pair":(a,b)}
    return {"contradiction_detected":False,"pair":None}

def calculate_sentence_variety_score(resp:str)->float:
    sents=split_sentences(resp)
    if len(sents)<2: return 5.0
    lengths=[len(re.findall(r'\b\w+\b',s)) for s in sents]
    mean=sum(lengths)/len(lengths); std=statistics.stdev(lengths) if len(lengths)>1 else 0
    return round(min(max((std/mean)*10,0),10),2)

def calculate_repetition_penalty(resp:str)->Dict[str,Any]:
    words=re.findall(r'\b\w+\b',resp.lower())
    if len(words)<2: return {"repetition_percentage":0.0,"repeated_bigrams":[]}
    bigrams=[(words[i],words[i+1]) for i in range(len(words)-1)]
    uniq=set(bigrams); rep_ratio=1-(len(uniq)/len(bigrams)) if bigrams else 0
    top_repeats=[bg for bg in uniq if bigrams.count(bg)>1][:3]
    return {"repetition_percentage":round(rep_ratio*100,2),"repeated_bigrams":top_repeats}

# ------- Creative proxies -------
def compute_instruction_coverage(prompt:str,response:str)->float:
    chunks=chunk_prompt(prompt); sents=split_sentences(response)
    _,_,cov=map_response_to_prompt_chunks(chunks,sents); return cov

def compute_assumption_risk(resp:str)->float:
    ass=1 if contains_assumption(resp) else 0
    ass+=len(re.findall(r'\bas you know\b|\bgiven that\b|\bsuppose\b',resp.lower()))
    denom=max(1,len(split_sentences(resp))); return round(min(1.0,ass/denom),3)

def compute_user_frustration_proxy(prompt:str,response:str,mapping:List[Dict[str,Any]])->float:
    sents=split_sentences(response);
    if not sents: return 0.0
    orphan=sum(1 for m in mapping if m.get("is_orphan"))/len(sents)
    rec=1 if contains_recommendation(response) and not re.search(r'\b(recommend|should|advice)\b',prompt,re.I) else 0
    extr=calculate_extraneous_content_ratio(prompt,response)
    return round(min(1.0,0.6*orphan+0.2*extr+0.2*rec),3)

def compute_overconfidence_score(prompt:str,response:str)->float:
    return round(min(1.0,calculate_extraneous_content_ratio(prompt,response)*(1-hedging_ratio(response))),3)

def compute_hallucination_risk(prompt:str,response:str)->float:
    extr=calculate_extraneous_content_ratio(prompt,response)
    ent=detect_entity_mismatch(prompt,response)['mismatch_count']
    num=detect_number_mismatch(prompt,response)['extra_numbers_count']
    mapping,_,_=map_response_to_prompt_chunks(chunk_prompt(prompt),split_sentences(response))
    orphan=sum(1 for m in mapping if m.get("is_orphan"))/max(1,len(split_sentences(response)))
    raw=0.4*extr+0.25*(ent/(ent+1))+0.2*(num/(num+1))+0.15*orphan
    return round(min(1.0,raw),3)



def pronoun_resolution_score(prompt:str,response:str)->float:
    sents=split_sentences(response);
    if not sents: return 1.0
    pronoun_count=0; unresolved=0; ptoks=preprocess_text(prompt)
    for sent in sents:
        for w in re.findall(r'\b[a-zA-Z]+\b',sent.lower()):
            if w in PRONOUNS:
                pronoun_count+=1
                if not (ptoks&preprocess_text(sent)): unresolved+=1
    return 1.0 if pronoun_count==0 else round(1-(unresolved/pronoun_count),3)

# ------- Domain classification (MiniLM prototypes) -------
DOMAIN_PROTOTYPES = {
    "QA": [
        "What is the capital of France?",
        "Who invented the telephone?",
        "When did World War II end?",
        "How do you make a chocolate cake?",
        "What are the symptoms of the common cold?",
        "Where is the Great Barrier Reef located?",
        "Is it possible to travel faster than light?",
        "Does this sentence contain a verb?"
    ],
    "Summarization": [
        "Summarize this text.",
        "Give me a TLDR of the passage.",
        "Condense the following article.",
        "Provide a brief overview of this document.",
        "Create a short abstract for the research paper.",
        "Please provide the main points of the presentation."
    ],
    "Reasoning": [
        "Explain why the sky is blue.",
        "Prove that triangles have 180 degrees.",
        "Reason step by step about this math problem.",
        "Deduce the cause of the fire based on the evidence.",
        "Analyze the arguments presented in this essay.",
        "Justify your solution to the logic puzzle."
    ]
}
DOMAIN_EMBS={}
if SEM_MODEL:
    for d,exs in DOMAIN_PROTOTYPES.items():
        DOMAIN_EMBS[d]=SEM_MODEL.encode(exs,convert_to_tensor=True)

def classify_domain(prompt:str)->str:
    if not SEM_MODEL: return _classify_domain_rules(prompt)
    emb=SEM_MODEL.encode([prompt],convert_to_tensor=True)
    best_score,best_domain=0.0,"Other"
    for d,ex_emb in DOMAIN_EMBS.items():
        sims=util.cos_sim(emb,ex_emb).cpu().numpy().flatten()
        avg=sims.mean()
        if avg>best_score: best_score,best_domain=avg,d
    return best_domain if best_score>=0.10 else "Other"

def _classify_domain_rules(prompt:str)->str:
    pl=prompt.lower()
    if any(q in pl for q in ["what","who","where","when","how","is","does"]): return "QA"
    if any(s in pl for s in ["summarize","summary","tl;dr","condense"]): return "Summarization"
    if any(r in pl for r in ["explain","reason","prove","why","step-by-step"]): return "Reasoning"
    return "Other"

# ------- Domain adherence -------
def domain_adherence(domain:str,prompt:str,response:str)->float:
    sents=split_sentences(response); words=preprocess_text(response)
    plen,rlen=len(preprocess_text(prompt)),len(words)
    if domain=="QA":
        return round(min(1.0,(plen/max(1,rlen))*0.8+(1/max(1,len(sents)))),3)
    if domain=="Summarization":
        ratio=rlen/max(1,plen); return round(1-abs(0.35-ratio),3)
    if domain=="Reasoning":
        conn=["because","therefore","hence","thus","so"]; has_conn=any(c in response.lower() for c in conn)
        chain_ok=len(sents)>=2; return round((0.5*has_conn+0.5*chain_ok),3)
    return 0.5

# ------- Builder -------
def build_compact_feature_vector(prompt:str,response:str)->Dict[str,Any]:
    sents=split_sentences(response); chunks=chunk_prompt(prompt)
    mapping,backtrackability,prompt_cov=map_response_to_prompt_chunks(chunks,sents)
    extr=calculate_extraneous_content_ratio(prompt,response)
    keyword_rec=calculate_keyword_recall(prompt,response)['recall_percentage']/100.0
    instruction_cov=compute_instruction_coverage(prompt,response)
    return {
        "alignment_balance":round(keyword_rec-extr,3),
        "keyword_recall":round(keyword_rec,3),
        "extraneous_ratio":extr,
        "semantic_alignment":semantic_sim(prompt,response),
        "semantic_drift":round(1-semantic_sim(sents[0],sents[-1]) if len(sents)>=2 else 0.0,3) if sents else 0.0,
        "backtrackability":backtrackability,
        "prompt_chunk_coverage":prompt_cov,
        "instruction_coverage":round(instruction_cov,3),
        "sentence_variety":calculate_sentence_variety_score(response),
        "repetition_penalty_pct":calculate_repetition_penalty(response)['repetition_percentage'],
        "contradiction_flag":int(detect_internal_contradictions_lexical(response)['contradiction_detected']),
        "pronoun_resolution_score":pronoun_resolution_score(prompt,response),
        "assumption_risk":compute_assumption_risk(response),
        "user_frustration_proxy":compute_user_frustration_proxy(prompt,response,mapping),
        "overconfidence":compute_overconfidence_score(prompt,response),
        "hallucination_risk":compute_hallucination_risk(prompt,response),
        "domain":classify_domain(prompt),
        "domain_adherence":domain_adherence(classify_domain(prompt),prompt,response),
    }

def build_full_diagnostic(prompt:str,response:str)->Dict[str,Any]:
    sents=split_sentences(response); chunks=chunk_prompt(prompt)
    mapping,backtrackability,prompt_cov=map_response_to_prompt_chunks(chunks,sents)
    compact=build_compact_feature_vector(prompt,response)
    diag={
        "prompt":prompt,"response":response,"prompt_chunks":chunks,
        "prompt_chunk_weights":prompt_chunk_weights(chunks,prompt),
        "sentence_mappings":mapping,"backtrackability":backtrackability,
        "prompt_coverage":prompt_cov,"keyword_recall":calculate_keyword_recall(prompt,response),
        "extraneous_content":{"extraneous_ratio":calculate_extraneous_content_ratio(prompt,response)},
        "entity_mismatch":detect_entity_mismatch(prompt,response),
        "number_mismatch":detect_number_mismatch(prompt,response),
        "sentence_variety_score":calculate_sentence_variety_score(response),
        "repetition":calculate_repetition_penalty(response),
        "contradiction":detect_internal_contradictions_lexical(response),
        "assumption_risk":compute_assumption_risk(response),
        "user_frustration_proxy":compute_user_frustration_proxy(prompt,response,mapping),
        "overconfidence":compute_overconfidence_score(prompt,response),
        "semantic_alignment":semantic_sim(prompt,response),
        "semantic_drift":compact["semantic_drift"],
        "compact_features":compact
    }
    return diag

# ------- IO -------
def save_results(results:List[Dict[str,Any]],json_path="eval_results_full.json",csv_path="eval_results_compact.csv"):
    with open(json_path,"w",encoding="utf-8") as f: json.dump(results,f,indent=2,ensure_ascii=False)
    rows=[]
    for r in results:
        compact=r.get("compact_features") or build_compact_feature_vector(r.get("prompt",""),r.get("response",""))
        row={"prompt":r.get("prompt",""),"response":r.get("response","")}
        for k,v in compact.items(): row[f"feat.{k}"]=json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v
        rows.append(row)
    if rows:
        with open(csv_path,"w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)

# ------- Main Execution Block (Modified to iterate through all data) -------
if __name__ == "__main__":
    # Load the results.json file
    try:
        with open("results.json", 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: 'results.json' file not found. Please ensure it's in the same directory.")
        data = []

    # This list will hold all the diagnostic results
    all_results = []
    
    # Iterate through each entry in the loaded data
    if data:
        print("Starting evaluation of all prompts and responses...")
        for entry in data:
            question = entry.get("question")
            responses = entry.get("responses", {})
            
            # Ensure the entry has a question and at least one response
            if question and responses:
                print(f"\nProcessing prompt: '{question}'")
                
                # Iterate through each persona's response for the current prompt
                for persona, response_text in responses.items():
                    print(f"  - Evaluating response from: {persona}")
                    
                    # Call the full diagnostic function for the current pair
                    diagnostic = build_full_diagnostic(question, response_text)
                    
                    # Add prompt and persona context to the diagnostic result
                    diagnostic['prompt'] = question
                    diagnostic['persona'] = persona
                    
                    # Append the complete result to our list
                    all_results.append(diagnostic)
        
        # Save all the results to JSON and CSV files
        if all_results:
            save_results(all_results)
            print("\nEvaluation complete!")
            print("Results saved to 'eval_results_full.json' and 'eval_results_compact.csv'")
        else:
            print("\nNo valid prompt-response pairs found to evaluate.")
    else:
        print("No data to process. Exiting.")
