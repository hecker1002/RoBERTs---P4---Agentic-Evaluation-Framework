from __future__ import annotations
import re
import json
import csv
import statistics
import os
import requests
from typing import Dict, Any, List, Tuple
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# --- Required imports for BERT inference ---
# The existing feature calculation logic is preserved.

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

# ==============================================================================
# Step 1: LLM Judge Configuration (from your provided code)
# ==============================================================================

# The URL for the OpenRouter chat completions API.
OPENROUTER_API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "mistralai/mistral-7b-instruct"

# ==============================================================================
# Step 2: LLM Judge Function (from your provided code)
# ==============================================================================

def judge_response(question, response_to_judge, criteria):
    """
    Judges an LLM's response based on a set of provided criteria.

    Args:
        question (str): The original question or prompt given to the LLM.
        response_to_judge (str): The LLM's response that needs to be judged.
        criteria (list): A list of strings, where each string is a criterion
                         for judging the response (e.g., "Accuracy", "Clarity").

    Returns:
        dict or None: A dictionary with the scores and justifications for each
                      criterion, or None if an error occurred.
    """
    if not OPENROUTER_API_KEY or OPENROUTER_API_KEY == "YOUR_OPENROUTER_API_KEY":
        print("Error: Please replace 'YOUR_OPENROUTER_API_KEY' with your actual key.")
        return None

    judge_prompt = f"""
You are an impartial judge, evaluating the quality of a response from an AI assistant.
Your task is to score the AI assistant's response on a scale of 1 to 10 for each of the following criteria.

For each criterion, provide a single-sentence justification.

Criteria to Judge:
{'- '.join(criteria)}

Evaluation Task:
1. Carefully read the original question.
2. Read the AI assistant's response.
3. For each criterion listed above, give a score from 1 (lowest) to 10 (highest).
4. Provide a brief, one-sentence justification for each score.
5. Provide a single overall score for the response.

Output Format:
Return a JSON object with the following structure:
{{
  "scores": [
    {{"criterion": "...", "score": 0, "justification": "..."}},
    ...
  ],
  "overall_score": 0,
  "overall_justification": "..."
}}

---
Original Question:
{question}

---
AI Assistant's Response to Judge:
{response_to_judge}

---
Begin JSON Output:
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "user", "content": judge_prompt}
        ]
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        
        response_json = response.json()
        judge_response_text = response_json['choices'][0]['message']['content']
        
        # The judge's response is a JSON string, so we need to parse it.
        judge_output = json.loads(judge_response_text)
        
        return judge_output

    except requests.exceptions.RequestException as e:
        print(f"❌ API connection error: {e}")
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON response from the judge: {e}")
        print("Raw response from judge LLM:\n", judge_response_text)
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

    return None

# ==============================================================================
# Step 3: Existing Feature Calculation Functions (replicated for a self-contained script)
# ==============================================================================
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
def calculate_jaccard_similarity(s1:set,s2:set)->float:
    if not s1 or not s2: return 0.0
    return len(s1&s2)/len(s1|s2)
    
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
    "QA":["What is the capital of France?","Who invented the telephone?","When did World War II end?"],
    "Summarization":["Summarize this text.","Give me a TLDR of the passage.","Condense the following article."],
    "Reasoning":["Explain why the sky is blue.","Prove that triangles have 180 degrees.","Reason step by step about this math problem."]
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
    return best_domain if best_score>=0.35 else "Other"

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


# New function to generate features and format the input string for BERT
def get_inferred_features(prompt: str, response: str) -> Dict[str, Any]:
    """
    Generates all features from the provided prompt and response.

    Args:
        prompt (str): The original prompt text.
        response (str): The agent's response text.

    Returns:
        Dict[str, Any]: A dictionary of all calculated features.
    """
    return build_compact_feature_vector(prompt, response)

# ==============================================================================
# Step 4: New Final Report Generation Function
# ==============================================================================
def generate_final_report(question, response_text, bert_features, bert_prediction, llm_judgment):
    """
    Generates a combined report from BERT features, BERT prediction, and LLM judgment.
    """
    print("\n\n--- COMPREHENSIVE EVALUATION REPORT ---")
    print("-" * 50)
    print(f"**Original Question:** {question}")
    print(f"**Agent's Response:** {response_text.strip()}")
    print("-" * 50)

    # Report BERT Prediction
    bert_score = bert_prediction[0]['score']
    bert_label = bert_prediction[0]['label']
    print(f"\n### BERT Predicted Score: {bert_score:.4f} (Label: {bert_label})")
    
    # Analyze and reason about the BERT score based on features
    print("### Analysis of BERT Features")
    print("The BERT model's prediction is based on the following key features from the response:")
    
    # Identify high-impact features
    high_impact_features = {}
    if bert_features.get('extraneous_ratio', 0) > 0.5:
        high_impact_features['Extraneous Ratio'] = "A high ratio suggests a lot of content unrelated to the prompt, which can lower a factual score."
    if bert_features.get('keyword_recall', 0) < 0.5:
        high_impact_features['Keyword Recall'] = "Low recall indicates the response may not be addressing the core keywords of the prompt."
    if bert_features.get('semantic_drift', 0) > 0.2:
        high_impact_features['Semantic Drift'] = "High drift means the response's topic shifted from its beginning to its end, indicating a lack of coherence."
    if bert_features.get('contradiction_flag', 0) == 1:
        high_impact_features['Contradiction Flag'] = "An internal contradiction was detected, which is a strong indicator of a flawed or hallucinated response."

    if high_impact_features:
        for feature, reason in high_impact_features.items():
            print(f"- **{feature}:** {reason}")
    else:
        print("No significant negative features were detected. The response is generally aligned with the prompt.")
    
    # Report LLM Judge's detailed scores
    if llm_judgment:
        print("\n### LLM Judge's Detailed Report")
        for score_data in llm_judgment.get('scores', []):
            criterion = score_data.get('criterion')
            score = score_data.get('score')
            justification = score_data.get('justification')
            print(f"- **{criterion}**: {score}/10")
            print(f"  *Justification:* {justification}")
        
        overall_score = llm_judgment.get('overall_score')
        overall_justification = llm_judgment.get('overall_justification')
        print(f"\n--- LLM Judge's Overall Score: {overall_score}/10 ---")
        print(f"Overall Justification: {overall_justification}")
    else:
        print("\n❌ The LLM judge failed to provide a judgment. Please check the logs.")
    
    print("-" * 50)
    
def main():
    """
    Main function to perform inference using the fine-tuned BERT model and LLM judge.
    """
    # Use variables for paths to make it easier to change later.
    model_directory = "D:\\This Project\\Agentic Eval\\RoBERTs---P4---Agentic-Evaluation-Framework\\src\\BERT_config"
    
    # 1. Load the fine-tuned model and tokenizer
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_directory)
        tokenizer = AutoTokenizer.from_pretrained(model_directory)
        print("✅ Model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model or tokenizer from '{model_directory}'.")
        print(f"Please ensure the directory exists and contains all necessary files. Error: {e}")
        return

    # Create a pipeline for text classification
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
    
    # Corrected data structure
    dataset = [
        {
            "question": "What are some good places to visit in India?",
            "responses": {
                "Agent_1": {"response": "Hey, the SUM of 2 +3 =5 ."}
            }
        },
        {
            "question": "What is a good exercise?",
            "responses": {
                "Agent_2": {"response": "Running in the park can be a very good exercise nowadays"}
            }
        }
    ]
    
    if not dataset:
        print("No dataset provided to evaluate.")
        return

    # 2. Perform inference for each agent's response
    print("\n--- Running Comprehensive Evaluation ---")
    
    for item in dataset:
        question = item["question"]
        responses = item["responses"]

        for agent_name, agent_data in responses.items():
            response_text = agent_data["response"]
            
            # Get the features
            features = get_inferred_features(question, response_text)
            
            # Format the input for the BERT classifier
            input_prompt = f"{question}, {response_text}, "
            for key, value in features.items():
                input_prompt += f"{key}: {value}, "
            input_prompt = input_prompt.rstrip(', ')

            try:
                # Get the BERT prediction
                bert_prediction = classifier(input_prompt)
                
                # Define the criteria for the LLM judge
                evaluation_criteria = [
                    "Factual Accuracy (Is the information correct?)",
                    "Alignment with Prompt (Does it directly answer the question?)",
                    "Coherence (Is the response well-structured and easy to follow?)",
                    "Relevance of Information (Is all information pertinent to the prompt?)"
                ]
                
                # Get the LLM judge's report
                llm_judgment = judge_response(question, response_text, evaluation_criteria)

                # Generate and print the final report
                generate_final_report(question, response_text, features, bert_prediction, llm_judgment)

            except Exception as e:
                print(f" ❌ An error occurred during evaluation for agent '{agent_name}': {e}")
                
    print("\n--- Evaluation Complete ---")

def load_data(file_path):
    """
    Loads the labeled JSON dataset from the specified file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return None

if __name__ == "__main__":
    main()
