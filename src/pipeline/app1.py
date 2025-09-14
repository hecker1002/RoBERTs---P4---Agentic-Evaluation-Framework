from __future__ import annotations
import streamlit as st
import pandas as pd
import re
import json
import statistics
import requests
from typing import Dict, Any, List, Tuple
from collections import Counter
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import altair as alt


try:
    from sentence_transformers import SentenceTransformer, util
    SEM_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    SEM_MODEL, util = None, None
    st.warning("Warning: sentence-transformers failed. Falling back to lexical features. Error: " + str(e))

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
# LLM Judge Configuration
# ==============================================================================

OPENROUTER_API_KEY = "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"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_MODEL = "mistralai/mistral-7b-instruct"

# ==============================================================================
# LLM Judge Function (MODIFIED)
# ==============================================================================
def judge_response(question, response_to_judge, criteria, bert_score_data):
    """
    Judges an LLM's response based on a set of provided criteria,
    including the BERT score.
    """
    if not OPENROUTER_API_KEY:
        st.error("Error: OpenRouter API key not found. Please add it to your Streamlit secrets.")
        return None

    bert_score = bert_score_data['score']
    bert_label = bert_score_data['label']

    judge_prompt = f"""
You are an impartial judge, evaluating the quality of a response from an AI assistant.
Your task is to score the AI assistant's response on a scale of 1 to 10 for each of the following criteria.

For each criterion, provide a single-sentence justification.

Criteria to Judge:
{' - '.join(criteria)}

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
BERT Predicted Score (Instruction-Following Score of Response with respect to UUser Prompt is): {bert_score:.4f}
BERT Predicted Label: {bert_label}

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

        # FIX: Use a regex to extract the JSON object to handle extraneous text
        json_match = re.search(r'\{.*\}', judge_response_text, re.DOTALL)
        if json_match:
            clean_json_string = json_match.group(0)
            judge_output = json.loads(clean_json_string)
            return judge_output
        else:
            # If no JSON object is found, raise an error
            raise ValueError("No valid JSON object found in the LLM response.")

    except requests.exceptions.RequestException as e:
        st.error(f" API connection error: {e}")
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f" Failed to parse JSON response from the judge: {e}")
        st.code(judge_response_text, language="json")
    except Exception as e:
        st.error(f" An unexpected error occurred: {e}")

    return None

# ==============================================================================
# Existing Feature Calculation Functions (replicated for a self-contained script)
# ==============================================================================
# --- Basic text utils ---
def preprocess_text(text: str) -> set:
    if not text: return set()
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
    return {t for t in tokens if t not in STOP_WORDS}

def split_sentences(text: str) -> List[str]:
    if not text.strip(): return []
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]

# --- Prompt chunking & emphasis ---
def chunk_prompt(prompt) -> List[str]:
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

# --- Semantic helpers ---
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

# --- Heuristic detectors ---
def extract_numbers(text: str) -> List[str]: return re.findall(r'\b\d+(?:\.\d+)?\b', text)
def detect_entity_candidates(text: str) -> List[str]:
    return [m.strip() for m in re.findall(r'\b[A-Z][a-zA-Z0-9\-]*(?:\s+[A-Z][a-zA-Z0-9\-]*)*\b', text)]
def contains_assumption(text: str) -> bool: return any(p in text.lower() for p in ASSUMPTION_PHRASES)
def contains_recommendation(resp: str) -> bool: return any(rv in resp.lower() for rv in RECOMMENDATION_VERBS)
def hedging_ratio(resp: str) -> float:
    sents = split_sentences(resp)
    if not sents: return 0.0
    return round(sum(1 for s in sents if any(h in s.lower() for h in HEDGING_WORDS))/len(sents),3)

# --- Backtracking ---
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

# --- Simple metrics ---
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

# --- Creative proxies ---
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

# --- Domain classification (MiniLM prototypes) ---

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
    return best_domain if best_score>=0.35 else "Other"

def _classify_domain_rules(prompt:str)->str:
    pl=prompt.lower()
    if any(q in pl for q in ["what","who","where","when","how","is","does"]): return "QA"
    if any(s in pl for s in ["summarize","summary","tl;dr","condense"]): return "Summarization"
    if any(r in pl for r in ["explain","reason","prove","why","step-by-step"]): return "Reasoning"
    return "Other"

# --- Domain adherence ---
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

# --- New function to provide a placeholder for spelling errors ---
def calculate_spelling_errors(response: str) -> int:
    """
    A simple placeholder function for spelling errors, as a full spell-checker
    is outside the scope of this script.
    """
    return 0 # Placeholder for now, could be a real heuristic in future versions

# --- New function to calculate all features required for BERT input ---
def get_all_features(prompt: str, response: str) -> Dict[str, Any]:
    """
    Generates a full dictionary of all required features from the prompt and response.
    """
    sents = split_sentences(response)
    chunks = chunk_prompt(prompt)
    mapping, _, _ = map_response_to_prompt_chunks(chunks, sents)
    domain_type = classify_domain(prompt)
    
    return {
        "prompt": prompt,
        "response": response,
        "user_frustration_proxy": compute_user_frustration_proxy(prompt, response, mapping),
        "overconfidence": compute_overconfidence_score(prompt, response),
        "semantic_alignment": semantic_sim(prompt, response),
        "semantic_drift": round(1-semantic_sim(sents[0], sents[-1]) if len(sents) >= 2 else 0.0, 3) if sents else 0.0,
        "keyword_recall": calculate_keyword_recall(prompt, response)['recall_percentage'] / 100.0,
        "domain": domain_type,
        "domain_adherence": domain_adherence(domain_type, prompt, response),
        "repetition_percentage": calculate_repetition_penalty(response)['repetition_percentage'],
        "contradiction_detected": detect_internal_contradictions_lexical(response)['contradiction_detected'],
        "assumption_risk": compute_assumption_risk(response),
        "sentence_variety_score": calculate_sentence_variety_score(response),
        "extraneous_ratio": calculate_extraneous_content_ratio(prompt, response),
        "spelling_errors": calculate_spelling_errors(response)
    }

# Fine-tuning input string generator
def bert_input_string(idx: int, data: pd.DataFrame) -> str:
    """
    Generates a rich and structured BERT input string for fine-tuning,
    incorporating new features and categorizing metrics with assigned weights.
    """
    # Fetching the row at the specified index
    row = data.iloc[idx]

    # Extracting all the features from the row
    prompt = row['prompt']
    response = row['response']
    
    # Core Metrics
    user_frustration_proxy = row['user_frustration_proxy']
    overconfidence = row['overconfidence']
    semantic_alignment = row['semantic_alignment']
    semantic_drift = row['semantic_drift']
    keyword_recall = row['keyword_recall']
    
    domain = row['domain']
    domain_adherence = row['domain_adherence']
    repetition_percentage = row['repetition_percentage']
    contradiction_detected = row['contradiction_detected']
    assumption_risk = row['assumption_risk']
    sentence_variety_score = row['sentence_variety_score']
    extraneous_ratio = row['extraneous_ratio']
    spelling_errors = row['spelling_errors']

   
    input_bert = f"""
[ANALYSIS PROTOCOL]
A comprehensive evaluation of a user-AI interaction is being conducted. The following text and associated metrics are provided for analysis.

[USER INSTRUCTION]
{prompt}

[AI RESPONSE]
{response}

[CONTEXTUAL FRAMEWORK]
- Interaction Domain: {domain}
- Domain Adherence: {domain_adherence:.2%}

[CORE QUALITY METRICS]
The following metrics describe the fundamental quality of the response.
- Semantic Alignment (Cohesion): {semantic_alignment:.2%}
- Semantic Drift (Deviation): {semantic_drift:.2%}
- User Frustration Proxy: {user_frustration_proxy:.2%}
- Overconfidence Score: {overconfidence:.2%}
- Keyword Recall: {keyword_recall:.2%}

[BEHAVIORAL & LINGUISTIC SIGNALS]
These metrics highlight potential issues and stylistic characteristics.
- Repetition Percentage: {repetition_percentage:.2%}
- Contradiction Detected: {'Yes' if contradiction_detected else 'No'}
- Assumption Risk Score: {assumption_risk:.2%}
- Sentence Variety Score: {sentence_variety_score:.2%}
- Extraneous Information Ratio: {extraneous_ratio:.2%}
- Spelling Error Count: {spelling_errors}

[FEATURE WEIGHTS]
The relative importance of each metric for model optimization is as follows:
- Semantic Alignment: 0.25
- Domain Adherence: 0.20
- Keyword Recall: 0.15
- Overconfidence Score: 0.10
- User Frustration Proxy: 0.10
- Assumption Risk: 0.10
- Other Metrics (Combined): 0.10

[TASK]
Considering all the information and the assigned weights, predict the final Instruction-Following Score for this interaction.
"""
    return input_bert

# --- NEW: Function to save evaluation results to a JSON file ---
def save_evaluation_results(data: Dict[str, Any]):
    """Appends a new evaluation result to a local JSON file."""
    try:
        with open("eval_results_full.json", "r+") as f:
            file_data = json.load(f)
            file_data.append(data)
            f.seek(0)
            json.dump(file_data, f, indent=2)
    except FileNotFoundError:
        with open("eval_results_full.json", "w") as f:
            json.dump([data], f, indent=2)
    except json.JSONDecodeError:
        with open("eval_results_full.json", "w") as f:
            json.dump([data], f, indent=2)

# --- NEW: Function to display visualization and leaderboards ---
def display_evaluation_trends():
    """Reads historical data and displays charts and leaderboards."""
    try:
        with open("eval_results_full.json", "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        
        if df.empty:
            st.warning("No historical data to display. Run some evaluations first!")
            return

        st.markdown("---")
        st.subheader("Evaluation Trends & Analysis")

        # Create a new column 'bert_score' for visualization
        df['bert_score'] = df['bert_prediction'].apply(lambda x: x['score'])
        df['llm_overall_score'] = df['llm_judgment'].apply(lambda x: x.get('overall_score'))

        # Time-based trend chart
        st.write("#### Score Trends Over Time")
        # trend_chart = alt.Chart(df.reset_index()).mark_line().encode(
        #     x=alt.X('index', title='Evaluation Run'),
        #     y=alt.Y('score', title='Score'),
        #     color=alt.Color('source', title='Score Source'),
        #     tooltip=['source', 'score']
        # ).transform_fold(
        #     ['bert_score', 'llm_overall_score'],
        #     as_=['source', 'score']
        # ).interactive()

        
        trend_chart = (
        alt.Chart(df.reset_index())
        .transform_fold(
            ['bert_score', 'llm_overall_score'],
            as_=['source', 'score']
        )
        .mark_line()
        .encode(
            x=alt.X('index', title='Evaluation Run'),
            y=alt.Y('score:Q', title='Score'),
            color=alt.Color('source:N', title='Score Source'),
            tooltip=['source:N', 'score:Q']
        )
        .interactive()
)

        st.altair_chart(trend_chart, use_container_width=True)

        # Correlation Heatmap
        st.write("#### Feature Correlation Heatmap")
        # Select relevant numerical columns
        features_df = df[['bert_score', 'llm_overall_score', 'semantic_alignment', 'keyword_recall', 'extraneous_ratio', 'repetition_percentage']].dropna()
        corr_matrix = features_df.corr().stack().reset_index()
        corr_matrix.columns = ['Feature 1', 'Feature 2', 'Correlation']
        
        heatmap = alt.Chart(corr_matrix).mark_rect().encode(
            x=alt.X('Feature 1', axis=None),
            y=alt.Y('Feature 2', axis=None),
            color=alt.Color('Correlation', scale=alt.Scale(scheme='viridis')),
            tooltip=['Feature 1', 'Feature 2', alt.Tooltip('Correlation', format=".2f")]
        ).properties(
            title='Feature Correlation'
        ).interactive()
        st.altair_chart(heatmap, use_container_width=True)

        # Average LLM scores
        st.write("#### Average LLM Judge Scores")
        # Extract and flatten the scores from the 'llm_judgment' column
        score_data = []
        for _, row in df.iterrows():
            if row['llm_judgment'] and 'scores' in row['llm_judgment']:
                for item in row['llm_judgment']['scores']:
                    score_data.append(item)
        
        if score_data:
            scores_df = pd.DataFrame(score_data)
            avg_scores = scores_df.groupby('criterion')['score'].mean().reset_index()
            bar_chart = alt.Chart(avg_scores).mark_bar().encode(
                x=alt.X('criterion', sort='-y', title='Criterion'),
                y=alt.Y('score', title='Average Score'),
                tooltip=['criterion', alt.Tooltip('score', title='Average Score', format=".2f")]
            ).properties(
                title='Average Scores by Criterion'
            ).interactive()
            st.altair_chart(bar_chart, use_container_width=True)

        # Leaderboard
        st.write("#### Top Responses Leaderboard")
        if 'llm_overall_score' in df.columns:
            leaderboard_df = df.sort_values(by='llm_overall_score', ascending=False)
            leaderboard_df['Rank'] = leaderboard_df['llm_overall_score'].rank(ascending=False, method='min').astype(int)
            leaderboard_df = leaderboard_df[['Rank', 'llm_overall_score', 'bert_score', 'prompt', 'response']]
            leaderboard_df = leaderboard_df.rename(columns={
                'llm_overall_score': 'LLM Overall Score',
                'bert_score': 'BERT Score',
                'prompt': 'Prompt',
                'response': 'Response'
            })
            st.dataframe(leaderboard_df, hide_index=True, use_container_width=True)

    except FileNotFoundError:
        st.warning("No historical evaluation data found. Run an evaluation to start collecting.")
    except Exception as e:
        st.error(f"❌ An error occurred while loading historical data: {e}")
        st.info("The `eval_results_full.json` file may be corrupted or malformed. Try clearing the file to start fresh.")

def generate_final_report(question, response_text, bert_features, bert_prediction, llm_judgment):
    """
    Generates a combined report from BERT features, BERT prediction, and LLM judgment.
    """
    st.markdown("---")
    st.subheader("Comprehensive Evaluation Report")
    st.write(f"**Original Question:** {question}")
    st.write(f"**Agent's Response:** {response_text.strip()}")
    st.markdown("---")

    col1, col2 = st.columns(2)
    # Report BERT Prediction
    bert_score = bert_prediction[0]['score']
    bert_label = bert_prediction[0]['label']
    with col1:
        st.metric("BERT Predicted Score", value=f"{bert_score:.4f}", help=f"The confidence score from the fine-tuned BERT model, with the predicted label: {bert_label}")
    
    # Report LLM Judge's overall score
    with col2:
        if llm_judgment:
            overall_score = llm_judgment.get('overall_score')
            st.metric("LLM Judge's Overall Score", value=f"{overall_score}/10", help=f"Overall justification: {llm_judgment.get('overall_justification')}")
    
    # Report LLM Judge's detailed scores
    if llm_judgment:
        st.markdown("### LLM Judge's Detailed Report")
        for score_data in llm_judgment.get('scores', []):
            criterion = score_data.get('criterion')
            score = score_data.get('score')
            justification = score_data.get('justification')
            st.write(f"- **{criterion}**: {score}/10")
            st.write(f"  *Justification:* {justification}")
    
    st.markdown("---")

def main():
    
    st.title("Agentic Evaluation Framework")
    st.sidebar.markdown("""
    ### About
    This application evaluates the quality of an LLM response based on a given prompt. It uses:
    - **BERT** to predict a score based on a set of calculated features.
    - An **LLM judge** to provide a qualitative, detailed report.
    """)
    st.sidebar.markdown("---")
    #st.sidebar.info("This app relies on an external API (OpenRouter) and a pre-trained BERT model. Make sure you have your API key configured.")
    
    # User inputs
    st.header("Enter Your Data")
    question = st.text_area("Original Prompt:", height=100)
    response_text = st.text_area("LLM Response to Evaluate:", height=200)
    
    # Add a unique key to the button to avoid caching issues
    if st.button("Run Evaluation", key="evaluate_button", use_container_width=True):
        if not question or not response_text:
            st.warning("Please enter both a prompt and a response to run the evaluation.")
        else:
            with st.spinner("Running evaluation..."):
                try:
                    # NOTE: This path is specific to a local system.
                    # It will need to be changed for deployment.
                    model_directory = "D:\\This Project\\Agentic Eval\\RoBERTs---P4---Agentic-Evaluation-Framework\\src\\BERT_config"
                    model = AutoModelForSequenceClassification.from_pretrained(model_directory)
                    tokenizer = AutoTokenizer.from_pretrained(model_directory)
                    st.success("✅ Model and tokenizer loaded successfully.")
                except Exception as e:
                    st.error(f"❌ Error loading model or tokenizer. Please ensure you have your fine-tuned BERT model correctly configured for deployment. Error: {e}")
                    return

                classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
                
                # Get all features required for the BERT input string
                features = get_all_features(question, response_text)
                
                # Create a DataFrame and generate the correctly formatted input
                df_features = pd.DataFrame([features])
                input_bert_formatted = bert_input_string(0, df_features)
                
                try:
                    # Get the BERT prediction with the correctly formatted input
                    bert_prediction = classifier(input_bert_formatted, truncation=True)
                    
                    # Extract BERT score and label
                    bert_score_data = {
                        "score": bert_prediction[0]['score'],
                        "label": bert_prediction[0]['label']
                    }

                    # Define the criteria for the LLM judge
                    evaluation_criteria = [
                        "Factual Accuracy (Is the information correct?)",
                        "Alignment with Prompt (Does it directly answer the question?)",
                        "Coherence (Is the response well-structured and easy to follow?)",
                        "Relevance of Information (Is all information pertinent to the prompt?)",
                        "Agreement with BERT Score (How well does the response's quality align with the provided BERT prediction?)"
                    ]
                    
                    # Get the LLM judge's report, passing the BERT data
                    llm_judgment = judge_response(question, response_text, evaluation_criteria, bert_score_data)

                    # Create a data dictionary to save
                    evaluation_data = {
                        "prompt": question,
                        "response": response_text,
                        "bert_features": features,
                        "bert_prediction": bert_prediction[0],
                        "llm_judgment": llm_judgment
                    }
                    save_evaluation_results(evaluation_data)

                    # Generate and print the final report
                    generate_final_report(question, response_text, features, bert_prediction, llm_judgment)

                except Exception as e:
                    st.error(f"❌ An error occurred during evaluation: {e}")
            
    # Display full diagnostic JSON in an expander
    if question and response_text:
        with st.expander("View Raw Diagnostic Data"):
            st.json(get_all_features(question, response_text))

    st.sidebar.markdown("---")
    st.sidebar.subheader("Evaluation Dashboard")
    # Button to display trends and leaderboard
    if st.sidebar.button("Show Evaluation Trends"):
        display_evaluation_trends()

if __name__ == "__main__":
    main()
