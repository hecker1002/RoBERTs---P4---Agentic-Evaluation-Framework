import os
import requests
import json

# ==============================================================================
# Step 1: Configuration
# ==============================================================================

# Your OpenRouter API key. This is the same key you used in the previous script.
OPENROUTER_API_KEY = "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"

# The URL for the OpenRouter chat completions API.
API_URL = "https://openrouter.ai/api/v1/chat/completions"

# The name of the LLM that will act as the judge. 
# "mistralai/mistral-7b-instruct" is a good, free option.
JUDGE_MODEL = "mistralai/mistral-7b-instruct"

# ==============================================================================
# Step 2: LLM Judge Function
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

    # Construct the clear, direct prompt for the judge LLM.
    # The prompt instructs the judge to act as a fair and objective evaluator.
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
        response.raise_for_status() # Raises an HTTPError for bad responses
        
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
# Step 3: Example Usage
# ==============================================================================

if __name__ == "__main__":
    # Define a sample question and an LLM response to test.
    sample_question = "Explain the concept of photosynthesis in simple terms."
    
    # You can paste any LLM response you want to judge here.
    sample_response = """
    Photosynthesis is how plants make food using light. They take in sunlight, water from their roots, and carbon dioxide from the air. Using the energy from the sun, they turn these ingredients into sugar, which is their food. A byproduct of this process is oxygen, which is released into the atmosphere.
    """
    
    # Define the criteria for the judge. Be as specific as possible.
    evaluation_criteria = [
        "Clarity and Simplicity (Is it easy for a layperson to understand?)",
        "Accuracy (Are all the facts correct?)",
        "Completeness (Does it cover all the key components of the process?)",
        "Conciseness (Is the response to the point without unnecessary detail?)"
    ]
    
    print("Sending evaluation request to the judge LLM...")
    
    # Call the function to get the scores.
    judgment = judge_response(sample_question, sample_response, evaluation_criteria)
    
    if judgment:
        print("\n--- JUDGMENT REPORT ---")
        for score_data in judgment.get('scores', []):
            criterion = score_data.get('criterion')
            score = score_data.get('score')
            justification = score_data.get('justification')
            print(f"\nCriterion: {criterion}")
            print(f"Score: {score}/10")
            print(f"Justification: {justification}")
        
        overall_score = judgment.get('overall_score')
        overall_justification = judgment.get('overall_justification')
        print(f"\n--- OVERALL SCORE: {overall_score}/10 ---")
        print(f"Overall Justification: {overall_justification}")
    else:
        print("Failed to get a judgment. Check the console for errors.")
