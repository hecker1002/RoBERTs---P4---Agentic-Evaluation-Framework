import os
import json
from typing import List, Dict, Any
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
# Assuming api_call.py contains your Google API key as a string variable: google_api
# from api_call import google_api 

# --- A placeholder for your API key for demonstration ---
# In your actual code, use your import: from api_call import google_api
google_api = "YOUR_GOOGLE_API_KEY" # Replace with your actual key or use the import
# --------------------------------------------------------

# Set up environment variable for the Gemini API key.
os.environ["GOOGLE_API_KEY"] = google_api 

# Ensure the GOOGLE_API_KEY is set.
if not os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") == "YOUR_GOOGLE_API_KEY":
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it or replace 'os.getenv' with your key.")

def get_llm(temperature: float = 0.7, model_name: str = "gemini-1.5-flash"):
    """
    Initializes and returns a ChatGoogleGenerativeAI LLM instance.
    """
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

def create_agents() -> Dict[str, LLMChain]:
    """
    Creates three distinct LLMChain "agents" with different personas.
    """
    llm = get_llm()

    # Agent 1: The Factual Reporter
    reporter_template = "You are a professional news reporter. Your goal is to provide a concise, factual, and unbiased answer to the following question. Avoid any conversational filler.\nQuestion: {question}"
    reporter_prompt = PromptTemplate.from_template(reporter_template)
    reporter_chain = LLMChain(llm=llm, prompt=reporter_prompt)

    # Agent 2: The Enthusiastic Storyteller
    storyteller_template = "You are an enthusiastic storyteller. Answer the question in a friendly, conversational, and engaging tone. Feel free to add interesting, but potentially unverified, anecdotes.\nQuestion: {question}"
    storyteller_prompt = PromptTemplate.from_template(storyteller_template)
    storyteller_chain = LLMChain(llm=llm, prompt=storyteller_prompt)

    # Agent 3: The Skeptical Analyst
    analyst_template = "You are a cautious and skeptical analyst. Answer the following question but be sure to qualify your answer with phrases like 'It is believed that...' or 'Sources suggest...'. Do not state anything as an absolute fact.\nQuestion: {question}"
    analyst_prompt = PromptTemplate.from_template(analyst_template)
    analyst_chain = LLMChain(llm=llm, prompt=analyst_prompt)

    return {
        "Factual Reporter": reporter_chain,
        "Enthusiastic Storyteller": storyteller_chain,
        "Skeptical Analyst": analyst_chain
    }

def main():
    """
    Main function to run agents, processing questions in chunks and saving progress.
    """
    OUTPUT_FILENAME = "results.json"
    CHUNK_SIZE = 3  # <-- Set your desired chunk size here

    all_questions = [
        "What is the capital of France?",
        "Explain the concept of quantum entanglement in simple terms.",
        "Who invented the telephone, and in what year?",
        "What is the process of photosynthesis?",
        "Describe the main plot of the book 'Dune'.",
        "What are the primary functions of the human liver?",
        "Explain the difference between nuclear fission and fusion."
    ]

    all_results = []
    if os.path.exists(OUTPUT_FILENAME):
        with open(OUTPUT_FILENAME, 'r') as f:
            try:
                all_results = json.load(f)
                print(f"✅ Loaded {len(all_results)} existing results from {OUTPUT_FILENAME}")
            except json.JSONDecodeError:
                print(f"⚠️ Warning: Could not parse {OUTPUT_FILENAME}. Starting fresh.")
    
    processed_questions = {result.get('question') for result in all_results}
    questions_to_process = [q for q in all_questions if q not in processed_questions]

    if not questions_to_process:
        print("\n✅ All questions have already been processed and saved. Nothing to do.")
        return

    print(f"\nTotal questions to process: {len(questions_to_process)} | Chunk Size: {CHUNK_SIZE}\n")

    agents = create_agents()

    for i, question in enumerate(questions_to_process):
        print(f"--- Processing Question {i+1}/{len(questions_to_process)}: '{question}' ---")

        current_result = {"question": question, "responses": {}}

        try:
            for agent_name, agent_chain in agents.items():
                print(f"--> Running agent: {agent_name}...")
                agent_response = agent_chain.invoke({"question": question})['text'].strip()
                current_result["responses"][agent_name] = agent_response
            
            # Add the completed result to our in-memory list
            all_results.append(current_result)
            
            # --- Conditional Save Logic ---
            # Save if the chunk is full OR if it's the last question in the list.
            is_last_question = (i == len(questions_to_process) - 1)
            if (i + 1) % CHUNK_SIZE == 0 or is_last_question:
                print("\n" + "="*25)
                print(f"CHUNK COMPLETE. Saving progress...")
                with open(OUTPUT_FILENAME, 'w') as f:
                    json.dump(all_results, f, indent=4)
                print(f"✅ Successfully saved {len(all_results)} total results to {OUTPUT_FILENAME}.")
                print("="*25 + "\n")

        except Exception as e:
            print(f"\n❌ An error occurred while processing question: '{question}'. Error: {e}")
            print("Attempting to save any progress made in the current chunk...")
            if all_results:
                with open(OUTPUT_FILENAME, 'w') as f:
                    json.dump(all_results, f, indent=4)
                print(f"✅ Progress saved. {len(all_results)} total results are stored.")
            print("Stopping execution. Run the script again to resume.")
            break

    print("--- All tasks completed. ---")

if __name__ == "__main__":
    main()