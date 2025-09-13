import os
import json
import tempfile
from typing import List, Dict, Any
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from api_call import google_api 
import random
from e4_data.ques import quesn_prompts

''' random response length for each prompt and each ai agent pair '''

# Set up environment variable for the Gemini API key.
os.environ["GOOGLE_API_KEY"] = google_api 

# Ensuring the GOOGLE_API_KEY is set.
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

# make the instance of the LLM model 
def get_llm(temperature = 0.7, model_name= "gemini-2.5-flash-preview-05-20"):
    """
    Initializes and returns a ChatGoogleGenerativeAI LLM instance.
    """
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

def create_agents() :
    """
    Creates three distinct LLMChain "agents" with different personas.
    The word count is now a dynamic variable in the prompt template.
    """
    llm = get_llm()

    # Agent 1: The Factual Reporter (simulates the "best" agent)
    reporter_template = """
    You are a good news reporter. Your goal is to provide a Concise, Factual, and Unbiased answer to the following question as you think is correct. Avoid any conversational filler.
    Strictly answer in {word_count} words.
    Question: {question}
    """
    reporter_prompt = PromptTemplate.from_template(reporter_template)
    reporter_chain = LLMChain(llm=llm, prompt=reporter_prompt)


    # Agent 2: The Enthusiastic Storyteller (simulates the "hallucinating" agent)
    storyteller_template = """
    You are an enthusiastic, carefree storyteller. Answer the question in a friendly, conversational, and super engaging tone. Feel free to add interesting, but potentially unverified, anecdotes for the sake of storytelling.
    Strictly answer in {word_count} words.
    Question: {question}
    """
    storyteller_prompt = PromptTemplate.from_template(storyteller_template)
    storyteller_chain = LLMChain(llm=llm, prompt=storyteller_prompt)


    # Agent 3: The Skeptical Analyst (simulates the "medium" agent)
    analyst_template = """
    You are a cautious skeptical and a very deep critical analyst. You try to NOT share Wrong info. Answer the following question but be sure to qualify your answer with phrases like "It is believed that..." or "Sources suggest...". 
    Do not state anything as an absolute fact unless you are very sure.
    Strictly answer in {word_count} words.
    Question: {question}
    """
    analyst_prompt = PromptTemplate.from_template(analyst_template)
    analyst_chain = LLMChain(llm=llm, prompt=analyst_prompt)
    
    return {
        "Factual Reporter": reporter_chain,
        "Enthusiastic Storyteller": storyteller_chain,
        "Skeptical Analyst": analyst_chain
    }

def main():
    OUTPUT_FILENAME = "results.json"
    CHUNK_SIZE = 5  # <-- Set your desired chunk size here

    all_questions = quesn_prompts

    all_results = []
    if os.path.exists(OUTPUT_FILENAME):
        try:
            with open(OUTPUT_FILENAME, 'r') as f:
                all_results = json.load(f)
            print(f"✅ Loaded {len(all_results)} existing results from {OUTPUT_FILENAME}")
        except json.JSONDecodeError:
            print(f"⚠ Warning: Could not parse {OUTPUT_FILENAME}. Starting fresh.")
    
    processed_questions = {result.get('question') for result in all_results}
    questions_to_process = [q for q in all_questions if q not in processed_questions]
    random.shuffle(questions_to_process) # Randomize the order

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
                # Generate a NEW random word count for each agent
                random_word_count = random.randint(20, 80)
                
                print(f"--> Running agent: {agent_name} with word count: {random_word_count}...")
                agent_response = agent_chain.invoke({"question": question, "word_count": random_word_count})['text'].strip()
                print(f"Agent Response: {agent_response}")
                print("-" * 20)

                current_result["responses"][agent_name] = agent_response
            
            # Add the completed result to our in-memory list
            all_results.append(current_result)
            
            # --- Conditional Save Logic with Atomic Write ---
            is_last_question = (i == len(questions_to_process) - 1)
            if (i + 1) % CHUNK_SIZE == 0 or is_last_question:
                print("\n" + "="*25)
                print(f"CHUNK COMPLETE. Saving progress...")
                
                # Create a temporary file
                temp_file = None
                try:
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir='.', suffix='.tmp', encoding='utf-8')
                    json.dump(all_results, temp_file, indent=4)
                    temp_file.flush()
                    temp_file.close()
                    os.replace(temp_file.name, OUTPUT_FILENAME)
                    print(f"✅ Successfully saved {len(all_results)} total results to {OUTPUT_FILENAME}.")
                except Exception as e:
                    print(f"❌ Error during atomic save: {e}")
                    if temp_file and os.path.exists(temp_file.name):
                        os.remove(temp_file.name)
                
                print("="*25 + "\n")

        except Exception as e:
            print(f"\n❌ An error occurred while processing question: '{question}'. Error: {e}")
            print("Attempting to save any progress made in the current chunk...")
            if all_results:
                # Use atomic write logic here as well for safety
                try:
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir='.', suffix='.tmp', encoding='utf-8')
                    json.dump(all_results, temp_file, indent=4)
                    temp_file.flush()
                    temp_file.close()
                    os.replace(temp_file.name, OUTPUT_FILENAME)
                    print(f"✅ Progress saved. {len(all_results)} total results are stored.")
                except Exception as save_e:
                    print(f"❌ Error during emergency save: {save_e}")
                    if temp_file and os.path.exists(temp_file.name):
                        os.remove(temp_file.name)
            print("Stopping execution. Run the script again to resume.")
            break

    print("--- All tasks completed. ---")

if __name__ == "__main__":
    main()
