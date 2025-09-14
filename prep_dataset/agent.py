import os
import json
import tempfile
from typing import List, Dict, Any
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from api_call import google_api 
import random

# Our own dir for SAVING the result Dataset
from e4_data.ques import quesn_prompts

'''
This script generates responses for a list of prompts using three different LLM agents.
It processes the prompts in the original order they appear in the 'quesn_prompts' list,
without any random shuffling.
'''

# Setting up environment variable for the Gemini ( FLASH) API key.
os.environ["GOOGLE_API_KEY"] = google_api 

# Ensuring the GOOGLE_API_KEY is set.
# Jsut a formality 
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

# making the instance of the LLM model 
def get_llm(temperature = 0.7, model_name= "gemini-2.5-flash-preview-05-20"):
    ''' Initializes and returns a ChatGoogleGenerativeAI LLM instance. '''
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

def create_agents() :
    '''Creates three distinct LLMChain "AI Agents" with different personas ( Behaviour ) for less stress on pipeline.
    PS -> The word count is a dynamic variable in the prompt template.
    '''
    llm = get_llm()

    '''Agent 1  ===> The Factual Reporter ''' 

    reporter_template = """
    You are a good news reporter. Your goal is to provide a Concise, Factual, and Unbiased answer to the following question as you think is correct. Avoid any conversational filler.
    Strictly answer in {word_count} words.
    Question: {question}
    """
    reporter_prompt = PromptTemplate.from_template(reporter_template)
    reporter_chain = LLMChain(llm=llm, prompt=reporter_prompt)


    '''Agent 2  ===> The Enthusiastic Storyteller '''

    storyteller_template = """
    You are an enthusiastic, carefree storyteller. Answer the question in a friendly, conversational, and super engaging tone. Feel free to add interesting, but potentially unverified, anecdotes for the sake of storytelling.
    Strictly answer in {word_count} words.
    Question: {question}
    """
    storyteller_prompt = PromptTemplate.from_template(storyteller_template)
    storyteller_chain = LLMChain(llm=llm, prompt=storyteller_prompt)


    '''Agent 3  ===> The Skeptical Analyst '''

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

    ''' A safety LOOP to kee saving the reponses ( evne if we run out of API limits )'''
    OUTPUT_FILENAME = "results.json"
    CHUNK_SIZE = 5  

    all_questions = quesn_prompts

    all_results = []

    if os.path.exists(OUTPUT_FILENAME):

        try:
            with open(OUTPUT_FILENAME, 'r') as f:
                all_results = json.load(f)

            print(f" Loaded {len(all_results)} existing results from {OUTPUT_FILENAME}")

        except json.JSONDecodeError:
            print(f"!!! Warning: Could not parse {OUTPUT_FILENAME}. Starting AGAIN ...")
    
    processed_questions = {result.get('question') for result in all_results}
    
    # Here ,we are Processing the questions in the original order without shuffling
    questions_to_process = [q for q in all_questions if q not in processed_questions]

    if not questions_to_process:
        print("All questions have already been processed and saved. Nothing to do.")
        return

    print(f"\nTotal questions to process: {len(questions_to_process)} | Chunk Size: {CHUNK_SIZE}\n")

    agents = create_agents()

    for i, question in enumerate(questions_to_process):
        print(f"--- Processing Question {i+1}/{len(questions_to_process)}: '{question}' ---")

        current_result = {"question": question, "responses": {}}

        try:
            for agent_name, agent_chain in agents.items():
                
                ''' generating a random word count between 20 and 80 for each agent's response ( for Produce More variablity and
                DO not allow BIAS in the SYSTEM )'''
                random_word_count = random.randint(20, 80)
                
                print(f"-- Running agent: {agent_name} with the current Word Count: {random_word_count}...")

                agent_response = agent_chain.invoke({"question": question, "word_count": random_word_count})['text'].strip()
                print(f"Agent Response: {agent_response}")
                print("-" * 20)

                current_result["responses"][agent_name] = agent_response

            # adding the COMPLETE result in memory Now .
            all_results.append(current_result)
            
            '''Conditional Save Logic with Atomic Write==>ATOMIC write means => Write to a temp file and then rename it to the target file'''
            is_last_question = (i == len(questions_to_process) - 1)
            if (i + 1) % CHUNK_SIZE == 0 or is_last_question:
                print("\n" + "="*25)
                print(f"CHUNK COMPLETE. Saving progress Now ... WAIT ....")
                
                # Create a temporary file ( for the atomic write )
                temp_file = None

                try:
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir='.', suffix='.tmp', encoding='utf-8')
                    json.dump(all_results, temp_file, indent=4)
                    temp_file.flush()
                    temp_file.close()
                    os.replace(temp_file.name, OUTPUT_FILENAME)
                    print(f" Successfully saved {len(all_results)} total results to {OUTPUT_FILENAME}.")
                except Exception as e:
                    print(f"xx Error during atomic save: {e}")
                    if temp_file and os.path.exists(temp_file.name):
                        os.remove(temp_file.name)
                
                print("="*25 + "\n")

        except Exception as e:
            print(f"xx An error occurred while processing question: '{question}'. Error: {e}")
            print("Attempting to save any progress made in the current chunk...")
            if all_results:
                # Use atomic write logic here as well for safety
                try:
                    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir='.', suffix='.tmp', encoding='utf-8')
                    json.dump(all_results, temp_file, indent=4)
                    temp_file.flush()
                    temp_file.close()
                    os.replace(temp_file.name, OUTPUT_FILENAME)
                    print(f".. Progress saved. {len(all_results)} Total results are stored.")
                except Exception as save_e:
                    print(f"xx Error during emergency save: {save_e}")
                    if temp_file and os.path.exists(temp_file.name):
                        os.remove(temp_file.name)

            print("Stopping execution. If we want to resume ,we will  run the script AGAIN.")
            break

    print(" All tasks completed ")

if __name__ == "__main__":
    main()
