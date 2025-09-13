import os
import json
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
    You are a professional news reporter. Your goal is to provide a concise, factual, and unbiased answer to the following question. Avoid any conversational filler.
    Strictly answer in {word_count} words.
    Question: {question}
    """
    reporter_prompt = PromptTemplate.from_template(reporter_template)
    reporter_chain = LLMChain(llm=llm, prompt=reporter_prompt)


    # Agent 2: The Enthusiastic Storyteller (simulates the "hallucinating" agent)
    storyteller_template = """
    You are an enthusiastic storyteller. Answer the question in a friendly, conversational, and engaging tone. Feel free to add interesting, but potentially unverified, anecdotes.
    Strictly answer in {word_count} words.
    Question: {question}
    """
    storyteller_prompt = PromptTemplate.from_template(storyteller_template)
    storyteller_chain = LLMChain(llm=llm, prompt=storyteller_prompt)


    # Agent 3: The Skeptical Analyst (simulates the "medium" agent)
    analyst_template = """
    You are a cautious and skeptical analyst. Answer the following question but be sure to qualify your answer with phrases like "It is believed that..." or "Sources suggest...". Do not state anything as an absolute fact.
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
    """
    Main function to run the agents, get their responses, and save them to a JSON file.
    """
    questions = quesn_prompts

    if not questions:
        print("No questions found in 'questions.txt'. Exiting.")
        return

    agents = create_agents()
    all_results = []

    for i, question in enumerate(questions):
        print(f"--- Processing Question {i+1}/{len(questions)}: '{question}' ---")
        
        question_result = {
            "question": question,
            "responses": {}
        }

        for agent_name, agent_chain in agents.items():
            # Generate a NEW random word count for each agent
            random_word_count = random.randint(50, 200)

            print(f"--> Running agent: {agent_name} with a word count of {random_word_count}...")
            
            # Get agent response by passing both the question and the word count
            agent_response = agent_chain.invoke({"question": question, "word_count": random_word_count})['text'].strip()
            print(f"Agent Response: {agent_response}")
            print("-" * 20)

            # Add the response to the current question's result dictionary
            question_result["responses"][agent_name] = agent_response
        
        # Add the complete question result to the main list
        all_results.append(question_result)

    # Save the results to a JSON file
    try:
        with open("results.json", "w", encoding='utf-8') as json_file:
            json.dump(all_results, json_file, indent=4)
        print("Successfully saved all results to 'results.json'.")
    except IOError as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    main()
