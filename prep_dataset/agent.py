import os
import json
from typing import List, Dict, Any
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from api_call import google_api 

# Set up environment variable for the Gemini API key.
# This assumes you have your API key stored in the environment.
# If not, you can replace os.getenv("GOOGLE_API_KEY") with your actual key.
os.environ["GOOGLE_API_KEY"] = google_api 

# Ensure the GOOGLE_API_KEY is set.
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it or replace 'os.getenv' with your key.")

def get_llm(temperature: float = 0.7, model_name: str = "gemini-2.5-flash-preview-05-20"):
    """
    Initializes and returns a ChatGoogleGenerativeAI LLM instance.
    NOTE: The model name has been updated to a currently available version.
    """
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)

def create_agents() -> Dict[str, LLMChain]:
    """
    Creates three distinct LLMChain "agents" with different personas.
    - The "Best" agent (Factual Reporter) is designed to be concise and factual.
    - The "Medium" agent (Skeptical Analyst) is cautious and qualifies answers.
    - The "Hallucinating" agent (Enthusiastic Storyteller) is conversational and might add unverified details.
    """
    llm = get_llm()

    # Agent 1: The Factual Reporter (simulates the "best" agent)
    reporter_template = """
    You are a professional news reporter. Your goal is to provide a concise, factual, and unbiased answer to the following question. Avoid any conversational filler.
    Question: {question}
    """
    reporter_prompt = PromptTemplate.from_template(reporter_template)
    reporter_chain = LLMChain(llm=llm, prompt=reporter_prompt)

    # Agent 2: The Enthusiastic Storyteller (simulates the "hallucinating" agent)
    storyteller_template = """
    You are an enthusiastic storyteller. Answer the question in a friendly, conversational, and engaging tone. Feel free to add interesting, but potentially unverified, anecdotes.
    Question: {question}
    """
    storyteller_prompt = PromptTemplate.from_template(storyteller_template)
    storyteller_chain = LLMChain(llm=llm, prompt=storyteller_prompt)

    # Agent 3: The Skeptical Analyst (simulates the "medium" agent)
    analyst_template = """
    You are a cautious and skeptical analyst. Answer the following question but be sure to qualify your answer with phrases like "It is believed that..." or "Sources suggest...". Do not state anything as an absolute fact.
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
    Main function to run the agents and get their responses.
    """
    questions = [
        "What is the capital of France?",
        "Explain the concept of quantum entanglement in simple terms.",
        "Who invented the telephone, and in what year?"
    ]

    agents = create_agents()

    for i, question in enumerate(questions):
        print(f"--- Processing Question {i+1}/{len(questions)}: '{question}' ---")

        for agent_name, agent_chain in agents.items():
            print(f"--> Running agent: {agent_name}...")
            
            # Get agent response
            agent_response = agent_chain.invoke({"question": question})['text'].strip()
            print(f"Agent Response: {agent_response}")
            print("-" * 20)

if __name__ == "__main__":
    main()
