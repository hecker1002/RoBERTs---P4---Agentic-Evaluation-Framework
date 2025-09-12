import os
import uuid
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# LangChain imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ChatInteraction(BaseModel):
    """Data model for a single chat interaction."""
    id: str = Field(description="Unique identifier for the interaction.", default_factory=lambda: str(uuid.uuid4()))
    prompt: str = Field(description="The user's input prompt.")
    response: str = Field(description="The AI's generated response.")

def get_gemini_response(user_prompt: str) -> str:
    """
    Initializes the Gemini model and creates a LangChain chain to get a response.
    
    Args:
        user_prompt: The prompt string from the user.

    Returns:
        The response string from the Gemini model.
    """
    # Initialize the Gemini chat model
    # Make sure your GOOGLE_API_KEY is set in your .env file
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20")

    # Define the prompt template
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers user questions."),
        ("human", "{user_prompt}")
    ])

    # Define the output parser to get a string response
    output_parser = StrOutputParser()

    # Create the LangChain chain by piping the components together
    chain = prompt_template | llm | output_parser

    # Invoke the chain with the user's prompt
    response = chain.invoke({"user_prompt": user_prompt})
    
    return response

def main():
    """
    Main function to run the chat interaction.
    """
    # Load environment variables from a .env file (for the API key)
    load_dotenv()

    # Check if the API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found.")
        print("Please create a .env file and add your Google API key to it.")
        return

    print("Gemini Chat with Pydantic Output")
    print("---------------------------------")

    # Get user input from the console
    user_input = input("Enter your prompt: ")
    
    if not user_input:
        print("No prompt provided. Exiting.")
        return

    print("\nProcessing your request...")

    # Get the response from the model
    try:
        ai_response = get_gemini_response(user_input)

        # Structure the data into our Pydantic model
        chat_interaction = ChatInteraction(
            prompt=user_input,
            response=ai_response
        )

        # Print the structured output as a JSON object
        print("\n--- Structured Output (JSON) ---")
        # .model_dump_json() is a Pydantic method to serialize the object to a JSON string
        print(chat_interaction.model_dump_json(indent=2))
        print("--------------------------------\n")

    except Exception as e:
        print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    main()