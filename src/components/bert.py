# import json
# import re
# from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# def load_data(file_path):
#     """
#     Loads the labeled JSON dataset from the specified file.
#     """
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             return json.load(f)
        
#     except FileNotFoundError:
#         print(f"Error: The file '{file_path}' was not found.")
#         return None
#     except json.JSONDecodeError:
#         print(f"Error: The file '{file_path}' is not a valid JSON file.")
#         return None

# def get_inferred_features(prompt, response):
#     """
#     Simulates the feature extraction process (e.g., using regex or a simpler classifier).
#     This function must be structured to generate the same features used during fine-tuning.
    
#     Args:
#         prompt (str): The original prompt text.
#         response (str): The agent's response text.

#     Returns:
#         tuple: A tuple of feature values (is_bias, is_toxic, is_following_rule_regex).
#     """
#     is_bias = False
#     is_toxic = False

#     # Checks for specific instructions ( LIKE  "list ten words") with Regex
#     is_following_rule_regex = False

#     if "list ten words" in prompt.lower() and re.search(r'\b(?:[a-z]+\s*){10}\b', response.lower()):
#         is_following_rule_regex = True
#     elif "poem about the seasons" in prompt.lower() and response.count("Winter") >= 1 and response.count("Summer") >= 1:
#         is_following_rule_regex = True
    
#     return is_bias, is_toxic, is_following_rule_regex

# def main():
#     """
#     Main function to perform inference using the fine-tuned BERT model.
#     """
#     # Use variables for paths to make it easier to change later.
#     model_directory = "D:\\This Project\\Agentic Eval\\RoBERTs---P4---Agentic-Evaluation-Framework\\src\\BERT_config"
#     data_file = "D:\\This Project\\Agentic Eval\\RoBERTs---P4---Agentic-Evaluation-Framework\\src\\config\\human_label.json"

#     # 1. Load the fine-tuned model and tokenizer
#     try:
#         model = AutoModelForSequenceClassification.from_pretrained(model_directory)
#         tokenizer = AutoTokenizer.from_pretrained(model_directory)
#         print(" Model and tokenizer loaded successfully.")
#     except Exception as e:
#         print(f" Error loading model or tokenizer from '{model_directory}'.")
#         print(f"Please ensure the directory exists and contains all necessary files. Error: {e}")
#         return

#     # Create a pipeline for text classification
    
#     # we are doing a single label classiifcation 
#     classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

#     # 2. Load the labeled dataset
#     dataset = load_data(data_file)
#     if not dataset:
#         return

#     # 3. Perform inference for each agent's response
#     print("Running Inference on Labeled Dataset ---\n")
    
#     for item in dataset:
#         question = item["question"]
#         responses = item["responses"]

#         print(f"\nQuestion: {question}")
#         print("-" * (len(question) + 10))

#         for agent_name, agent_data in responses.items():
#             response_text = agent_data["response"]
            
#             # This is the crucial step: generate the features to match the training data.
#             is_bias, is_toxic, is_following_rule_regex = get_inferred_features(question, response_text)
            
#             # Construct the input string *exactly* as it was used for fine-tuning.
#             # This format is based on your description: {prompt}, {response}, is_bias: {is_bias_value}, ...
#             input_prompt = (
#                 f"{question}, {response_text}, "
#                 f"is_bias: {is_bias}, "
#                 f"is_toxic: {is_toxic}, "
#                 f"is_following_rule_regex: {is_following_rule_regex}"
#             )
            
#             # Predict the hallucination score
#             try:
#                 prediction = classifier(input_prompt)
#                 score = prediction[0]['score']
#                 label = prediction[0]['label']
            
#                 print(f"  Agent: {agent_name}")
#                 print(f"  Response: {response_text[:70]}...")
#                 print(f"  Inferred Features: bias={is_bias}, toxic={is_toxic}, rules_followed={is_following_rule_regex}")
#                 print(f"  Predicted Hallucination Score (BERT): {score:.4f} (Label: {label})")
#                 print("-" * 50)
#             except Exception as e:
#                 print(f"  Error predicting score for agent '{agent_name}': {e}")
                
#     print("\n--- Inference Complete ---")

# if __name__ == "__main__":
#     main()
