import re

def validate_with_regex(pattern: str, text: str, flags=0) -> bool:
    try:
        compiled_pattern = re.compile(pattern, flags)
    except re.error as e:
        print(f"Error compiling regex pattern: {e}")
        return False
    # Search for the pattern in the text
    match = compiled_pattern.search(text)
    # Return True if a match is found, otherwise False
    return match is not None
def find_len_bool(k:int,ans:str)->bool:
    leni=len(ans.split(" "))+1
    delta=0.34*k//1+1
    return leni<=k+delta and leni>=k-delta


bullet_pattern = r"^\s*[\*\-\•]\s+.+"
single_para_pattern = r"^(?!.*(\r?\n){2,}).*$"
csv_pattern = r"^[^,]+(,\s*[^,]+)*$"
bold_pattern= r"\*{2}(.*?)\*{2}"
date_pattern = r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,)?\s+\d{2,4}|\d{4}-\d{2}-\d{2})\b"
url_pattern=r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/\/=]*)"
capital_pattern=r"\b[A-Z]+\b"
question_pattern=r"\b\w+\?"





# # Example 1: Check for bullet points
# llm_response_bullets = """
# Here are the key points:
# - First item.
# - Second item.
# - Third item.
# """
# bullet_pattern = r"^\s*[\*\-\•]\s+.+"
# is_valid = validate_with_regex(bullet_pattern, llm_response_bullets, flags=re.MULTILINE)
# print(f"Response has bullet points? {is_valid}")  # Output: True

# # Example 2: Check for a single paragraph (should fail)
# llm_response_multi_para = "This is the first paragraph.\n\nThis is the second."
# single_para_pattern = r"^(?!.*(\r?\n){2,}).*$"
# is_valid = validate_with_regex(single_para_pattern, llm_response_multi_para, flags=re.DOTALL)
# print(f"Response is a single paragraph? {is_valid}") # Output: False

# # Example 3: Check for a comma-separated list
# llm_response_csv = "Apples, Oranges, Pears, Grapes"
# csv_pattern = r"^[^,]+(,\s*[^,]+)*$"
# is_valid = validate_with_regex(csv_pattern, llm_response_csv)
# print(f"Response is a comma-separated list? {is_valid}") # Output: True

# text = "This is **very important** and should be noted. This is **another one**."
# pattern = r"\*{2}(.*?)\*{2}"

# bold_words = re.findall(pattern, text)
# print(bold_words)
# # Expected Output: ['very important', 'another one']

# import re

# text = """
# Meeting on 2025-09-12 was a success. 
# Please review the notes from 09/12/2025.
# The next event is scheduled for September 15, 2025.
# The old date was 12-09-2025.
# """

# # Using the general, all-in-one pattern
# pattern = r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,)?\s+\d{2,4}|\d{4}-\d{2}-\d{2})\b"

# dates = re.findall(pattern, text, re.IGNORECASE) # Use IGNORECASE for month names




# print(dates)
# # Expected Output: ['2025-09-12', '09/12/2025', 'September 15, 2025', '12-09-2025']

