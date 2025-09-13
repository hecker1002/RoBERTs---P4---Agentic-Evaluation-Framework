from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
import nltk
from sentence_transformers.cross_encoder import CrossEncoder
from itertools import combinations
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.spatial.distance import jensenshannon

model = SentenceTransformer('all-MiniLM-L6-v2')
nlp_key_word = spacy.load("en_core_web_sm")

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:                  # resource missing
        ok = nltk.download('punkt_tab', quiet=True)
        if not ok:
            raise RuntimeError(f"Failed to download NLTK resource: {pkg_id}")

# for internal coherence
nli_model = CrossEncoder('cross-encoder/nli-roberta-base')

try:
    stopwords.words('english')
except LookupError:
    print("Downloading NLTK resources...")
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')


def calculate_semantic_similarity(prompt: str, response: str) -> float:
    """
    input are prompt and response , then converted to embeddings and further checking its cosine similarity
    """
    # Encode the prompt and response into dense vector embeddings
    embeddings = model.encode([prompt, response])

    # Compute the cosine similarity between the two embeddings
    # The result is a 2x2 matrix, we need the value at [0, 1]
    similarity_score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

    return similarity_score

def calculate_keyword_recall(prompt: str, response: str) -> dict:
    """
    Returns:
        A dictionary containing the recall percentage, a list of keywords
        from the prompt, and a list of the keywords found in the response.
    """
    # Process the prompt to identify keywords
    # 
    prompt_doc = nlp_key_word(prompt)
    prompt_keywords = set()
    # We use a set to store unique keywords in their base form (lemma)
    for token in prompt_doc:
        # We consider non-stop-word nouns, proper nouns, and verbs as keywords
        if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop:
            prompt_keywords.add(token.lemma_.lower())

    # If no keywords are found in the prompt, recall is not applicable.
    if not prompt_keywords:
        return {
            "recall_percentage": 0.0,
            "prompt_keywords": [],
            "found_keywords": []
        }

    # Process the response and get a set of its lemmas for efficient lookup
    response_doc = nlp_key_word(response)
    response_lemmas = {token.lemma_.lower() for token in response_doc}

    # Find the intersection of keywords between prompt and response
    found_keywords = prompt_keywords.intersection(response_lemmas)

    # Calculate the recall percentage
    recall_percentage = (len(found_keywords) / len(prompt_keywords)) * 100

    return {
        "recall_percentage": round(recall_percentage, 2),
        "prompt_keywords": sorted(list(prompt_keywords)),
        "found_keywords": sorted(list(found_keywords))
    }

def detect_internal_contradictions_nli(response: str):
    """
    Detects internal contradictions in a block of text using an NLI model.

    Args:
        response: The text to be checked for contradictions.

    Returns:
        A dictionary with the most contradictory pair of sentences and their score.
    """
    sentences = nltk.sent_tokenize(response)

    if len(sentences) < 2:
        print("Not enough sentences to compare.")
        return None

    # Generate all unique pairs of sentences
    sentence_pairs = list(combinations(sentences, 2))

    # The NLI model expects a list of lists/tuples: [ [sent1, sent2], [sent1, sent3], ... ]
    scores = nli_model.predict(sentence_pairs)

    # The model outputs scores for [contradiction, entailment, neutral] for each pair
    # We are interested in the highest contradiction score (index 0)
    max_contradiction_score = -1
    most_contradictory_pair = None

    for i, score in enumerate(scores):
        contradiction_score = score[0]
        if contradiction_score > max_contradiction_score:
            max_contradiction_score = contradiction_score
            most_contradictory_pair = sentence_pairs[i]
    ## Output is a sentence pair
    return {
        "most_contradictory_pair": most_contradictory_pair,
        "contradiction_score": max_contradiction_score
    }

def preprocess_text_topic(text: str) -> list[str]:
    """
    Cleans and tokenizes text for LDA by lemmatizing, removing stopwords, and punctuation.
    """
    # Tokenize and convert to lower case
    tokens = word_tokenize(text.lower())
    # Remove punctuation and non-alphabetic characters
    tokens = [word for word in tokens if word.isalpha()]
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

def calculate_topic_divergence(prompt: str, response: str, num_topics: int = 5) -> dict:
    """
    Calculates topic divergence between a prompt and a response using LDA and Jensen-Shannon distance.

    Args:
        prompt: The user's input string.
        response: The AI's response string.
        num_topics: The number of latent topics to discover.

    Returns:
        A dictionary containing the JS-divergence score and the topic distributions.
    """
    # 1. Preprocess both the prompt and response
    processed_prompt = " ".join(preprocess_text_topic(prompt))
    processed_response = " ".join(preprocess_text_topic(response))

    if not processed_prompt or not processed_response:
        return {
            "js_divergence": 1.0, # Max divergence if one is empty
            "prompt_topic_dist": [],
            "response_topic_dist": [],
            "topics": {}
        }

    documents = [processed_prompt, processed_response]

    # 2. Create a document-term matrix
    vectorizer = CountVectorizer()
    doc_term_matrix = vectorizer.fit_transform(documents)

    # 3. Train the LDA model on the combined corpus
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(doc_term_matrix)

    # 4. Get the topic distributions for the prompt and response
    topic_distributions = lda.transform(doc_term_matrix)
    prompt_topic_dist = topic_distributions[0]
    response_topic_dist = topic_distributions[1]

    # 5. Calculate the Jensen-Shannon divergence
    # scipy's jensenshannon calculates the square root of the JS-divergence, which is a true metric.
    js_divergence = jensenshannon(prompt_topic_dist, response_topic_dist)

    # (Optional) Extract the top words for each topic for interpretability
    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for topic_idx, topic in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-5 - 1:-1]]
        topics[f"Topic {topic_idx}"] = ", ".join(top_words)

    return {
        "js_divergence": js_divergence,
        "prompt_topic_dist": prompt_topic_dist.tolist(),
        "response_topic_dist": response_topic_dist.tolist(),
        "topics": topics
    }

def main_internal_coherence():
    response_with_contradiction = (
    "Our company's headquarters are located in Paris, the capital of France. We pride ourselves on our rich European heritage. The main office is situated in Rome, as we have no presence in France."
    )
    contradiction_results = detect_internal_contradictions_nli(response_with_contradiction)

    if contradiction_results:
        pair = contradiction_results['most_contradictory_pair']
        score = contradiction_results['contradiction_score']
        print("Detected a potential contradiction!")
        print(f"Sentence 1: '{pair[0]}'")
        print(f"Sentence 2: '{pair[1]}'")
        # A score > 0.5 is a good indicator of a potential contradiction
        print(f"Contradiction Score: {score:.4f}")


def main_keyword_recall():
    prompt="what does Python script uses to parse files"
    response_good = "Certainly! This Python script uses the csv module to parse your file and print the header."
    result_good = calculate_keyword_recall(prompt, response_good)

    print("--- Good Response Analysis ---")
    print(f"Prompt Keywords: {result_good['prompt_keywords']}")
    print(f"Keywords Found: {result_good['found_keywords']}")
    print(f"Keyword Recall: {result_good['recall_percentage']}%")

def main_cos_similarity ():
# Example 1: High Similarity
    prompt_1 = "What is the capital of France?"
    response_1 = "The capital of France is Paris, a major European city and a global center for art."

    similarity_1 = calculate_semantic_similarity(prompt_1, response_1)
    print(f"Prompt: \"{prompt_1}\"")
    print(f"Response: \"{response_1}\"")
    print(f"Cosine Similarity Score: {similarity_1:.4f}\n")

def main_topic():
    prompt = "Can you explain the best Python libraries for data visualization? I need to create bar charts and scatter plots."

# --- Test Case 1: On-Topic Response (Low Divergence Expected) ---
    response_on_topic = ("For data visualization in Python, Matplotlib is the foundational library. "
                     "You can easily create bar charts and scatter plots with it. "
                     "Seaborn is built on top of Matplotlib and offers more aesthetic plots.")

    divergence_on_topic = calculate_topic_divergence(prompt, response_on_topic)
    print("--- On-Topic Analysis ---")
    print(f"JS Divergence Score: {divergence_on_topic['js_divergence']:.4f}")
    print("Identified Topics:", divergence_on_topic['topics'])


if __name__ == "__main__":
    # main_cos_similarity()
    # main_keyword_recall()
    # main_internal_coherence()
    main_topic()