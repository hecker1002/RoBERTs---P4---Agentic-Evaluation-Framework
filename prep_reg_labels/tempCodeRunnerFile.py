  prompt_doc = nlp_key_word(prompt)
    prompt_keywords = set()
    # We use a set to store unique keywords in their base form (lemma)
    for token in prompt_doc:
        # We consider non-stop-word nouns, proper nouns, and verbs as keywords
        if token.pos_ in ["NOUN", "PROPN", "VERB"] and not token.is_stop:
            prompt_keywords.add(token.lemma_.lower())