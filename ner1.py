import requests
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
import spacy

# Fetch news article using NewsAPI
def fetch_news_article(api_key):
    url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
    response = requests.get(url)
    
    if response.status_code != 200:
        print(f"Error: Unable to fetch news articles. Status code: {response.status_code}")
        return None
    
    data = response.json()
    if 'articles' not in data or not data['articles']:
        print("No news articles found in the response.")
        return None
    
    article = data['articles'][0]
    article_text = f"{article['title']}. {article['description']}"
    
    return article_text

# Extract entities using NLTK (Rule-based)
def extract_entities_nltk(text):
    nltk.download('punkt')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    tree = ne_chunk(pos_tags)
    
    entities = []
    for subtree in tree:
        if hasattr(subtree, 'label'):
            entity_name = ' '.join(c[0] for c in subtree)
            entity_type = subtree.label()
            entities.append((entity_name, entity_type))
    return entities

# Extract entities using SpaCy (ML-based)
def extract_entities_spacy(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Compare entities from NLTK and SpaCy
def compare_entities(nltk_entities, spacy_entities):
    nltk_set = set(nltk_entities)
    spacy_set = set(spacy_entities)
    
    common = nltk_set & spacy_set
    only_nltk = nltk_set - spacy_set
    only_spacy = spacy_set - nltk_set
    
    return common, only_nltk, only_spacy

# API key for NewsAPI
api_key = 'e95114202431462da66561d708f9963c'
article_text = fetch_news_article(api_key)

if article_text:
    print("Fetched Article Text:", article_text)
    
    # Extract entities using NLTK
    nltk_entities = extract_entities_nltk(article_text)
    print("NLTK Entities:", nltk_entities)
    
    # Extract entities using SpaCy
    spacy_entities = extract_entities_spacy(article_text)
    print("SpaCy Entities:", spacy_entities)
    
    # Compare entities
    common_entities, nltk_only_entities, spacy_only_entities = compare_entities(nltk_entities, spacy_entities)
    
    print("Common Entities:", common_entities)
    print("Entities only in NLTK:", nltk_only_entities)
    print("Entities only in SpaCy:", spacy_only_entities)
    
    # Discussion
    print("\nDiscussion:")
    print("1. Common Entities: These are entities that both NLTK and SpaCy agree on, indicating strong confidence in their presence and type.")
    print("2. Entities only in NLTK: These might include entities identified based on rules but missed by the ML model, indicating potential gaps in the training data of the model.")
    print("3. Entities only in SpaCy: These might include more nuanced entities detected by the ML model that the rule-based approach missed, showcasing the strength of machine learning in handling diverse contexts.")
else:
    print("No article text fetched.")
