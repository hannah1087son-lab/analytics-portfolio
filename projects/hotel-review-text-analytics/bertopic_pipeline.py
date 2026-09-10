"""Representative BERTopic pipeline from the hotel-review text analytics project.

The complete executed notebook is included as topic_modeling_notebook.html.
"""
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic

# Domain-specific stop words were added to standard English stop words.
custom_stops = list(ENGLISH_STOP_WORDS) + domain_stops

docs = df["reviewContent"].dropna().astype(str).tolist()

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

vectorizer_model = CountVectorizer(
    stop_words=custom_stops,
    min_df=1,
    ngram_range=(1, 2),
)

topic_model = BERTopic(
    embedding_model=embedding_model,
    vectorizer_model=vectorizer_model,
    min_topic_size=15,
    nr_topics="auto",
    calculate_probabilities=False,
    verbose=True,
)

topics, probs = topic_model.fit_transform(docs)
df = df.dropna(subset=["reviewContent"]).copy()
df["topic"] = topics

topic_info = topic_model.get_topic_info()
