# Goodreads Personalized Book Recommender

## Business Problem

Generic popularity lists do not account for individual reading
preferences. This project explored how collaborative filtering and LLM
re-ranking could create more personalized recommendations.

## Data

-   1,192 active users
-   9,964 books
-   Approximately 164,000 ratings
-   98.5% sparse user-item matrix

## Approach

Compared three recommendation approaches: - Bias-based baseline -
User-based collaborative filtering with Pearson similarity - Item-based
collaborative filtering with cosine similarity

UBCF Pearson was selected for ranked recommendations because it produced
the strongest Precision@10 and Recall@10. Gemini was then used as a
second-stage re-ranker to incorporate a user's mood or stated
preferences.

## Product Layer

Built a Streamlit interface that allowed a user to generate
recommendations, provide a mood/preference, and compare rankings before
and after LLM re-ranking.

## Business Takeaway

A retrieval + re-ranking architecture can combine
collaborative-filtering evidence with more flexible natural-language
preferences without retraining the core recommender.

## Tools

Python \| Collaborative Filtering \| Gemini \| Streamlit \|
Recommendation Metrics
