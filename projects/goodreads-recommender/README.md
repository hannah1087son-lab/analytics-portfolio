# Personalized Goodreads Recommender

**Question:** Can book recommendations reflect both a reader's rating
history and what they feel like reading right now?

![Project summary](images/project-overview.png)

## What I built

I compared a baseline model with user- and item-based collaborative
filtering. **User-based collaborative filtering with Pearson similarity
performed best on Precision@10 and Recall@10**, so I used it to generate
the initial recommendations.

I then added Gemini as a second step to re-rank those books based on a
reader's stated preference, such as mood or genre.

![Recommendation pipeline](images/pipeline.png)

## From model to product

I built the recommendation experience in Streamlit so a user can choose
a profile, generate recommendations, add a preference and see the
updated ranking with short explanations.

**[View the Streamlit code](app.py)**

## Tools

`Python` `pandas` `Surprise` `Collaborative Filtering` `Gemini`
`Streamlit`
