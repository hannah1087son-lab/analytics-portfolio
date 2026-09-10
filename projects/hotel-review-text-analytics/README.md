# Hotel Review Text Analytics

**Question:** What are guests consistently talking about, and where are
the clearest opportunities to improve the experience?

![Project summary](images/project-overview.png)

## What I found

The reviews consistently centered on five areas:

![Review themes](images/topic-themes.png)

The analysis helped separate strengths such as location and amenities
from operational issues around check-in, front-desk interactions and
service recovery.

## What I did

I cleaned the review text and used both LDA and BERTopic to find
recurring themes. For BERTopic, I used sentence-transformer embeddings
and included bigrams so phrases such as "front desk" stayed meaningful.
I then translated the model output into customer-experience themes that
could be used by a business team.

## Explore the analysis

**[View the Python pipeline](bertopic_pipeline.py)** · **[View the
executed notebook](topic_modeling_notebook.html)**

## Tools

`Python` `BERTopic` `LDA` `SentenceTransformers` `scikit-learn` `NLP`
