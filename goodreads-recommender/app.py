"""
Goodreads Book Concierge — Streamlit app (OPAN6604, Project 2).

Pipeline:
  1. Collaborative filtering generates Top-N candidate books for a chosen user.
  2. (Optional) A Gemini LLM re-ranks those candidates to a stated preference
     (a mood / genre) and writes a one-line reason for each pick.

Run from the folder that contains Books.csv and Ratings.csv:
    streamlit run app.py

The Gemini API key is NEVER hard-coded. It is read from st.secrets or the
GEMINI_API_KEY environment variable, or typed into the sidebar at run time.
"""

import os
import math
import json
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from surprise import KNNWithMeans, Dataset, Reader


# ======================================================================
# CONFIG  —  THE ONE SWAPPABLE SEAM
# ----------------------------------------------------------------------
# The whole "which CF model wins" question lives in these two lines. When
# the model decision is final, change SELECTED_MODEL only — every other
# part of the app calls get_candidates() and never names a model, so
# nothing downstream has to change.
# ======================================================================
#
# These are the two collaborative-filtering variants evaluated in the final
# notebook (alongside the BaselineOnly benchmark). Both use KNNWithMeans, which
# centers each neighbor's ratings on their own mean before combining them — the
# right fit for Goodreads' high, skewed ratings. UBCF Pearson was selected as
# the best model by Precision@10 / Recall@10.
MODEL_CONFIGS = {
    "UBCF Pearson": dict(k=200, sim_options={"name": "pearson", "user_based": True}),
    "IBCF Cosine":  dict(k=200, sim_options={"name": "cosine",  "user_based": False}),
}
SELECTED_MODEL = "UBCF Pearson"      # <-- selected model from notebook evaluation
DEFAULT_MIN_RATINGS = 20
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Brand palette — identical to the notebook plots so the app and the deck
# read as one project.
NAVY = "#2F4F3E"; BLUE = "#6B4F3A"; PALE = "#EFE4D6"
RED  = "#9E4F3F"; GOLD = "#B08D57"; BG   = "#FCF8F2"; INK = "#2C2A28"


# ======================================================================
# PAGE SETUP + STYLES
# ======================================================================
st.set_page_config(page_title="Book Concierge", page_icon="📖", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');

:root {{
    --forest: #2F4F3E;
    --walnut: #6B4F3A;
    --paper: #FCF8F2;
    --paper2: #F5EFE6;
    --cream: #FFFDF8;
    --ink: #2C2A28;
    --muted: #6E6258;
    --brass: #B08D57;
    --border: #E4D7C8;
}}

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

.stApp {{
    background:
        radial-gradient(circle at top left, rgba(176,141,87,.18), transparent 34%),
        radial-gradient(circle at top right, rgba(47,79,62,.13), transparent 31%),
        linear-gradient(180deg, var(--paper) 0%, var(--paper2) 100%);
}}

.block-container {{
    padding-top: 2.0rem;
    max-width: 1180px;
}}

/* Force readable text everywhere Streamlit may override theme colors */
.stApp, .main p, .main span, .main div, .main li,
[data-testid="stMarkdownContainer"], label, .stCaptionContainer {{
    color: var(--ink) !important;
}}

h1, h2, h3, h4, h5, h6, [data-testid="stHeading"] {{
    color: var(--forest) !important;
    letter-spacing: -.01em;
}}

[data-testid="stCaptionContainer"] {{
    color: var(--muted) !important;
}}

.hero-card {{
    position: relative;
    overflow: hidden;
    background:
        linear-gradient(135deg, rgba(255,253,248,.98) 0%, rgba(248,241,231,.96) 100%);
    border: 1px solid var(--border);
    border-radius: 30px;
    padding: 2.0rem 2.2rem;
    margin-bottom: 1.3rem;
    box-shadow: 0 24px 60px rgba(72,48,30,.12), 0 1px 0 rgba(255,255,255,.9) inset;
}}

.hero-card:before {{
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    width: 8px;
    height: 100%;
    background: linear-gradient(180deg, var(--forest), var(--brass), var(--walnut));
}}

.hero-card:after {{
    content: "";
    position:absolute;
    right:-80px;
    top:-90px;
    width:260px;
    height:260px;
    border-radius:50%;
    background: radial-gradient(circle, rgba(176,141,87,.16), transparent 65%);
}}

.hero-title {{
    font-family: 'Cormorant Garamond', serif;
    font-weight: 700;
    font-size: 4.15rem;
    color: var(--forest) !important;
    line-height: .96;
    margin-bottom: .55rem;
}}

.hero-sub {{
    color: var(--muted) !important;
    font-size: 1.08rem;
    line-height: 1.6;
    margin-bottom: 1rem;
    max-width: 760px;
}}

.flow {{
    display:flex;
    flex-wrap:wrap;
    gap:.55rem;
    margin-top: 1rem;
}}

.flow-step {{
    background: #F3E8D8;
    color: var(--walnut) !important;
    border:1px solid #E4D2BC;
    padding:.48rem .78rem;
    border-radius:999px;
    font-size:.82rem;
    font-weight:700;
}}

.model-badge {{
    display:inline-block;
    background: var(--forest);
    color:white !important;
    font-size:.78rem;
    font-weight:700;
    padding:.34rem .8rem;
    border-radius:999px;
    letter-spacing:.02em;
    box-shadow: 0 8px 18px rgba(47,79,62,.18);
}}

.profile-card {{
    background: rgba(255,253,248,.96);
    border:1px solid var(--border);
    border-radius:22px;
    padding:1.1rem 1.2rem;
    box-shadow:0 14px 34px rgba(72,48,30,.08);
    height:100%;
}}

.profile-card h4 {{
    margin:0 0 .55rem 0;
    color:var(--forest) !important;
    font-family:'Cormorant Garamond', serif;
    font-size:1.55rem;
    font-weight:700;
}}

.profile-card div {{
    color: var(--ink) !important;
}}

.profile-pill {{
    display:inline-block;
    background:#F1E3D1;
    color:var(--walnut) !important;
    font-weight:700;
    padding:.27rem .6rem;
    border-radius:999px;
    margin:.15rem .2rem .15rem 0;
    font-size:.78rem;
    border:1px solid #E0CEB8;
}}

.controls-label {{
    font-family:'Cormorant Garamond', serif;
    font-weight:700;
    color:var(--forest) !important;
    font-size:1.75rem;
    margin:.2rem 0 .25rem 0;
}}

/* Inputs and select boxes */
[data-baseweb="select"] > div,
[data-testid="stTextInput"] input {{
    background-color: #FFFDF8 !important;
    border-color: var(--border) !important;
    color: var(--ink) !important;
    border-radius: 14px !important;
}}

.stSlider [data-testid="stTickBar"] {{
    color: var(--muted) !important;
}}

/* Buttons */
div.stButton > button {{
    border-radius:999px !important;
    font-weight:700 !important;
    border:1px solid var(--forest) !important;
    background: var(--forest) !important;
    color:#FFFFFF !important;
    box-shadow: 0 10px 22px rgba(47,79,62,.16);
    transition: all .18s ease;
}}

div.stButton > button:hover {{
    background: var(--walnut) !important;
    border-color: var(--walnut) !important;
    transform: translateY(-1px);
    box-shadow: 0 14px 28px rgba(107,79,58,.20);
}}

div.stButton > button * {{
    color:#FFFFFF !important;
}}

/* Expanders, alerts, dividers */
[data-testid="stExpander"] {{
    background: rgba(255,253,248,.72) !important;
    border:1px solid var(--border) !important;
    border-radius:18px !important;
}}

[data-testid="stAlert"] {{
    background: #FFFDF8 !important;
    color: var(--ink) !important;
    border: 1px solid rgba(176,141,87,.35) !important;
    border-radius: 16px !important;
}}

hr {{
    border-color: rgba(107,79,58,.16) !important;
}}

</style>
""", unsafe_allow_html=True)


# Card styles live here (plain string, hex values inline) because the shelf is
# rendered inside an isolated HTML component that can't see the page's <style>.
CARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');
* { box-sizing:border-box; }
body { margin:0; font-family:'Inter',sans-serif; background:transparent; color:#2C2A28; }
.shelf { display:grid; grid-template-columns:repeat(5,1fr); gap:18px; padding:2px; }
.card { background:#FFFDF8; border-radius:20px; overflow:hidden; border:1px solid #E4D7C8;
        box-shadow:0 16px 34px rgba(72,48,30,.12); display:flex; flex-direction:column;
        transition: transform .22s ease, box-shadow .22s ease; }
.card:hover { transform: translateY(-7px); box-shadow:0 24px 44px rgba(72,48,30,.18); }
.cover-wrap { position:relative; height:250px; background:linear-gradient(135deg,#2F4F3E,#6B4F3A); }
.cover-wrap img { width:100%; height:100%; object-fit:cover; display:block; }
.rank-chip { position:absolute; top:9px; left:9px; min-width:32px; height:32px; padding:0 8px; border-radius:50%;
             background:#2F4F3E; color:#fff; font-weight:700; display:flex; align-items:center;
             justify-content:center; font-size:.9rem; box-shadow:0 3px 8px rgba(0,0,0,.25); }
.rank-chip.top { background:#B08D57; }
.move { position:absolute; top:9px; right:9px; font-size:.72rem; font-weight:800;
        padding:.2rem .5rem; border-radius:999px; background:#FFFDF8; box-shadow:0 2px 8px rgba(0,0,0,.18); }
.move.up { color:#2F4F3E; } .move.down { color:#9E4F3F; } .move.same { color:#7A746D; }
.card-body { padding:.78rem .85rem .95rem; display:flex; flex-direction:column; gap:.38rem; flex:1; }
.b-title { font-family:'Cormorant Garamond',serif; font-weight:700; color:#2F4F3E; font-size:1.18rem;
           line-height:1.08; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
           overflow:hidden; }
.b-author { color:#6E6258; font-size:.82rem; line-height:1.25; }
.b-row { display:flex; gap:.35rem; flex-wrap:wrap; margin-top:.2rem; }
.pill { font-size:.68rem; padding:.18rem .46rem; border-radius:999px; font-weight:700; }
.pill.pred { background:#F1E3D1; color:#6B4F3A; }
.pill.avg { background:#EDF3EC; color:#2F4F3E; }
.pill.cnt { background:#EEE8DF; color:#6E6258; }
.reason { font-size:.8rem; color:#2C2A28; background:#F6EFE5; border-left:3px solid #B08D57;
          padding:.48rem .58rem; border-radius:8px; margin-top:.2rem; font-style:italic; line-height:1.35; }
</style>
"""


# Carousel styles — also isolated inside the component iframe.
CAROUSEL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');
* { box-sizing:border-box; }
body { margin:0; font-family:'Inter',sans-serif; background:transparent; color:#2C2A28; }

.wrap { max-width:940px; margin:0 auto; }
.carousel { display:flex; align-items:center; justify-content:center; gap:18px; }
.nav {
  flex:0 0 auto; width:48px; height:48px; border-radius:50%; cursor:pointer;
  border:1px solid #D9C8B6; background:#FFFDF8; color:#2F4F3E; font-size:1.7rem;
  line-height:1; font-weight:800; box-shadow:0 10px 24px rgba(72,48,30,.12);
  transition: transform .15s ease, box-shadow .15s ease, background .15s ease;
}
.nav:hover { transform:scale(1.08); background:#2F4F3E; color:#fff;
             box-shadow:0 14px 28px rgba(47,79,62,.24); }
.stage { flex:1 1 auto; min-width:0; }

.spotlight {
  display:flex; gap:28px; background:#FFFDF8; border:1px solid #E4D7C8;
  border-radius:26px; padding:24px;
  box-shadow:0 28px 70px rgba(72,48,30,.18), 0 1px 0 rgba(255,255,255,.9) inset;
}
.spot-cover {
  position:relative; flex:0 0 240px; height:360px; border-radius:16px; overflow:hidden;
  background:linear-gradient(135deg,#2F4F3E,#6B4F3A);
  box-shadow:0 24px 46px rgba(72,48,30,.30), 0 8px 16px rgba(72,48,30,.18);
}
.spot-cover img { width:100%; height:100%; object-fit:cover; display:block; }
.nocover { display:flex; align-items:center; justify-content:center; height:100%;
           color:#fff; font-family:'Cormorant Garamond',serif; font-size:1.35rem; text-align:center;
           padding:18px; }
.spot-rank {
  position:absolute; top:12px; left:12px; min-width:38px; height:38px; padding:0 9px;
  border-radius:999px; background:#2F4F3E; color:#fff; font-weight:800; font-size:.95rem;
  display:flex; align-items:center; justify-content:center; box-shadow:0 3px 9px rgba(0,0,0,.30);
}
.spot-rank.top { background:#B08D57; }
.mv { position:absolute; top:12px; right:12px; font-size:.72rem; font-weight:800;
      padding:.24rem .58rem; border-radius:999px; background:#FFFDF8; box-shadow:0 2px 8px rgba(0,0,0,.18); }
.mv.up { color:#2F4F3E; } .mv.down { color:#9E4F3F; } .mv.same { color:#7A746D; }

.spot-info { flex:1; display:flex; flex-direction:column; justify-content:center; gap:.65rem; min-width:0; }
.spot-title { font-family:'Cormorant Garamond',serif; font-weight:700; color:#2F4F3E;
              font-size:2.15rem; line-height:1.02; letter-spacing:-.01em; }
.spot-author { color:#6E6258; font-size:.98rem; line-height:1.4; }
.spot-pills { display:flex; gap:.45rem; flex-wrap:wrap; margin-top:.25rem; }
.pill { font-size:.78rem; padding:.28rem .6rem; border-radius:999px; font-weight:700; }
.pill.pred { background:#F1E3D1; color:#6B4F3A; }
.pill.avg { background:#EDF3EC; color:#2F4F3E; }
.pill.cnt { background:#EEE8DF; color:#6E6258; }
.spot-reason {
  font-size:.94rem; color:#2C2A28; background:#F6EFE5; border-left:4px solid #B08D57;
  padding:.7rem .82rem; border-radius:10px; margin-top:.35rem; line-height:1.45;
}

.dots { display:flex; gap:8px; justify-content:center; margin:16px 0 4px;
        padding:0 60px; }
.dot { width:8px; height:8px; border-radius:50%; background:#D1C2B3; cursor:pointer;
       transition: background .15s ease, transform .15s ease; }
.dot.on { background:#2F4F3E; transform:scale(1.35); }

.strip { display:flex; gap:11px; overflow-x:auto; padding:12px 60px 4px;
         scroll-behavior:smooth; }
.strip::-webkit-scrollbar { height:7px; }
.strip::-webkit-scrollbar-thumb { background:#D1C2B3; border-radius:999px; }
.thumb {
  position:relative; flex:0 0 74px; height:112px; border-radius:11px; overflow:hidden;
  cursor:pointer; border:2px solid transparent; background:linear-gradient(135deg,#2F4F3E,#6B4F3A);
  transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
}
.thumb img { width:100%; height:100%; object-fit:cover; display:block; }
.t-nocover { display:flex; align-items:center; justify-content:center; height:100%;
             color:#fff; font-weight:800; }
.thumb:hover { transform:translateY(-4px); box-shadow:0 8px 16px rgba(72,48,30,.22); }
.thumb.active { border-color:#B08D57; box-shadow:0 8px 18px rgba(176,141,87,.32); }
.t-rank { position:absolute; bottom:4px; left:4px; font-size:.62rem; font-weight:800;
          color:#fff; background:rgba(47,79,62,.88); padding:.06rem .32rem; border-radius:6px; }
</style>
"""


# Explanation-card styles for the re-ranked list (the "why the AI re-ordered"
# view). Isolated inside the component iframe like the others.
EXPLAIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500;600;700&display=swap');
* { box-sizing:border-box; }
body { margin:0; font-family:'Inter',sans-serif; background:transparent; color:#2C2A28; }
.ex-list { display:flex; flex-direction:column; gap:13px; max-width:940px; margin:0 auto; }
.ex-card {
  display:flex; align-items:stretch; gap:15px; background:#FFFDF8; border:1px solid #E4D7C8;
  border-radius:18px; padding:13px 15px; box-shadow:0 12px 26px rgba(72,48,30,.10);
  transition: transform .15s ease, box-shadow .15s ease;
}
.ex-card:hover { transform:translateY(-2px); box-shadow:0 16px 32px rgba(72,48,30,.15); }
.ex-rank {
  flex:0 0 40px; height:40px; border-radius:50%; background:#2F4F3E; color:#fff;
  font-weight:800; font-size:1.05rem; display:flex; align-items:center; justify-content:center;
  align-self:center;
}
.ex-rank.top { background:#B08D57; }
.ex-cover { flex:0 0 58px; height:88px; border-radius:10px; overflow:hidden; align-self:center;
            background:linear-gradient(135deg,#2F4F3E,#6B4F3A); }
.ex-cover img { width:100%; height:100%; object-fit:cover; display:block; }
.ex-nocover { display:flex; align-items:center; justify-content:center; height:100%; font-size:1.4rem; color:#FFF; }
.ex-body { flex:1; display:flex; flex-direction:column; gap:.28rem; justify-content:center; min-width:0; }
.ex-head { display:flex; align-items:center; gap:.6rem; flex-wrap:wrap; }
.ex-title { font-family:'Cormorant Garamond',serif; font-weight:700; color:#2F4F3E; font-size:1.35rem;
            line-height:1.08; }
.ex-move { font-size:.7rem; font-weight:800; padding:.18rem .52rem; border-radius:999px; }
.ex-move.up { background:#EDF3EC; color:#2F4F3E; }
.ex-move.down { background:#F7E7E1; color:#9E4F3F; }
.ex-move.same { background:#EEE8DF; color:#6E6258; }
.ex-author { color:#6E6258; font-size:.84rem; }
.ex-reason { font-size:.9rem; color:#2C2A28; line-height:1.45; border-left:4px solid #B08D57;
             background:#F6EFE5; padding:.48rem .62rem; border-radius:8px; margin-top:.15rem; }
</style>
"""


# ======================================================================
# DATA + MODEL  (cached so Streamlit's top-to-bottom reruns stay fast)
# ======================================================================
@st.cache_data(show_spinner=False)
def load_data():
    """Read the two CSVs and build a book_id -> title lookup."""
    books = pd.read_csv("Books.csv")
    ratings = pd.read_csv("Ratings.csv")
    title_of = dict(zip(books["book_id"], books["title"]))
    return books, ratings, title_of


@st.cache_resource(show_spinner=False)
def fit_cf_model(model_name: str):
    """Fit the chosen CF model ONCE and reuse it across all sessions.
    Keyed by model_name, so switching models refits only when needed."""
    _, ratings, _ = load_data()
    reader = Reader(rating_scale=(1, 5))                 # Goodreads is 1–5
    data = Dataset.load_from_df(ratings[["user_id", "book_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    cfg = MODEL_CONFIGS[model_name]
    model = KNNWithMeans(k=cfg["k"], sim_options=cfg["sim_options"], verbose=False)
    model.fit(trainset)
    return model


@st.cache_data(show_spinner=False)
def get_candidates(user_id: int, min_ratings: int, model_name: str, top_n: int = 10):
    """THE SEAM. Return the model-agnostic Top-N candidate list for a user.

    Excludes books the user already rated and books with fewer than
    `min_ratings` ratings (CF can score a thinly-rated book 5/5 on almost no
    evidence). Each candidate carries the metadata the cards and the LLM need.
    """
    books, ratings, title_of = load_data()
    model = fit_cf_model(model_name)

    counts = ratings["book_id"].value_counts()
    avg_rating = ratings.groupby("book_id")["rating"].mean()
    popular = set(counts[counts >= min_ratings].index)
    seen = set(ratings.loc[ratings["user_id"] == user_id, "book_id"])
    meta = books.set_index("book_id")

    scored = []
    for b in books["book_id"]:
        if b in seen or b not in popular:
            continue
        row = meta.loc[b]
        scored.append({
            "book_id": int(b),
            "title": title_of.get(b, "Unknown"),
            "author": str(row.get("authors", "Unknown")),
            "year": (int(row["original_publication_year"])
                     if pd.notna(row.get("original_publication_year")) else None),
            "avg_rating": round(float(avg_rating.get(b, 0.0)), 2),
            "n_ratings": int(counts.get(b, 0)),
            "image_url": (row.get("image_url") if pd.notna(row.get("image_url")) else None),
            "predicted": round(float(model.predict(user_id, b).est), 3),
        })
    scored.sort(key=lambda r: -r["predicted"])
    return scored[:top_n]


@st.cache_data(show_spinner=False)
def get_reader_profile(user_id: int):
    """Small, demo-friendly profile summary for the selected reader."""
    books, ratings, _ = load_data()
    hist = ratings[ratings["user_id"] == user_id].copy()
    if hist.empty:
        return {"n_rated": 0, "avg_rating": 0, "top_books": []}
    merged = hist.merge(books[["book_id", "title", "authors"]], on="book_id", how="left")
    top_books = (merged.sort_values(["rating", "title"], ascending=[False, True])
                 .head(3)["title"].tolist())
    return {
        "n_rated": int(hist.shape[0]),
        "avg_rating": round(float(hist["rating"].mean()), 2),
        "top_books": top_books,
    }


# ======================================================================
# LLM RE-RANKING LAYER  (imports kept local so the app loads even if the
# google-genai package isn't installed yet)
# ======================================================================
def get_gemini_key() -> str:
    """Look for the key in secrets, then env var, then the sidebar input.
    Never stored in the file."""
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY") or st.session_state.get("gemini_key", "")


def llm_rerank(preference: str, candidates: list, llm_return: int = 5):
    """Send the CF candidates + the user's preference to Gemini and get back a
    re-ranked subset with a one-line reason each. Returns list[dict]."""
    from google import genai
    from pydantic import BaseModel, Field

    class BookPick(BaseModel):
        title:  str = Field(description="Exact title copied from the candidate list.")
        reason: str = Field(description="One sentence on why it fits the preference.")
        rank:   int = Field(description="Final rank, 1 = best fit.")

    catalog = "\n".join(
        f"- {c['title']} by {c['author']} ({c['year']}) | "
        f"avg {c['avg_rating']} | CF score {c['predicted']}"
        for c in candidates
    )
    system = (
        "You are a personal book concierge. You receive a list of candidate "
        "books chosen for a user by a collaborative-filtering model. Re-rank "
        "them by how well they fit the user's stated preference. Use ONLY the "
        f"books provided — never invent new ones. Return exactly {llm_return} "
        "books in your re-ranked order, each with a one-sentence reason tied to the "
        "preference. If there are fewer than that many strong matches, still return "
        "the best available options from the candidate list."
    )
    client = genai.Client(api_key=get_gemini_key())
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"User's preference: {preference}\n\nCandidates:\n{catalog}",
        config={
            "system_instruction": system,
            "response_mime_type": "application/json",
            "response_schema": list[BookPick],
            "temperature": 0.7,
            "max_output_tokens": 1200,
        },
    )
    return [{"title": p.title, "reason": p.reason, "rank": p.rank} for p in resp.parsed]


# ======================================================================
# RENDERING HELPERS
# ======================================================================
def _cover_html(c, rank, move=None, reason=None):
    """Build one cover card as a single line of HTML (no stray newlines)."""
    img = f'<img src="{c["image_url"]}" alt="cover">' if c.get("image_url") else ""
    move_html = ""
    if move is not None:
        cls = "up" if move > 0 else "down" if move < 0 else "same"
        sym = f"&uarr;{move}" if move > 0 else (f"&darr;{abs(move)}" if move < 0 else "=")
        move_html = f'<span class="move {cls}">{sym}</span>'
    yr = f' &middot; {c["year"]}' if c.get("year") else ""
    reason_html = f'<div class="reason">{reason}</div>' if reason else ""
    chip = "rank-chip top" if rank == 1 else "rank-chip"
    return (
        f'<div class="card"><div class="cover-wrap">{img}'
        f'<span class="{chip}">{rank}</span>{move_html}</div>'
        f'<div class="card-body"><div class="b-title">{c["title"]}</div>'
        f'<div class="b-author">{c["author"]}{yr}</div>'
        f'<div class="b-row">'
        f'<span class="pill pred">&#9733; {c["predicted"]} predicted</span>'
        f'<span class="pill avg">{c["avg_rating"]} avg</span>'
        f'<span class="pill cnt">{c["n_ratings"]} ratings</span></div>'
        f'{reason_html}</div></div>'
    )


def render_shelf(cards, moves=None, reasons=None):
    """Render the cover grid inside an isolated HTML component.

    Using components.html (an iframe) instead of st.markdown avoids
    Streamlit's markdown processor breaking up multi-element HTML, which is
    why the grid renders as a real CSS grid here.
    """
    moves = moves or {}
    reasons = reasons or {}
    cards_html = "".join(
        _cover_html(c, i + 1, moves.get(c["title"]), reasons.get(c["title"]))
        for i, c in enumerate(cards)
    )
    doc = CARD_CSS + f'<div class="shelf">{cards_html}</div>'
    rows = math.ceil(len(cards) / 5)            # grid is fixed at 5 columns
    per_row = 455 if reasons else 370           # taller rows when reasons show
    components.html(doc, height=rows * per_row + 24, scrolling=False)


def render_carousel(cards, moves=None, reasons=None):
    """Render the candidates as an interactive spotlight carousel.

    One book is shown large in the center with full details; left/right arrows
    and a clickable thumbnail strip move the spotlight. All navigation runs in
    vanilla JS inside the iframe, so there are no Streamlit reruns — it works on
    Community (free) Streamlit and feels instant.
    """
    moves = moves or {}
    reasons = reasons or {}

    payload = []
    for i, c in enumerate(cards):
        payload.append({
            "rank": i + 1,
            "title": c.get("title", "Unknown"),
            "author": c.get("author", ""),
            "year": c.get("year"),
            "img": c.get("image_url") or "",
            "predicted": c.get("predicted"),
            "avg": c.get("avg_rating"),
            "n": c.get("n_ratings"),
            "move": moves.get(c["title"]),      # +ve up, -ve down, None if n/a
            "reason": reasons.get(c["title"], ""),
        })
    data_json = json.dumps(payload)

    doc = CAROUSEL_CSS + """
    <div class="wrap">
      <div class="carousel">
        <button class="nav prev" id="prevBtn" aria-label="Previous">&#8249;</button>
        <div class="stage" id="stage"></div>
        <button class="nav next" id="nextBtn" aria-label="Next">&#8250;</button>
      </div>
      <div class="dots" id="dots"></div>
      <div class="strip" id="strip"></div>
    </div>

    <script>
      const DATA = __DATA__;
      let idx = 0;

      function star(v) { return v == null ? "" : "&#9733; " + v; }
      function moveBadge(m) {
        if (m == null) return "";
        if (m > 0)  return '<span class="mv up">&uarr; up ' + m + '</span>';
        if (m < 0)  return '<span class="mv down">&darr; down ' + Math.abs(m) + '</span>';
        return '<span class="mv same">= held</span>';
      }

      function renderStage() {
        const c = DATA[idx];
        const cover = c.img
          ? '<img src="' + c.img + '" alt="cover">'
          : '<div class="nocover">' + (c.title || '') + '</div>';
        const yr = c.year ? ' &middot; ' + c.year : '';
        const reason = c.reason
          ? '<div class="spot-reason">' + c.reason + '</div>' : '';
        const chipCls = (c.rank === 1) ? 'spot-rank top' : 'spot-rank';
        document.getElementById('stage').innerHTML =
          '<div class="spotlight">' +
            '<div class="spot-cover">' + cover +
              '<span class="' + chipCls + '">#' + c.rank + '</span>' +
              moveBadge(c.move) +
            '</div>' +
            '<div class="spot-info">' +
              '<div class="spot-title">' + c.title + '</div>' +
              '<div class="spot-author">' + (c.author || '') + yr + '</div>' +
              '<div class="spot-pills">' +
                '<span class="pill pred">' + star(c.predicted) + ' predicted</span>' +
                '<span class="pill avg">' + (c.avg != null ? c.avg : '?') + ' avg</span>' +
                '<span class="pill cnt">' + (c.n != null ? c.n : '?') + ' ratings</span>' +
              '</div>' + reason +
            '</div>' +
          '</div>';
        document.querySelectorAll('.thumb').forEach((t, i) =>
          t.classList.toggle('active', i === idx));
        document.querySelectorAll('.dot').forEach((d, i) =>
          d.classList.toggle('on', i === idx));
      }

      function buildStrip() {
        const strip = document.getElementById('strip');
        const dots  = document.getElementById('dots');
        strip.innerHTML = DATA.map((c, i) =>
          '<div class="thumb" data-i="' + i + '">' +
            (c.img ? '<img src="' + c.img + '">'
                   : '<div class="t-nocover">' + c.rank + '</div>') +
            '<span class="t-rank">#' + c.rank + '</span>' +
          '</div>').join('');
        dots.innerHTML = DATA.map((_, i) =>
          '<span class="dot" data-i="' + i + '"></span>').join('');
        strip.querySelectorAll('.thumb').forEach(t =>
          t.addEventListener('click', () => { idx = +t.dataset.i; renderStage();
            t.scrollIntoView({behavior:'smooth', inline:'center', block:'nearest'}); }));
        dots.querySelectorAll('.dot').forEach(d =>
          d.addEventListener('click', () => { idx = +d.dataset.i; renderStage(); }));
      }

      function go(step) {
        idx = (idx + step + DATA.length) % DATA.length;
        renderStage();
        const active = document.querySelectorAll('.thumb')[idx];
        if (active) active.scrollIntoView({behavior:'smooth', inline:'center', block:'nearest'});
      }

      document.getElementById('prevBtn').addEventListener('click', () => go(-1));
      document.getElementById('nextBtn').addEventListener('click', () => go(1));
      document.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowLeft')  go(-1);
        if (e.key === 'ArrowRight') go(1);
      });

      buildStrip();
      renderStage();
    </script>
    """.replace("__DATA__", data_json)
    height = 620 if reasons else 590
    components.html(doc, height=height, scrolling=False)


def render_rerank_explanations(ranked_cards, moves, reasons):
    """Render the AI re-ranked picks as a vertical list of explanation cards
    (rank + cover + title + movement vs CF + the LLM's reason), rather than a
    flat table. This makes the 'why' the AI re-ordered things the focal point.
    """
    rows = []
    for i, c in enumerate(ranked_cards):
        mv = moves.get(c["title"])
        if mv is None:
            mv_html = ""
        elif mv > 0:
            mv_html = f'<span class="ex-move up">&uarr; up {mv} from CF</span>'
        elif mv < 0:
            mv_html = f'<span class="ex-move down">&darr; down {abs(mv)} from CF</span>'
        else:
            mv_html = '<span class="ex-move same">= held position</span>'
        cover = (f'<img src="{c["image_url"]}" alt="">'
                 if c.get("image_url") else '<div class="ex-nocover">📖</div>')
        yr = f' &middot; {c["year"]}' if c.get("year") else ""
        rank_cls = "ex-rank top" if i == 0 else "ex-rank"
        rows.append(
            f'<div class="ex-card">'
            f'<div class="{rank_cls}">{i + 1}</div>'
            f'<div class="ex-cover">{cover}</div>'
            f'<div class="ex-body">'
            f'<div class="ex-head"><span class="ex-title">{c["title"]}</span>{mv_html}</div>'
            f'<div class="ex-author">{c["author"]}{yr}</div>'
            f'<div class="ex-reason">{reasons.get(c["title"], "")}</div>'
            f'</div></div>'
        )
    doc = EXPLAIN_CSS + f'<div class="ex-list">{"".join(rows)}</div>'
    components.html(doc, height=len(rows) * 150 + 24, scrolling=False)


def _clean_title(title: str) -> str:
    """Normalize titles so the LLM can be matched back to the exact CF candidate.
    Gemini sometimes drops subtitles, punctuation, or series notes; this keeps
    the AI layer from showing only one book when several returned titles are
    slightly formatted differently from Books.csv.
    """
    import re
    t = str(title).lower()
    t = re.sub(r"\([^)]*\)", "", t)        # remove parenthetical subtitles
    t = re.sub(r"[^a-z0-9]+", " ", t)       # punctuation -> spaces
    return " ".join(t.split())


def match_candidate(llm_title, candidates):
    """Map an LLM-returned title back to a candidate using exact and fuzzy matching."""
    if not llm_title:
        return None

    # 1) Exact match
    for c in candidates:
        if c["title"] == llm_title:
            return c

    wanted = _clean_title(llm_title)
    if not wanted:
        return None

    # 2) Normalized exact match
    for c in candidates:
        if _clean_title(c["title"]) == wanted:
            return c

    # 3) Prefix / containment match for subtitles and series labels
    for c in candidates:
        candidate_key = _clean_title(c["title"])
        if candidate_key.startswith(wanted) or wanted.startswith(candidate_key):
            return c
        if wanted in candidate_key or candidate_key in wanted:
            return c

    return None


# ======================================================================
# UI
# ======================================================================
# Fail clearly if the data files aren't next to the app (the classic
# "worked in the notebook, FileNotFoundError in the app" trap).
if not (os.path.exists("Books.csv") and os.path.exists("Ratings.csv")):
    st.error("Couldn't find Books.csv / Ratings.csv. Launch the app from the "
             "folder that contains them:  `streamlit run app.py`")
    st.stop()

books, ratings, title_of = load_data()
user_ids = sorted(ratings["user_id"].unique())

# ---- Hero ----
st.markdown(
    f'''<div class="hero-card">
        <div class="hero-title">Goodreads Book Concierge</div>
        <div class="hero-sub">A polished reading room for discovering books chosen by readers with similar taste, then refined by the mood you want to read in.</div>
        <span class="model-badge">Candidate model · {SELECTED_MODEL}</span>
        <div class="flow">
            <span class="flow-step">1 · Choose reader</span>
            <span class="flow-step">2 · Generate shelf</span>
            <span class="flow-step">3 · Describe mood</span>
            <span class="flow-step">4 · AI re-ranks</span>
        </div>
    </div>''',
    unsafe_allow_html=True
)

# ---- Control bar (was the sidebar) ----
# Sits directly under the hero so the whole "build your shelf" workflow reads
# left-to-right across the top of the page instead of down a sidebar.
st.markdown('<div class="controls-label">Build your shelf</div>', unsafe_allow_html=True)

if "user_id" not in st.session_state:
    st.session_state.user_id = user_ids[0]

c_reader, c_dice, c_topn, c_minr = st.columns([2.2, 1.1, 2.0, 2.0])

with c_reader:
    st.session_state.user_id = st.selectbox(
        "Choose a reader", user_ids,
        index=user_ids.index(st.session_state.user_id),
    )
    user_id = int(st.session_state.user_id)

with c_dice:
    # Spacer nudges the button down so it lines up with the selectbox input.
    st.markdown('<div style="height:1.75rem"></div>', unsafe_allow_html=True)
    if st.button("Surprise me", use_container_width=True):
        import random
        st.session_state.user_id = random.choice(user_ids)
        st.rerun()

with c_topn:
    top_n = st.slider("Number of recommendations", 5, 20, 10)

with c_minr:
    min_ratings = st.slider(
        "Popularity filter", 1, 200, DEFAULT_MIN_RATINGS,
        help="Hide thinly-rated books — CF can score a book 5/5 on 1–2 ratings.",
    )

# Model switch + API key tucked into an expander so the bar stays uncluttered.
with st.expander("Model settings & API key"):
    mc1, mc2 = st.columns(2)
    with mc1:
        active_model = st.selectbox(
            "CF model used for candidates",
            list(MODEL_CONFIGS.keys()),
            index=list(MODEL_CONFIGS).index(SELECTED_MODEL),
            help="The recommender that produces the Top-N list the LLM re-ranks.",
        )
    with mc2:
        if not get_gemini_key():
            st.session_state.gemini_key = st.text_input(
                "Gemini API key (for AI re-rank)", type="password",
                help="Not stored or submitted. Get a free key from Google AI Studio.",
            )
        else:
            st.caption("✓ Gemini key detected")

st.divider()

# ---- Reader profile ----
profile = get_reader_profile(user_id)
top_book_html = "".join(f'<span class="profile-pill">{b[:34]}</span>' for b in profile["top_books"]) \
    or '<span style="color:#7F8C8D; font-size:.86rem;">No ratings yet</span>'
p1, p2 = st.columns(2)
with p1:
    st.markdown(
        f'''<div class="profile-card">
            <h4>Reader {user_id} profile</h4>
            <div><b>{profile["n_rated"]}</b> books rated</div>
            <div><b>{profile["avg_rating"]}</b> average rating</div>
        </div>''',
        unsafe_allow_html=True
    )
with p2:
    st.markdown(
        f'''<div class="profile-card">
            <h4>Top-rated examples</h4>
            <div style="color:#34495E; font-size:.86rem; margin-bottom:.4rem;">Books this reader rated most highly</div>
            <div>{top_book_html}</div>
        </div>''',
        unsafe_allow_html=True
    )

# ---- CF candidates ----
st.subheader("Your Recommended Shelf")
st.caption(f"Curated for Reader {user_id}. Browse with the arrows or thumbnails.")
with st.spinner("Fitting model and scoring books…"):
    candidates = get_candidates(user_id, min_ratings, active_model, top_n)

if not candidates:
    st.warning("No books pass the filter — lower the popularity filter.")
    st.stop()

render_carousel(candidates)

# ---- AI re-ranking ----
st.divider()
st.subheader("Refine the shelf by mood")
st.caption("Choose a quick mood or write your own. The AI layer re-ranks the same CF candidates and explains why.")

if "mood_pref" not in st.session_state:
    st.session_state.mood_pref = ""

mood_cols = st.columns(5)
mood_examples = ["Dark + thoughtful", "Cozy mystery", "Comfort romance", "Fast fantasy", "Science & nature"]
for col, mood in zip(mood_cols, mood_examples):
    if col.button(mood, use_container_width=True):
        st.session_state.mood_pref = mood.lower().replace("+", "and")

preference = st.text_input(
    "Reading mood",
    key="mood_pref",
    placeholder="e.g. something dark, scary, yet thought-provoking",
)
go = st.button("Re-rank this shelf", type="primary", disabled=not preference.strip())

sig = (user_id, active_model, top_n, min_ratings, preference.strip())
if go:
    if not get_gemini_key():
        st.warning("Add a Gemini API key in the sidebar to use the AI concierge.")
    else:
        try:
            with st.spinner("The concierge is reading…"):
                picks = llm_rerank(preference.strip(), candidates, llm_return=top_n)
            st.session_state.rerank = {"sig": sig, "picks": picks}
        except ModuleNotFoundError:
            st.error("The `google-genai` package isn't installed. Run: "
                     "`pip install google-genai`")
        except Exception as e:
            st.error(f"AI re-rank failed: {e}")

# Show the re-ranked shelf if it matches the current selection
rr = st.session_state.get("rerank")
if rr and rr["sig"] == sig:
    picks = sorted(rr["picks"], key=lambda p: p.get("rank", 999))
    cf_order = {c["title"]: i for i, c in enumerate(candidates)}
    ranked_cards, moves, reasons = [], {}, {}
    used_titles = set()

    # First: use the AI's order wherever we can match it back to a CF candidate.
    for p in picks:
        c = match_candidate(p.get("title"), candidates)
        if not c or c["title"] in used_titles:
            continue
        ranked_cards.append(c)
        used_titles.add(c["title"])
        reasons[c["title"]] = p.get("reason", "A strong match for the mood you chose.")

    # Second: if Gemini returned fewer books, or title matching missed any, append
    # the remaining CF candidates so the AI shelf is a true full re-ranked shelf
    # instead of a single-card result.
    for c in candidates:
        if c["title"] not in used_titles:
            ranked_cards.append(c)
            used_titles.add(c["title"])
            reasons.setdefault(c["title"], "Still recommended by the collaborative filtering shelf.")
        if len(ranked_cards) >= top_n:
            break

    # Movement is calculated after the final AI order is assembled.
    for new_idx, c in enumerate(ranked_cards):
        moves[c["title"]] = cf_order.get(c["title"], new_idx) - new_idx   # +ve = moved up

    st.caption(f'Re-ranked for: *"{preference.strip()}"*  ·  arrows show movement vs the original CF order')
    if ranked_cards:
        top = ranked_cards[0]
        st.success(f"Top AI pick: **{top['title']}** — {reasons.get(top['title'], 'best fit for your mood')}")
        # Full AI-ranked shelf, not just the top pick.
        render_carousel(ranked_cards, moves=moves, reasons=reasons)
        st.markdown("##### Why the concierge ranked them this way")
        render_rerank_explanations(ranked_cards, moves, reasons)

# ---- Reading history ----
st.divider()
with st.expander(f"What reader {user_id} has already rated"):
    hist = (ratings[ratings["user_id"] == user_id]
            .merge(books[["book_id", "title", "authors"]], on="book_id")
            [["title", "authors", "rating"]]
            .sort_values("rating", ascending=False)
            .reset_index(drop=True))
    hist.columns = ["Book", "Author", "Their rating"]
    hist.index = hist.index + 1
    st.dataframe(hist, use_container_width=True)