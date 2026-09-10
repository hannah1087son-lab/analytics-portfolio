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
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from surprise import KNNBasic, Dataset, Reader


# ======================================================================
# CONFIG  —  THE ONE SWAPPABLE SEAM
# ----------------------------------------------------------------------
# The whole "which CF model wins" question lives in these two lines. When
# the model decision is final, change SELECTED_MODEL only — every other
# part of the app calls get_candidates() and never names a model, so
# nothing downstream has to change.
# ======================================================================
MODEL_CONFIGS = {
    "UBCF Pearson": dict(k=200, sim_options={"name": "pearson", "user_based": True}),
    "UBCF Cosine":  dict(k=200, sim_options={"name": "cosine",  "user_based": True}),
    "IBCF Pearson": dict(k=200, sim_options={"name": "pearson", "user_based": False}),
    "IBCF Cosine":  dict(k=200, sim_options={"name": "cosine",  "user_based": False}),
}
SELECTED_MODEL = "UBCF Pearson"      # <-- change this single line later
DEFAULT_MIN_RATINGS = 20
GEMINI_MODEL = "gemini-2.5-flash-lite"

# Brand palette — identical to the notebook plots so the app and the deck
# read as one project.
NAVY = "#1B4F72"; BLUE = "#2E86C1"; PALE = "#AED6F1"
RED  = "#E74C3C"; GOLD = "#F39C12"; BG   = "#F8FBFF"; INK = "#34495E"


# ======================================================================
# PAGE SETUP + STYLES
# ======================================================================
st.set_page_config(page_title="Book Concierge", page_icon="📚", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.block-container {{ padding-top: 2.2rem; }}

.hero-title {{
    font-family: 'Fraunces', serif; font-weight: 600;
    font-size: 2.6rem; color: {NAVY}; line-height: 1.1; margin-bottom: .2rem;
}}
.hero-sub {{ color: {INK}; font-size: 1.02rem; margin-bottom: 1rem; }}
.model-badge {{
    display:inline-block; background:{NAVY}; color:white; font-size:.78rem;
    font-weight:600; padding:.25rem .7rem; border-radius:999px; letter-spacing:.02em;
}}
</style>
""", unsafe_allow_html=True)


# Card styles live here (plain string, hex values inline) because the shelf is
# rendered inside an isolated HTML component that can't see the page's <style>.
CARD_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');
* { box-sizing:border-box; }
body { margin:0; font-family:'Inter',sans-serif; background:transparent; }
.shelf { display:grid; grid-template-columns:repeat(5,1fr); gap:16px; }
.card { background:#fff; border-radius:14px; overflow:hidden; border:1px solid #E8F0F8;
        box-shadow:0 2px 10px rgba(27,79,114,.10); display:flex; flex-direction:column; }
.cover-wrap { position:relative; height:230px; background:linear-gradient(135deg,#AED6F1,#2E86C1); }
.cover-wrap img { width:100%; height:100%; object-fit:cover; display:block; }
.rank-chip { position:absolute; top:8px; left:8px; width:30px; height:30px; border-radius:50%;
             background:#1B4F72; color:#fff; font-weight:600; display:flex; align-items:center;
             justify-content:center; font-size:.9rem; box-shadow:0 1px 4px rgba(0,0,0,.3); }
.rank-chip.top { background:#F39C12; }
.move { position:absolute; top:8px; right:8px; font-size:.72rem; font-weight:700;
        padding:.15rem .45rem; border-radius:8px; background:#fff; }
.move.up { color:#1E8449; } .move.down { color:#E74C3C; } .move.same { color:#95A5A6; }
.card-body { padding:.7rem .8rem .85rem; display:flex; flex-direction:column; gap:.35rem; flex:1; }
.b-title { font-family:'Fraunces',serif; font-weight:600; color:#1B4F72; font-size:1.0rem;
           line-height:1.2; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
           overflow:hidden; }
.b-author { color:#34495E; font-size:.8rem; }
.b-row { display:flex; gap:.35rem; flex-wrap:wrap; margin-top:.2rem; }
.pill { font-size:.7rem; padding:.16rem .45rem; border-radius:7px; font-weight:600; }
.pill.pred { background:#FCE7CC; color:#9A6400; }
.pill.avg { background:#E4F0FA; color:#2E86C1; }
.pill.cnt { background:#ECF0F1; color:#34495E; }
.reason { font-size:.8rem; color:#34495E; background:#EAF3FB; border-left:3px solid #2E86C1;
          padding:.45rem .55rem; border-radius:6px; margin-top:.2rem; font-style:italic; }
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
    model = KNNBasic(k=cfg["k"], sim_options=cfg["sim_options"], verbose=False)
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
        f"books provided — never invent new ones. Return the top {llm_return} "
        "in your re-ranked order, each with a one-sentence reason tied to the "
        "preference."
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


def match_candidate(llm_title, candidates):
    """Map an LLM-returned title back to a candidate (exact, then prefix)."""
    for c in candidates:
        if c["title"] == llm_title:
            return c
    key = llm_title[:30].strip().lower()
    for c in candidates:
        if c["title"][:30].strip().lower() == key:
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

# ---- Sidebar controls ----
with st.sidebar:
    st.header("Settings")

    # Optional live model switch — handy for comparing models while your
    # partner decides. The SUBMITTED default stays SELECTED_MODEL.
    with st.expander("Candidate model", expanded=False):
        active_model = st.selectbox(
            "CF model used for candidates",
            list(MODEL_CONFIGS.keys()),
            index=list(MODEL_CONFIGS).index(SELECTED_MODEL),
            help="The recommender that produces the Top-N list the LLM re-ranks.",
        )

    if "user_id" not in st.session_state:
        st.session_state.user_id = user_ids[0]
    if st.button("🎲 Surprise me", use_container_width=True):
        import random
        st.session_state.user_id = random.choice(user_ids)

    st.session_state.user_id = st.selectbox(
        "Reader (user ID)", user_ids,
        index=user_ids.index(st.session_state.user_id),
    )
    user_id = int(st.session_state.user_id)

    top_n = st.slider("How many candidates", 5, 20, 10)
    min_ratings = st.slider(
        "Minimum ratings per book", 1, 200, DEFAULT_MIN_RATINGS,
        help="Hide thinly-rated books — CF can score a book 5/5 on 1–2 ratings.",
    )

    if not get_gemini_key():
        st.session_state.gemini_key = st.text_input(
            "Gemini API key (for AI re-rank)", type="password",
            help="Not stored or submitted. Get a free key from Google AI Studio.",
        )

# ---- Hero ----
st.markdown('<div class="hero-title">The Book Concierge</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Collaborative filtering finds books for a reader; '
    'then an AI concierge re-ranks them to your mood.</div>', unsafe_allow_html=True)
st.markdown(f'<span class="model-badge">Candidate model · {active_model}</span>',
            unsafe_allow_html=True)

# ---- Dataset stats ----
n_users, n_books = ratings["user_id"].nunique(), ratings["book_id"].nunique()
density = 100 * len(ratings) / (n_users * n_books)
m1, m2, m3, m4 = st.columns(4)
m1.metric("Readers", f"{n_users:,}")
m2.metric("Books rated", f"{n_books:,}")
m3.metric("Ratings", f"{len(ratings):,}")
m4.metric("Matrix filled", f"{density:.1f}%")

st.divider()

# ---- CF candidates ----
st.subheader(f"Picks for reader {user_id}")
with st.spinner("Fitting model and scoring books…"):
    candidates = get_candidates(user_id, min_ratings, active_model, top_n)

if not candidates:
    st.warning("No books pass the filter — lower the minimum ratings.")
    st.stop()

render_shelf(candidates)

# ---- AI re-ranking ----
st.divider()
st.subheader("Personalize with the AI concierge")
preference = st.text_input(
    "Tell the concierge what you're in the mood for",
    placeholder="e.g. something dark, scary, yet thought-provoking",
)
go = st.button("✨ Re-rank for me", type="primary", disabled=not preference.strip())

sig = (user_id, active_model, top_n, min_ratings, preference.strip())
if go:
    if not get_gemini_key():
        st.warning("Add a Gemini API key in the sidebar to use the AI concierge.")
    else:
        try:
            with st.spinner("The concierge is reading…"):
                picks = llm_rerank(preference.strip(), candidates, llm_return=min(5, top_n))
            st.session_state.rerank = {"sig": sig, "picks": picks}
        except ModuleNotFoundError:
            st.error("The `google-genai` package isn't installed. Run: "
                     "`pip install google-genai`")
        except Exception as e:
            st.error(f"AI re-rank failed: {e}")

# Show the re-ranked shelf if it matches the current selection
rr = st.session_state.get("rerank")
if rr and rr["sig"] == sig:
    picks = sorted(rr["picks"], key=lambda p: p["rank"])
    cf_order = {c["title"]: i for i, c in enumerate(candidates)}
    ranked_cards, moves, reasons = [], {}, {}
    for new_idx, p in enumerate(picks):
        c = match_candidate(p["title"], candidates)
        if not c:
            continue
        ranked_cards.append(c)
        if c["title"] in cf_order:
            moves[c["title"]] = cf_order[c["title"]] - new_idx   # +ve = moved up
        reasons[c["title"]] = p["reason"]
    st.caption(f'Re-ranked for: *"{preference.strip()}"*  ·  arrows show movement vs the CF order')
    render_shelf(ranked_cards, moves=moves, reasons=reasons)

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