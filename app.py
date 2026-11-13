"""Hybrid recommendation engine prototype.

Content-based (TF-IDF on tags) + item-item collaborative filtering,
blended with a tunable alpha, with explanations and an offline evaluation.
Run:  streamlit run app.py
Self-test (no UI):  python app.py
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------------------------------------------- data
ITEMS = [
    ("Inception", "scifi thriller heist dreams"),
    ("Interstellar", "scifi space drama time"),
    ("The Matrix", "scifi action cyberpunk ai"),
    ("Blade Runner 2049", "scifi noir ai cyberpunk"),
    ("Arrival", "scifi drama aliens language"),
    ("Ex Machina", "scifi thriller ai"),
    ("Mad Max: Fury Road", "action chase postapocalyptic"),
    ("John Wick", "action thriller revenge"),
    ("Die Hard", "action thriller christmas"),
    ("Mission: Impossible Fallout", "action spy thriller"),
    ("The Dark Knight", "action crime superhero thriller"),
    ("Heat", "crime thriller heist"),
    ("The Godfather", "crime drama mafia"),
    ("Goodfellas", "crime drama mafia"),
    ("Pulp Fiction", "crime comedy nonlinear"),
    ("Knives Out", "mystery comedy crime"),
    ("Gone Girl", "mystery thriller drama"),
    ("Se7en", "crime thriller mystery noir"),
    ("The Grand Budapest Hotel", "comedy quirky adventure"),
    ("Superbad", "comedy teen"),
    ("The Hangover", "comedy friends"),
    ("Step Brothers", "comedy absurd"),
    ("Paddington 2", "comedy family heartwarming"),
    ("Toy Story", "animation family adventure"),
    ("Spirited Away", "animation fantasy family"),
    ("Coco", "animation family music"),
    ("Inside Out", "animation family emotions"),
    ("Your Name", "animation romance fantasy"),
    ("La La Land", "romance music drama"),
    ("Before Sunrise", "romance drama dialogue"),
    ("The Notebook", "romance drama"),
    ("Pride and Prejudice", "romance period drama"),
    ("The Lord of the Rings", "fantasy adventure epic"),
    ("Harry Potter", "fantasy family adventure"),
    ("Pan's Labyrinth", "fantasy war drama"),
    ("The Shining", "horror thriller"),
    ("Get Out", "horror thriller social"),
    ("Hereditary", "horror drama"),
    ("A Quiet Place", "horror thriller family"),
    ("Whiplash", "drama music intense"),
]
TAGS = ["scifi", "action", "crime", "comedy", "animation", "romance", "fantasy", "horror"]


def build_catalog() -> pd.DataFrame:
    return pd.DataFrame(ITEMS, columns=["title", "tags"])


def synth_ratings(catalog: pd.DataFrame, n_users=120, per_user=14, seed=7) -> pd.DataFrame:
    """Synthetic users with 2 favourite genres each. Replace with real data later."""
    rng = np.random.default_rng(seed)
    R = np.zeros((n_users, len(catalog)))
    item_tags = [set(t.split()) for t in catalog["tags"]]
    for u in range(n_users):
        fav = set(rng.choice(TAGS, 2, replace=False))
        for i in rng.choice(len(catalog), per_user, replace=False):
            base = 4.3 if item_tags[i] & fav else 2.2
            R[u, i] = float(np.clip(round(base + rng.normal(0, 0.8)), 1, 5))
    return pd.DataFrame(R, columns=catalog["title"])


# ----------------------------------------------------------------- model
class HybridRecommender:
    def __init__(self, catalog: pd.DataFrame, ratings: pd.DataFrame):
        self.catalog = catalog.reset_index(drop=True)
        self.titles = list(self.catalog["title"])
        tfidf = TfidfVectorizer().fit_transform(self.catalog["tags"])
        self.content_sim = cosine_similarity(tfidf)
        R = ratings[self.titles].to_numpy()
        mask = R > 0
        means = np.where(mask.sum(1) > 0, R.sum(1) / np.maximum(mask.sum(1), 1), 0)
        centered = np.where(mask, R - means[:, None], 0.0)  # remove per-user bias
        self.collab_sim = cosine_similarity(centered.T)
        np.fill_diagonal(self.collab_sim, 0)
        np.fill_diagonal(self.content_sim, 0)
        self.popularity = mask.sum(0) / mask.shape[0]

    @staticmethod
    def _norm(x):
        rng = x.max() - x.min()
        return (x - x.min()) / rng if rng > 1e-9 else np.zeros_like(x)

    def scores(self, user: np.ndarray, alpha: float):
        rated = user > 0
        liked = user >= 4
        if not rated.any():  # cold start -> popularity
            return self._norm(self.popularity), np.zeros(len(user)), self.popularity
        content = self.content_sim[:, liked].mean(1) if liked.any() else np.zeros(len(user))
        w = np.where(rated, user - 3.0, 0.0)
        denom = np.abs(self.collab_sim[:, rated]).sum(1) + 1e-9
        collab = self.collab_sim @ w / denom
        hybrid = alpha * self._norm(collab) + (1 - alpha) * self._norm(content)
        return hybrid, content, collab

    def recommend(self, user: np.ndarray, alpha=0.5, k=5) -> pd.DataFrame:
        hybrid, _, _ = self.scores(user, alpha)
        hybrid = np.where(user > 0, -np.inf, hybrid)  # hide already-rated
        top = np.argsort(-hybrid)[:k]
        liked = np.where(user >= 4)[0]
        rows = []
        for i in top:
            why = "popular with everyone (cold start)"
            if len(liked):
                blend = alpha * self.collab_sim[i, liked] + (1 - alpha) * self.content_sim[i, liked]
                j = liked[int(np.argmax(blend))]
                why = f"because you liked {self.titles[j]}"
            rows.append({"title": self.titles[i], "score": round(float(hybrid[i]), 3),
                         "tags": self.catalog.loc[i, "tags"], "why": why})
        return pd.DataFrame(rows)


# ------------------------------------------------------------ evaluation
def evaluate(ratings: pd.DataFrame, catalog: pd.DataFrame, alpha: float, k=5, seed=0) -> float:
    """Hold out one liked item per user; hit-rate@k of recovering it."""
    rng = np.random.default_rng(seed)
    titles = list(catalog["title"])
    R = ratings[titles].to_numpy().copy()
    held = {}
    for u in range(len(R)):
        liked = np.where(R[u] >= 4)[0]
        if len(liked) >= 3:
            i = int(rng.choice(liked))
            held[u] = i
    train = R.copy()
    for u, i in held.items():
        train[u, i] = 0
    model = HybridRecommender(catalog, pd.DataFrame(train, columns=titles))
    hits = 0
    for u, i in held.items():
        rec = model.recommend(train[u], alpha, k)
        hits += titles[i] in set(rec["title"])
    return hits / max(len(held), 1)


# -------------------------------------------------------------------- UI
def run_ui():
    import streamlit as st

    st.set_page_config(page_title="Recommender Prototype", layout="centered")
    st.title("Hybrid Recommendation Engine")

    @st.cache_resource
    def load():
        cat = build_catalog()
        rat = synth_ratings(cat)
        return cat, rat, HybridRecommender(cat, rat)

    cat, rat, model = load()
    alpha = st.slider("Blend: content-based (0) ↔ collaborative (1)", 0.0, 1.0, 0.5, 0.05)
    k = st.slider("How many recommendations", 3, 10, 5)

    st.subheader("Rate a few titles")
    picks = st.multiselect("Titles you've seen", model.titles)
    user = np.zeros(len(model.titles))
    for t in picks:
        user[model.titles.index(t)] = st.slider(t, 1, 5, 4, key=t)

    st.subheader("Recommendations")
    st.dataframe(model.recommend(user, alpha, k), hide_index=True, use_container_width=True)

    with st.expander("Offline evaluation (hit-rate@5, synthetic data)"):
        if st.button("Run evaluation"):
            for a in (0.0, 0.5, 1.0):
                st.write(f"alpha={a}: hit-rate@5 = {evaluate(rat, cat, a):.2f}")


if __name__ == "__main__":
    import sys
    if "streamlit" in sys.modules:
        run_ui()
    else:
        cat = build_catalog()
        rat = synth_ratings(cat)
        m = HybridRecommender(cat, rat)
        me = np.zeros(len(cat))
        for t, r in [("Inception", 5), ("The Matrix", 5), ("Superbad", 2)]:
            me[m.titles.index(t)] = r
        print(m.recommend(me, 0.5, 5).to_string(index=False))
        for a in (0.0, 0.5, 1.0):
            print(f"alpha={a} hit-rate@5={evaluate(rat, cat, a):.2f}")
