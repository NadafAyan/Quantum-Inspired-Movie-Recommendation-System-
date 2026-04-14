"""
app.py - Movie Recommendation System (Streamlit Web App)
========================================================
This is the main entry point of the application.
Run with: streamlit run app.py

The app takes 3 movie names as input and recommends
5 similar movies using content-based filtering.
"""

import streamlit as st
from recommender import load_movies, build_similarity_matrix, get_recommendations

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 Movie Recommender",
    page_icon="🎬",
    layout="centered",
)

# ──────────────────────────────────────────────
# Custom CSS for a clean, modern UI
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Global ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    }

    /* ── Title area ── */
    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 0.2rem;
        letter-spacing: 1px;
    }
    .sub-title {
        text-align: center;
        font-size: 1.05rem;
        color: #b8b8d0;
        margin-bottom: 2rem;
    }

    /* ── Movie card ── */
    .movie-card {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(10px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(100, 80, 220, 0.25);
    }
    .movie-card h3 {
        margin: 0 0 0.4rem 0;
        color: #e0d4ff;
        font-size: 1.2rem;
    }
    .movie-card p {
        margin: 0.15rem 0;
        color: #c0bcd5;
        font-size: 0.92rem;
    }

    /* ── Badge ── */
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        color: #fff;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }

    /* ── Rank number ── */
    .rank {
        display: inline-block;
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: #1a1a2e;
        width: 28px;
        height: 28px;
        border-radius: 50%;
        text-align: center;
        line-height: 28px;
        font-weight: 800;
        font-size: 0.85rem;
        margin-right: 0.6rem;
        flex-shrink: 0;
    }

    /* ── Divider ── */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
        margin: 1.5rem 0;
    }

    /* ── Matched label ── */
    .matched-label {
        color: #82d982;
        font-size: 0.88rem;
    }
    .not-found-label {
        color: #ff6b6b;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Load data and build similarity (cached for speed)
# ──────────────────────────────────────────────


@st.cache_data
def init_recommender():
    """Load movie data and build the similarity matrix.
    This is cached so it only runs once, making the app fast."""
    df = load_movies()
    similarity_matrix = build_similarity_matrix(df)
    return df, similarity_matrix


# Load everything
df, similarity_matrix = init_recommender()

# Get list of all movie names for reference
all_movies = sorted(df["movie_name"].tolist())

# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────
st.markdown('<div class="main-title">🎬 Movie Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Enter 3 movies you love and discover 5 similar ones!</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Input Section
# ──────────────────────────────────────────────
st.markdown("### 🎥 Pick Your Favourites")

# Create 3 columns for the 3 input fields
col1, col2, col3 = st.columns(3)

with col1:
    movie1 = st.text_input("Movie 1", placeholder="e.g. Inception")
with col2:
    movie2 = st.text_input("Movie 2", placeholder="e.g. The Matrix")
with col3:
    movie3 = st.text_input("Movie 3", placeholder="e.g. Interstellar")

# Show available movies in an expander for reference
with st.expander("📋 Browse available movies"):
    # Display movies in a nice 3-column layout
    cols = st.columns(3)
    for i, movie in enumerate(all_movies):
        cols[i % 3].write(f"• {movie}")

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Recommend Button
# ──────────────────────────────────────────────
recommend_clicked = st.button("🚀 Recommend Movies", use_container_width=True, type="primary")

if recommend_clicked:
    # Collect input movies
    input_movies = [movie1.strip(), movie2.strip(), movie3.strip()]

    # ── Validation: Check all 3 fields are filled ──
    non_empty = [m for m in input_movies if m]
    if len(non_empty) < 3:
        st.error("⚠️ Please enter all 3 movie names to get recommendations!")
    else:
        # ── Get recommendations ──
        results = get_recommendations(df, similarity_matrix, input_movies, top_n=5)

        # ── Show matched movies ──
        if results["found"]:
            matched_text = ", ".join(f"**{name}**" for name in results["found"])
            st.markdown(
                f'<p class="matched-label">✅ Matched: {matched_text}</p>',
                unsafe_allow_html=True,
            )

        # ── Show not-found warnings ──
        if results["not_found"]:
            for name in results["not_found"]:
                st.warning(f'❌ Movie not found in dataset: **"{name}"**. Please check the spelling.')

        # ── Display recommendations ──
        if results["recommendations"]:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("### 🌟 Top 5 Recommendations For You")
            st.write("")

            for i, movie in enumerate(results["recommendations"], 1):
                # Build the genres as badges
                genres = movie["genre"].split()
                genre_badges = " ".join(f'<span class="badge">{g}</span>' for g in genres[:3])

                # Render movie card
                st.markdown(
                    f"""
                    <div class="movie-card">
                        <div style="display:flex; align-items:center;">
                            <span class="rank">{i}</span>
                            <h3>{movie['name']}</h3>
                        </div>
                        <p>🎬 <strong>Director:</strong> {movie['director']} &nbsp;|&nbsp;
                           📅 <strong>Year:</strong> {movie['year']} &nbsp;|&nbsp;
                           ⭐ <strong>Rating:</strong> {movie['rating']}</p>
                        <p>{genre_badges}</p>
                        <p style="color:#8a85a0; font-size:0.82rem; margin-top:0.4rem;">
                           Similarity Score: {movie['score']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        elif not results["not_found"]:
            st.info("No similar movies found. Try different movie names!")

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div style="text-align:center; color:#6c6888; font-size:0.82rem; padding:1rem 0;">
        Built with ❤️ using Python &amp; Streamlit &nbsp;|&nbsp;
        Content-Based Filtering with Cosine Similarity
    </div>
    """,
    unsafe_allow_html=True,
)
