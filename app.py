import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Product Finder",
    page_icon="🛍️",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTextInput > div > div > input {
        font-size: 18px;
        padding: 14px;
        border-radius: 12px;
    }
    .product-card {
        background: #1e2130;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
        border: 1px solid #2e3250;
    }
    .product-name {
        font-size: 16px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 6px;
    }
    .product-meta {
        font-size: 13px;
        color: #a0a8c0;
    }
    .badge {
        display: inline-block;
        background: #2e3a5e;
        color: #7eb3ff;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 12px;
        margin-right: 6px;
    }
    .score-bar {
        height: 4px;
        border-radius: 2px;
        background: linear-gradient(90deg, #4f8ef7, #a78bfa);
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model + data (cached so it only runs once) ──────────────────────────
@st.cache_resource(show_spinner="Loading recommendation engine...")
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner="Loading product index...")
def load_index():
    index = faiss.read_index('model_data/products.index')
    return index

@st.cache_data(show_spinner="Loading product data...")
def load_products():
    return pd.read_parquet('model_data/products.parquet')


# ── Search function ───────────────────────────────────────────────────────────
def search(query: str, top_k: int = 10, min_rating: float = 0.0, max_price: float = 999999):
    model = load_model()
    index = load_index()
    df = load_products()

    # Embed query
    query_vec = model.encode([query], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(query_vec)

    # Search — fetch extra results so we can filter
    scores, indices = index.search(query_vec, top_k * 5)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(df):
            continue
        row = df.iloc[idx]
        price = float(row.get('price', 0) or 0)
        rating = float(row.get('rating', 0) or 0)

        if rating < min_rating:
            continue
        if max_price < 999999 and price > max_price:
            continue

        results.append({
            'name': row.get('name', 'Unknown'),
            'category': row.get('category', 'N/A'),
            'rating': rating,
            'num_reviews': int(row.get('num_reviews', 0) or 0),
            'price': price,
            'image_url': row.get('image_url', ''),
            'product_url': row.get('product_url', ''),
            'score': float(score)
        })

        if len(results) >= top_k:
            break

    return results


# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🛍️ Amazon Product Finder")
st.markdown("Describe what you're looking for in plain English — brands, features, budget, anything.")

# Search bar
default_query = st.session_state.pop('query', '')
query = st.text_input(
    label="Search",
    placeholder="e.g. lightweight laptop for gaming under $800, or noise cancelling headphones for travel",
    label_visibility="collapsed",
    value=default_query
)

# Filters
with st.expander("⚙️ Filters (optional)"):
    col1, col2, col3 = st.columns(3)
    with col1:
        top_k = st.slider("Number of results", 5, 20, 10)
    with col2:
        min_rating = st.slider("Minimum rating ⭐", 0.0, 5.0, 0.0, step=0.5)
    with col3:
        max_price = st.number_input("Max price ($)", min_value=0, max_value=100000, value=0,
                                     help="Set to 0 for no limit")
        if max_price == 0:
            max_price = 999999

# Run search
if query.strip():
    with st.spinner("Finding best matches..."):
        results = search(query.strip(), top_k=top_k, min_rating=min_rating, max_price=max_price)

    if not results:
        st.warning("No products found matching your filters. Try relaxing the filters.")
    else:
        st.markdown(f"### Top {len(results)} results for *\"{query}\"*")
        st.markdown("---")

        for i, r in enumerate(results):
            pct = int(r['score'] * 100)
            stars = "⭐" * int(round(r['rating'])) if r['rating'] > 0 else ""
            price_str = f"${r['price']:.2f}" if r['price'] > 0 else "Price N/A"
            reviews_str = f"{r['num_reviews']:,} reviews" if r['num_reviews'] > 0 else ""

            col_img, col_info = st.columns([1, 5])

            with col_img:
                if r['image_url']:
                    try:
                        st.image(r['image_url'], width=90)
                    except:
                        st.markdown("🖼️")
                else:
                    st.markdown("🖼️")

            with col_info:
                name_display = r['name'][:120] + ("..." if len(r['name']) > 120 else "")
                url = r['product_url']
                name_html = f'<a href="{url}" target="_blank" style="text-decoration:none; color:#ffffff;">{name_display}</a>' if url else name_display

                st.markdown(f"""
                <div class="product-card">
                    <div class="product-name">{i+1}. {name_html}</div>
                    <div class="product-meta">
                        <span class="badge">{r['category']}</span>
                        <span>{stars} {r['rating']:.1f}</span>
                        {"&nbsp;·&nbsp;" + reviews_str if reviews_str else ""}
                        &nbsp;·&nbsp; <strong style="color:#7aedba">{price_str}</strong>
                    </div>
                    <div class="score-bar" style="width:{pct}%"></div>
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("---")
    st.markdown("#### 💡 Try searching for:")
    examples = [
        "bluetooth speaker waterproof for outdoor camping",
        "gaming chair with lumbar support under $300",
        "organic skincare moisturizer for sensitive skin",
        "running shoes lightweight for marathon",
        "4K monitor for graphic design under $500"
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"🔍 {ex}", key=f"ex_{i}"):
                st.session_state['query'] = ex
                st.rerun()
