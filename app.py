import os
import streamlit as st
import torch
import numpy as np
from typing import Dict, Set, List, Tuple

from main import (
    set_seed,
    maybe_extract_zip,
    read_movielens_split,
    read_u_item,
    build_id_maps,
    invert_map,
    to_implicit_sets,
    filter_users_with_pos,
    train_cgan,
    TrainConfig,
    Generator,
    Discriminator,
    recommend_for_user,
    format_recs,
    recall_at_k,
    ndcg_at_k,
    hitrate_at_k,
)

# Page config
st.set_page_config(
    page_title="Movie-Recommender",
    page_icon="🎬",
    layout="wide"
)

# Hide Streamlit default elements (GitHub, Share, Star) but KEEP the hamburger menu
hide_streamlit_style = """
    <style>
    /* Hide footer */
    footer {visibility: hidden;}
    
    /* Hide deploy button only */
    .stDeployButton {display: none;}
    
    /* Hide the decoration (colored line at top) - optional */
    [data-testid="stDecoration"] {display: none !important;}
    
    /* Adjust top padding */
    .block-container {padding-top: 2rem;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ----------------------------
# Session State Initialization
# ----------------------------
def init_session_state():
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    if "G" not in st.session_state:
        st.session_state.G = None
    if "D" not in st.session_state:
        st.session_state.D = None
    if "train_pos" not in st.session_state:
        st.session_state.train_pos = None
    if "test_pos" not in st.session_state:
        st.session_state.test_pos = None
    if "user2idx" not in st.session_state:
        st.session_state.user2idx = None
    if "item2idx" not in st.session_state:
        st.session_state.item2idx = None
    if "idx2user" not in st.session_state:
        st.session_state.idx2user = None
    if "idx2movieid" not in st.session_state:
        st.session_state.idx2movieid = None
    if "movieid_to_title" not in st.session_state:
        st.session_state.movieid_to_title = None
    if "num_users" not in st.session_state:
        st.session_state.num_users = 0
    if "num_items" not in st.session_state:
        st.session_state.num_items = 0
    if "users_eval" not in st.session_state:
        st.session_state.users_eval = []
    if "device" not in st.session_state:
        st.session_state.device = torch.device("cpu")
    if "new_user_ratings" not in st.session_state:
        st.session_state.new_user_ratings = {}
    if "data_loaded" not in st.session_state:
        st.session_state.data_loaded = False


@st.cache_data
def load_data(data_dir: str, split: str, pos_threshold: int):
    """Load and preprocess MovieLens data."""
    zip_path = os.path.join(data_dir, "a69190f7-2dfc-4313-a968-1222b827bbb9.zip")
    if not os.path.isfile(zip_path):
        zip_path = ""
    
    ml_dir = maybe_extract_zip(zip_path, data_dir)
    movieid_to_title = read_u_item(ml_dir)
    train_raw, test_raw = read_movielens_split(ml_dir, split=split)
    
    user2idx, item2idx = build_id_maps(train_raw, test_raw)
    idx2user = invert_map(user2idx)
    idx2movieid = invert_map(item2idx)
    
    num_users = len(user2idx)
    num_items = len(item2idx)
    
    train_pos, test_pos = to_implicit_sets(
        train_raw, test_raw, user2idx, item2idx, pos_threshold=pos_threshold
    )
    users_eval = filter_users_with_pos(train_pos, test_pos)
    
    return {
        "movieid_to_title": movieid_to_title,
        "user2idx": user2idx,
        "item2idx": item2idx,
        "idx2user": idx2user,
        "idx2movieid": idx2movieid,
        "num_users": num_users,
        "num_items": num_items,
        "train_pos": train_pos,
        "test_pos": test_pos,
        "users_eval": users_eval,
        "train_raw_len": len(train_raw),
        "test_raw_len": len(test_raw),
    }


def search_movies(movieid_to_title: Dict[int, str], query: str, limit: int = 20) -> List[Tuple[int, str]]:
    """Search movies by title."""
    q = query.lower().strip()
    if not q:
        return []
    matches = []
    for mid, title in movieid_to_title.items():
        if q in title.lower():
            matches.append((mid, title))
    matches.sort(key=lambda x: x[1])
    return matches[:limit]


def get_all_movies_sorted(movieid_to_title: Dict[int, str]) -> List[Tuple[int, str]]:
    """Get all movies sorted by title."""
    return sorted(movieid_to_title.items(), key=lambda x: x[1])


# ----------------------------
# Main App
# ----------------------------
def main():
    init_session_state()
    
    st.title("🎬 CGAN-Based Movie Recommender System")
    st.markdown("---")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        data_dir = st.text_input("Data Directory", value=".", help="Directory containing the dataset")
        split = st.selectbox("Dataset Split", ["u1", "u2", "u3", "u4", "u5"], index=0)
        pos_threshold = st.slider("Positive Rating Threshold", 1, 5, 4, 
                                   help="Ratings >= this value are considered positive")
        
        st.markdown("---")
        st.subheader("Training Parameters")
        epochs = st.slider("Epochs", 1, 50, 10)
        batch_size = st.selectbox("Batch Size", [256, 512, 1024, 2048], index=2)
        lr = st.select_slider("Learning Rate", options=[1e-4, 2e-4, 5e-4, 1e-3], value=2e-4)
        embed_dim = st.selectbox("Embedding Dimension", [32, 64, 128], index=1)
        noise_dim = st.selectbox("Noise Dimension", [16, 32, 64], index=1)
        hidden_dim = st.selectbox("Hidden Dimension", [128, 256, 512], index=1)
        seed = st.number_input("Random Seed", value=42, min_value=0)
        
        st.markdown("---")
        k = st.slider("Top-K Recommendations", 5, 50, 10)
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data & Training", "🎯 Get Recommendations", "👤 New User", "📈 Evaluation"])
    
    # Tab 1: Data Loading and Training
    with tab1:
        st.header("Load Data & Train Model")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Load Dataset", type="primary"):
                with st.spinner("Loading MovieLens data..."):
                    try:
                        set_seed(seed)
                        data = load_data(data_dir, split, pos_threshold)
                        
                        st.session_state.movieid_to_title = data["movieid_to_title"]
                        st.session_state.user2idx = data["user2idx"]
                        st.session_state.item2idx = data["item2idx"]
                        st.session_state.idx2user = data["idx2user"]
                        st.session_state.idx2movieid = data["idx2movieid"]
                        st.session_state.num_users = data["num_users"]
                        st.session_state.num_items = data["num_items"]
                        st.session_state.train_pos = dict(data["train_pos"])
                        st.session_state.test_pos = dict(data["test_pos"])
                        st.session_state.users_eval = data["users_eval"]
                        st.session_state.device = torch.device("cpu")
                        st.session_state.data_loaded = True
                        
                        st.success("✅ Data loaded successfully!")
                        st.info(f"📊 Train: {data['train_raw_len']} | Test: {data['test_raw_len']} | Users: {data['num_users']} | Items: {data['num_items']}")
                    except Exception as e:
                        st.error(f"❌ Error loading data: {str(e)}")
        
        with col2:
            if st.session_state.data_loaded:
                if st.button("🚀 Train Model", type="primary"):
                    with st.spinner("Training CGAN model... This may take a few minutes."):
                        try:
                            set_seed(seed)
                            cfg = TrainConfig(
                                epochs=epochs,
                                batch_size=batch_size,
                                lr=lr,
                                embed_dim=embed_dim,
                                noise_dim=noise_dim,
                                hidden=hidden_dim,
                            )
                            
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            G, D = train_cgan(
                                st.session_state.train_pos,
                                st.session_state.users_eval,
                                st.session_state.num_users,
                                st.session_state.num_items,
                                st.session_state.device,
                                cfg
                            )
                            
                            st.session_state.G = G
                            st.session_state.D = D
                            st.session_state.model_trained = True
                            
                            progress_bar.progress(100)
                            st.success("✅ Model trained successfully!")
                        except Exception as e:
                            st.error(f"❌ Error training model: {str(e)}")
            else:
                st.warning("⚠️ Please load the dataset first.")
        
        # Display current status
        st.markdown("---")
        st.subheader("Current Status")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Data Loaded", "✅ Yes" if st.session_state.data_loaded else "❌ No")
        with col_b:
            st.metric("Model Trained", "✅ Yes" if st.session_state.model_trained else "❌ No")
        with col_c:
            st.metric("Users Available", st.session_state.num_users if st.session_state.data_loaded else 0)
    
    # Tab 2: Get Recommendations for Existing Users
    with tab2:
        st.header("Get Recommendations for Existing Users")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first in the 'Data & Training' tab.")
        else:
            # User selection
            user_options = [f"User {st.session_state.idx2user[u]} (idx: {u})" 
                          for u in st.session_state.users_eval[:100]]
            
            selected_user = st.selectbox("Select a User", user_options)
            
            if selected_user and st.button("🎯 Get Recommendations"):
                user_idx = int(selected_user.split("idx: ")[1].rstrip(")"))
                
                with st.spinner("Generating recommendations..."):
                    recs = recommend_for_user(
                        st.session_state.G,
                        user_idx,
                        st.session_state.num_items,
                        seen=st.session_state.train_pos.get(user_idx, set()),
                        k=k,
                        device=st.session_state.device,
                        noise_samples=10
                    )
                    
                    rec_lines = format_recs(recs, st.session_state.idx2movieid, 
                                           st.session_state.movieid_to_title)
                
                st.subheader(f"🎬 Top-{k} Recommendations")
                for i, line in enumerate(rec_lines, 1):
                    st.write(f"{i}. {line}")
                
                # Show user's training history
                with st.expander("📚 User's Training History (Liked Movies)"):
                    user_history = st.session_state.train_pos.get(user_idx, set())
                    if user_history:
                        for item_idx in list(user_history)[:20]:
                            mid = st.session_state.idx2movieid.get(item_idx, None)
                            title = st.session_state.movieid_to_title.get(mid, "Unknown")
                            st.write(f"• {mid} | {title}")
                    else:
                        st.write("No training history available.")
    
    # Tab 3: New User Recommendations
    with tab3:
        st.header("👤 New User Recommendations")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first in the 'Data & Training' tab.")
        else:
            st.markdown("""
            Rate some movies to get personalized recommendations!
            Movies with rating ≥ threshold will be considered as "liked".
            """)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Movie search
                search_query = st.text_input("🔍 Search Movies", placeholder="Type movie title...")
                
                if search_query:
                    results = search_movies(st.session_state.movieid_to_title, search_query)
                    if results:
                        for mid, title in results:
                            if mid in st.session_state.item2idx:
                                col_title, col_rating = st.columns([3, 1])
                                with col_title:
                                    st.write(f"**{mid}** | {title}")
                                with col_rating:
                                    rating = st.selectbox(
                                        "Rate",
                                        options=[0, 1, 2, 3, 4, 5],
                                        key=f"rating_{mid}",
                                        label_visibility="collapsed"
                                    )
                                    if rating > 0:
                                        st.session_state.new_user_ratings[mid] = rating
                    else:
                        st.info("No movies found. Try a different search term.")
            
            with col2:
                st.subheader("Your Ratings")
                if st.session_state.new_user_ratings:
                    for mid, rating in st.session_state.new_user_ratings.items():
                        title = st.session_state.movieid_to_title.get(mid, "Unknown")
                        emoji = "👍" if rating >= pos_threshold else "👎"
                        st.write(f"{emoji} **{rating}/5** - {title[:30]}...")
                    
                    if st.button("🗑️ Clear All Ratings"):
                        st.session_state.new_user_ratings = {}
                        st.rerun()
                else:
                    st.info("No ratings yet. Search and rate movies!")
            
            st.markdown("---")
            
            # Get recommendations for new user
            if st.button("🎯 Get My Recommendations", type="primary"):
                if not st.session_state.new_user_ratings:
                    st.warning("Please rate at least one movie first!")
                else:
                    # Convert ratings to positive set
                    new_user_pos = set()
                    for mid, rating in st.session_state.new_user_ratings.items():
                        if rating >= pos_threshold and mid in st.session_state.item2idx:
                            new_user_pos.add(st.session_state.item2idx[mid])
                    
                    if not new_user_pos:
                        st.warning(f"No movies rated ≥ {pos_threshold}. Rate some movies higher!")
                    else:
                        with st.spinner("Generating personalized recommendations..."):
                            # Create temporary new user embedding
                            new_user_idx = st.session_state.num_users
                            
                            # For simplicity, we'll use a weighted average approach
                            # based on the Generator's item preferences
                            G = st.session_state.G
                            G.eval()
                            
                            # Use existing user embeddings that liked similar items
                            similar_users = []
                            for u in st.session_state.users_eval[:500]:
                                user_pos = st.session_state.train_pos.get(u, set())
                                overlap = len(user_pos & new_user_pos)
                                if overlap > 0:
                                    similar_users.append((u, overlap))
                            
                            similar_users.sort(key=lambda x: -x[1])
                            top_similar = [u for u, _ in similar_users[:10]]
                            
                            if top_similar:
                                # Aggregate recommendations from similar users
                                all_scores = np.zeros(st.session_state.num_items)
                                
                                for u in top_similar:
                                    u_tensor = torch.tensor([u], dtype=torch.long, 
                                                          device=st.session_state.device)
                                    with torch.no_grad():
                                        logits = G(u_tensor).squeeze(0).cpu().numpy()
                                    all_scores += logits
                                
                                # Exclude already rated movies
                                all_rated = set()
                                for mid in st.session_state.new_user_ratings.keys():
                                    if mid in st.session_state.item2idx:
                                        all_rated.add(st.session_state.item2idx[mid])
                                
                                for idx in all_rated:
                                    all_scores[idx] = -1e9
                                
                                top_k_indices = np.argsort(-all_scores)[:k].tolist()
                                rec_lines = format_recs(top_k_indices, 
                                                       st.session_state.idx2movieid,
                                                       st.session_state.movieid_to_title)
                                
                                st.subheader(f"🎬 Your Top-{k} Recommendations")
                                for i, line in enumerate(rec_lines, 1):
                                    st.write(f"{i}. {line}")
                            else:
                                st.warning("Not enough data to generate recommendations. Try rating more movies!")
    
    # Tab 4: Evaluation Metrics
    with tab4:
        st.header("📈 Model Evaluation")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first in the 'Data & Training' tab.")
        else:
            if st.button("🔄 Run Evaluation"):
                with st.spinner("Evaluating model on test set..."):
                    recalls, ndcgs, hits = [], [], []
                    
                    progress_bar = st.progress(0)
                    users_to_eval = st.session_state.users_eval[:200]  # Limit for speed
                    
                    for i, u in enumerate(users_to_eval):
                        seen = st.session_state.train_pos.get(u, set())
                        gt = st.session_state.test_pos.get(u, set())
                        
                        recs = recommend_for_user(
                            st.session_state.G, u, st.session_state.num_items,
                            seen=seen, k=k, device=st.session_state.device,
                            noise_samples=5
                        )
                        
                        recalls.append(recall_at_k(recs, gt, k))
                        ndcgs.append(ndcg_at_k(recs, gt, k))
                        hits.append(hitrate_at_k(recs, gt, k))
                        
                        progress_bar.progress((i + 1) / len(users_to_eval))
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"Recall@{k}", f"{np.mean(recalls):.4f}")
                    with col2:
                        st.metric(f"NDCG@{k}", f"{np.mean(ndcgs):.4f}")
                    with col3:
                        st.metric(f"HitRate@{k}", f"{np.mean(hits):.4f}")
                    
                    st.info(f"Evaluated on {len(users_to_eval)} users")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray;'>
        🎬 CGAN-Based Movie Recommender System | Developed By Saad, Farhan, Kirsh
        </div>
        """,
        unsafe_allow_html=True
    )
    
    


if __name__ == "__main__":
    main()
