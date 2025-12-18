# 🎬 CGAN-Based Recommendation System (MovieLens 100K)

This project implements a **Collaborative Generative Adversarial Network (CGAN)** for building a **user–item recommendation system** using the **MovieLens 100K** dataset.

Unlike traditional matrix factorization–based recommenders, this system frames recommendation as an **adversarial learning problem**, where:
- a **Generator** proposes plausible items for a user, and
- a **Discriminator** judges whether a user–item interaction looks real or generated.

The project supports:
- end-to-end training on MovieLens 100K,
- human-readable movie recommendations (movie title + ID),
- and an **interactive new-user flow**, where a new user rates a few movies and receives personalized recommendations.

---

## 📌 High-Level Overview

- **Domain**: Recommender Systems  
- **Dataset**: MovieLens 100K  
- **Feedback Type**: Implicit (ratings ≥ 4 treated as positive)  
- **Model**: Collaborative GAN (CGAN)  
- **Framework**: PyTorch  
- **Output**: Top-K movie recommendations excluding previously interacted items  

---

## 🚀 How to Run

### 1️⃣ Install dependencies
```bash
pip install torch numpy
```

> Python 3.9+ recommended

---

### 2️⃣ Dataset setup
Movielens dataset ml-100k already setup. See ml-100k-README for details about the datset.

---

### 3️⃣ Run the project

#### Normal training + recommendations
```bash
python main.py --epochs 10 --k 10
```

#### Training with an interactive new user
```bash
python main.py --epochs 10 --k 10 --interactive_new_user
```

You will be prompted to:
- search movies by title,
- rate movies (1–5),
- and receive Top-K recommendations after training.

---

## 🧠 In-Depth Project Explanation

### 1. Dataset Processing

The MovieLens 100K dataset consists of explicit ratings:
```
(userId, movieId, rating, timestamp)
```

We convert it to **implicit feedback**:
- rating ≥ 4 → positive interaction (1)
- otherwise → ignored

Each user is represented as:
```
user_pos[u] = {items liked by user u}
```

User and item IDs are remapped to contiguous indices to support embedding layers.

---

### 2. CGAN Architecture

#### Generator (G)
- Input: user embedding + random noise
- Output: scores over all items
- Role: generate items that *could plausibly* be liked by the user

Mathematically:
```
G(u, z) → item scores
```

---

#### Discriminator (D)
- Input: user embedding + item embedding
- Output: probability that the interaction is real
- Role: distinguish real user–item interactions from generated ones

Mathematically:
```
D(u, i) → P(real)
```

---

### 3. Training Objective

The model is trained adversarially:

- **Discriminator loss**
  - real interactions → label 1
  - fake interactions → label 0

- **Generator loss**
  - tries to fool the discriminator into predicting 1 for generated items

Binary Cross-Entropy (BCE) loss is used for both networks.

To stabilize training:
- generator-based negatives are mixed with random negative samples,
- mini-batch training is used.

---

### 4. Recommendation Generation

For a user `u`:
1. The generator produces item scores.
2. Items already interacted with (training set) are filtered out.
3. The top-K highest-scoring unseen items are returned.

To reduce randomness from GAN noise, scores are averaged across multiple noise samples.

---

### 5. Human-Readable Output

Movie recommendations are mapped back to:
```
movieId | movie title
```

using metadata from `u.item`, making results interpretable and presentable.

---

### 6. New User (Cold-Start) Handling

This project supports a **practical cold-start solution**:

1. A new user rates a few movies *before training*.
2. These ratings are added to the training set.
3. The CGAN is trained including this new user.
4. Recommendations are generated excluding the movies they already rated.

This avoids unstable post-hoc embedding fine-tuning and keeps the pipeline simple and robust.

---

## 📊 Evaluation Metrics

Evaluation is performed on held-out test interactions using:
- **Recall@K**
- **NDCG@K**
- **HitRate@K**

Only users with at least one train and test interaction are evaluated.

---

## 📁 Project Structure

```
.
├── main.py
├── README.md
├── (movielens dataset files)    
└── requirements.txt  
```

---

## 🔍 Notes & Limitations

- This implementation is **research-inspired**, not a full paper reproduction.
- Discrete item sampling is approximated via multinomial sampling.
- More advanced GAN techniques (e.g. Gumbel-Softmax, policy gradients) are intentionally avoided for clarity.

The goal is **conceptual correctness, interpretability, and reproducibility**.

---

## 🏁 Final Remarks

This project demonstrates how **adversarial learning** can be applied to recommender systems in a clean, end-to-end manner using a well-known dataset.

It is suitable for:
- academic coursework,
- research prototypes,
- and learning GAN-based recommendation techniques.
