# %% [markdown]
# # Movie Recommender Pipeline — Full Training on T4 GPU
#
# **Author:** Yashraj Jadhav  
# **Environment:** Google Colab (T4 GPU)  
# **Dataset:** MovieLens 32M (post-Spark feature engineering)  
#
# This notebook trains both models on the FULL dataset:
# 1. Matrix Factorization baseline
# 2. Neural Collaborative Filtering (two-tower GMF + MLP)
#
# Then evaluates with HR@K, NDCG@K, Coverage and saves results.

# %% [markdown]
# ## Cell 1 — GPU Check & Installs

# %%
import subprocess, sys

# Verify GPU
gpu_info = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(gpu_info.stdout[:500])
assert "T4" in gpu_info.stdout or "GPU" in gpu_info.stdout, "No GPU detected — switch to T4 runtime"

# Install extras (TF is pre-installed on Colab)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyarrow", "boto3"], check=True)

import tensorflow as tf
print(f"\nTensorFlow: {tf.__version__}")
print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
print(f"Num GPUs: {len(tf.config.list_physical_devices('GPU'))}")

# %% [markdown]
# ## Cell 2 — Upload Parquet Features
#
# **Option A (recommended):** Upload the Parquet files from Codespaces.  
# In Codespaces, zip the features:
# ```bash
# cd data/features
# zip -r features.zip interactions/ user_features/ item_features/
# ```
# Then upload `features.zip` to Colab using the file upload below.
#
# **Option B:** Download from S3 (requires AWS credentials).

# %%
import os

# === OPTION A: Upload from Codespaces ===
# Run this cell, click "Choose Files", upload features.zip
from google.colab import files

print("Upload features.zip from Codespaces (data/features/features.zip)")
print("If already uploaded, skip this cell.\n")

if not os.path.exists("/content/features/interactions/train"):
    uploaded = files.upload()  # Upload features.zip

    if "features.zip" in uploaded:
        os.makedirs("/content/features", exist_ok=True)
        import shutil
        shutil.move("features.zip", "/content/features.zip")
        subprocess.run(["unzip", "-o", "/content/features.zip", "-d", "/content/features/"], check=True)
        print("\n✓ Features extracted")
    else:
        print("⚠ No features.zip found in upload. Check filename.")
else:
    print("✓ Features already present, skipping upload.")

# Verify
for split in ["train", "val", "test"]:
    path = f"/content/features/interactions/{split}"
    if os.path.exists(path):
        n_files = len([f for f in os.listdir(path) if f.endswith(".parquet")])
        print(f"  ✓ {split}: {n_files} parquet file(s)")
    else:
        print(f"  ✗ {split}: NOT FOUND")

# %% [markdown]
# ## Cell 3 — Load Data & Inspect

# %%
import numpy as np
import pandas as pd
import time

TRAIN_PATH = "/content/features/interactions/train"
VAL_PATH   = "/content/features/interactions/val"
TEST_PATH  = "/content/features/interactions/test"

print("Loading Parquet splits...")
t0 = time.time()

train_df = pd.read_parquet(TRAIN_PATH)
val_df   = pd.read_parquet(VAL_PATH)
test_df  = pd.read_parquet(TEST_PATH)

print(f"Loaded in {time.time()-t0:.1f}s")
print(f"\nTrain: {len(train_df):>12,} rows")
print(f"Val:   {len(val_df):>12,} rows")
print(f"Test:  {len(test_df):>12,} rows")
print(f"Total: {len(train_df)+len(val_df)+len(test_df):>12,} rows")

# Dataset dimensions
n_users = max(train_df["user_idx"].max(), val_df["user_idx"].max(), test_df["user_idx"].max()) + 1
n_items = max(train_df["item_idx"].max(), val_df["item_idx"].max(), test_df["item_idx"].max()) + 1

print(f"\nn_users: {n_users:,}")
print(f"n_items: {n_items:,}")
print(f"\nTrain columns: {list(train_df.columns)}")
print(train_df.head())

# %% [markdown]
# ## Cell 4 — Negative Sampling (Vectorized)
#
# For implicit feedback, we need negative samples: movies each user did NOT rate.  
# Ratio: 4 negatives per positive (standard from He et al. 2017 NCF paper).
#
# **This is the most memory-intensive step.** Using vectorized approach for speed.

# %%
def generate_negative_samples_fast(df, n_items, neg_ratio=4, seed=42):
    """
    Vectorized negative sampling — much faster than row-by-row iteration.

    For each positive interaction, sample neg_ratio items the user hasn't rated.
    Uses grouped operations to avoid Python loops where possible.

    Args:
        df: DataFrame with user_idx, item_idx
        n_items: Total number of items
        neg_ratio: Negatives per positive
        seed: Random seed

    Returns:
        DataFrame with [user_idx, item_idx, label] columns
    """
    rng = np.random.RandomState(seed)

    # Build positive lookup per user
    print("Building positive item sets...")
    user_positive = df.groupby("user_idx")["item_idx"].apply(set).to_dict()

    # Positives
    positives = df[["user_idx", "item_idx"]].copy()
    positives["label"] = 1.0
    n_pos = len(positives)

    # Generate negatives per user
    print(f"Generating {neg_ratio}x negatives ({n_pos * neg_ratio:,} samples)...")
    t0 = time.time()

    neg_users = []
    neg_items = []

    # Process in chunks for progress reporting
    users = sorted(user_positive.keys())
    total_users = len(users)

    for i, uid in enumerate(users):
        pos_set = user_positive[uid]
        n_neg = len(pos_set) * neg_ratio

        # Oversample then filter — much faster than rejection sampling
        candidates = rng.randint(0, n_items, size=n_neg * 3)
        valid = [c for c in candidates if c not in pos_set][:n_neg]

        # If still short (very active user), fill remaining
        while len(valid) < n_neg:
            extra = rng.randint(0, n_items, size=n_neg)
            valid.extend([c for c in extra if c not in pos_set and c not in set(valid)])
            valid = valid[:n_neg]

        neg_users.extend([uid] * len(valid))
        neg_items.extend(valid)

        if (i + 1) % 20000 == 0:
            elapsed = time.time() - t0
            pct = 100 * (i + 1) / total_users
            print(f"  {pct:.0f}% ({i+1:,}/{total_users:,} users, {elapsed:.0f}s)")

    negatives = pd.DataFrame({
        "user_idx": np.array(neg_users, dtype=np.int32),
        "item_idx": np.array(neg_items, dtype=np.int32),
        "label": np.float32(0.0),
    })

    # Combine and shuffle
    combined = pd.concat([positives[["user_idx", "item_idx", "label"]], negatives], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    elapsed = time.time() - t0
    print(f"\n✓ Negative sampling complete in {elapsed:.0f}s")
    print(f"  Positives: {n_pos:>12,} ({100*n_pos/len(combined):.1f}%)")
    print(f"  Negatives: {len(negatives):>12,} ({100*len(negatives)/len(combined):.1f}%)")
    print(f"  Total:     {len(combined):>12,}")

    return combined


# Generate for train (4:1) and val (1:1 — less negatives for faster validation)
print("=" * 60)
print("TRAIN SET — Negative Sampling (4:1)")
print("=" * 60)
train_sampled = generate_negative_samples_fast(train_df, n_items, neg_ratio=4, seed=42)

print("\n" + "=" * 60)
print("VAL SET — Negative Sampling (1:1)")
print("=" * 60)
val_sampled = generate_negative_samples_fast(val_df, n_items, neg_ratio=1, seed=43)

# %% [markdown]
# ## Cell 5 — Create TF Datasets

# %%
BATCH_SIZE = 4096  # Large batches for stable gradients, fits in T4 VRAM

def make_tf_dataset(df, batch_size, shuffle=True):
    """Convert DataFrame to tf.data.Dataset with batching and prefetch."""
    inputs = np.stack([
        df["user_idx"].values.astype(np.int32),
        df["item_idx"].values.astype(np.int32),
    ], axis=1)
    labels = df["label"].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((inputs, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=100_000, seed=42)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_tf_dataset(train_sampled, BATCH_SIZE, shuffle=True)
val_ds   = make_tf_dataset(val_sampled, BATCH_SIZE, shuffle=False)

# Verify
for inputs, labels in train_ds.take(1):
    print(f"✓ Input shape:  {inputs.shape}")    # (4096, 2)
    print(f"✓ Labels shape: {labels.shape}")     # (4096,)
    print(f"✓ Label mean:   {labels.numpy().mean():.3f} (expect ~0.2 for 4:1 ratio)")

# Free memory — we no longer need the DataFrames
del train_sampled, val_sampled
import gc; gc.collect()

# %% [markdown]
# ## Cell 6 — Define Models
#
# Both model architectures are defined inline (same code as in the repo).

# %%
# ══════════════════════════════════════════════════════════════
# MODEL 1: Matrix Factorization Baseline
# ══════════════════════════════════════════════════════════════

class MatrixFactorization(tf.keras.Model):
    """
    Dot-product matrix factorization with bias terms.
    P(interaction) = sigmoid(global_bias + user_bias + item_bias + dot(u, v))
    """
    def __init__(self, n_users, n_items, embedding_dim=64, l2_reg=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim

        self.user_embedding = tf.keras.layers.Embedding(
            n_users, embedding_dim,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
            name="user_embedding",
        )
        self.item_embedding = tf.keras.layers.Embedding(
            n_items, embedding_dim,
            embeddings_initializer="he_normal",
            embeddings_regularizer=tf.keras.regularizers.l2(l2_reg),
            name="item_embedding",
        )
        self.user_bias = tf.keras.layers.Embedding(n_users, 1, embeddings_initializer="zeros", name="user_bias")
        self.item_bias = tf.keras.layers.Embedding(n_items, 1, embeddings_initializer="zeros", name="item_bias")
        self.global_bias = tf.Variable(0.0, name="global_bias")

    def call(self, inputs, training=False):
        user_ids, item_ids = inputs[:, 0], inputs[:, 1]
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        dot = tf.reduce_sum(user_emb * item_emb, axis=1)
        u_bias = tf.squeeze(self.user_bias(user_ids), axis=1)
        i_bias = tf.squeeze(self.item_bias(item_ids), axis=1)
        return tf.sigmoid(dot + u_bias + i_bias + self.global_bias)

    def get_config(self):
        return {"n_users": self.n_users, "n_items": self.n_items, "embedding_dim": self.embedding_dim}


# ══════════════════════════════════════════════════════════════
# MODEL 2: Neural Collaborative Filtering (Two-Tower)
# ══════════════════════════════════════════════════════════════

class NeuralCollaborativeFiltering(tf.keras.Model):
    """
    Two-tower NCF: GMF (element-wise product) + MLP (deep network) with fusion.
    Reference: He et al., "Neural Collaborative Filtering" (WWW 2017)
    """
    def __init__(self, n_users, n_items, gmf_dim=64, mlp_dim=64,
                 mlp_layers=None, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        if mlp_layers is None:
            mlp_layers = [128, 64, 32]

        self.n_users = n_users
        self.n_items = n_items
        self.gmf_dim = gmf_dim
        self.mlp_dim = mlp_dim
        self.mlp_layer_sizes = mlp_layers
        self.dropout_rate = dropout_rate

        # GMF Tower
        self.gmf_user_emb = tf.keras.layers.Embedding(n_users, gmf_dim, embeddings_initializer="he_normal", name="gmf_user_embedding")
        self.gmf_item_emb = tf.keras.layers.Embedding(n_items, gmf_dim, embeddings_initializer="he_normal", name="gmf_item_embedding")

        # MLP Tower
        self.mlp_user_emb = tf.keras.layers.Embedding(n_users, mlp_dim, embeddings_initializer="he_normal", name="mlp_user_embedding")
        self.mlp_item_emb = tf.keras.layers.Embedding(n_items, mlp_dim, embeddings_initializer="he_normal", name="mlp_item_embedding")

        self.mlp_dense_layers = []
        self.mlp_bn_layers = []
        self.mlp_dropout_layers = []
        for i, units in enumerate(mlp_layers):
            self.mlp_dense_layers.append(tf.keras.layers.Dense(units, activation="relu", name=f"mlp_dense_{i}"))
            self.mlp_bn_layers.append(tf.keras.layers.BatchNormalization(name=f"mlp_bn_{i}"))
            self.mlp_dropout_layers.append(tf.keras.layers.Dropout(dropout_rate, name=f"mlp_dropout_{i}"))

        # Fusion
        self.output_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")

    def call(self, inputs, training=False):
        user_ids, item_ids = inputs[:, 0], inputs[:, 1]

        # GMF Tower — element-wise (Hadamard) product
        gmf_user = self.gmf_user_emb(user_ids)
        gmf_item = self.gmf_item_emb(item_ids)
        gmf_output = gmf_user * gmf_item

        # MLP Tower — concatenate then feed through layers
        mlp_user = self.mlp_user_emb(user_ids)
        mlp_item = self.mlp_item_emb(item_ids)
        x = tf.concat([mlp_user, mlp_item], axis=1)

        for dense, bn, dropout in zip(self.mlp_dense_layers, self.mlp_bn_layers, self.mlp_dropout_layers):
            x = dense(x)
            x = bn(x, training=training)
            x = dropout(x, training=training)

        # Fusion — concatenate both towers, final prediction
        combined = tf.concat([gmf_output, x], axis=1)
        output = self.output_layer(combined)
        return tf.squeeze(output, axis=1)

    def get_config(self):
        return {"n_users": self.n_users, "n_items": self.n_items,
                "gmf_dim": self.gmf_dim, "mlp_dim": self.mlp_dim,
                "mlp_layers": self.mlp_layer_sizes, "dropout_rate": self.dropout_rate}


# Quick shape check
dummy = tf.constant([[0, 0], [1, 1]], dtype=tf.int32)
mf_test = MatrixFactorization(n_users, n_items, 64)
ncf_test = NeuralCollaborativeFiltering(n_users, n_items, 64, 64, [128, 64, 32], 0.2)
print(f"✓ MF output shape:  {mf_test(dummy).shape}")   # (2,)
print(f"✓ NCF output shape: {ncf_test(dummy).shape}")   # (2,)

mf_params = (n_users + n_items) * 64 + (n_users + n_items) + 1
ncf_params = (n_users + n_items) * 64 * 2 + 128*128 + 128 + 128*64 + 64 + 64*32 + 32 + 97
print(f"\nMF  total params:  ~{mf_params:,}")
print(f"NCF total params:  ~{ncf_params:,}")

del mf_test, ncf_test

# %% [markdown]
# ## Cell 7 — Train Matrix Factorization Baseline

# %%
print("=" * 70)
print("TRAINING: Matrix Factorization Baseline")
print("=" * 70)

MF_CONFIG = {
    "embedding_dim": 64,
    "lr": 0.001,
    "epochs": 15,
    "patience": 3,
}

mf_model = MatrixFactorization(n_users, n_items, embedding_dim=MF_CONFIG["embedding_dim"])

mf_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=MF_CONFIG["lr"]),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

mf_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=MF_CONFIG["patience"],
        restore_best_weights=True, verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1,
    ),
]

t0 = time.time()
mf_history = mf_model.fit(
    train_ds, validation_data=val_ds,
    epochs=MF_CONFIG["epochs"], callbacks=mf_callbacks, verbose=1,
)
mf_train_time = time.time() - t0

print(f"\n✓ MF Training complete in {mf_train_time:.0f}s ({mf_train_time/60:.1f} min)")
print(f"  Best val_loss: {min(mf_history.history['val_loss']):.4f}")
print(f"  Final val_auc: {mf_history.history['val_auc'][-1]:.4f}")

# %% [markdown]
# ## Cell 8 — Train Neural Collaborative Filtering

# %%
print("=" * 70)
print("TRAINING: Neural Collaborative Filtering (Two-Tower)")
print("=" * 70)

NCF_CONFIG = {
    "gmf_dim": 64,
    "mlp_dim": 64,
    "mlp_layers": [128, 64, 32],
    "dropout_rate": 0.2,
    "lr": 0.001,
    "epochs": 15,
    "patience": 3,
}

ncf_model = NeuralCollaborativeFiltering(
    n_users, n_items,
    gmf_dim=NCF_CONFIG["gmf_dim"],
    mlp_dim=NCF_CONFIG["mlp_dim"],
    mlp_layers=NCF_CONFIG["mlp_layers"],
    dropout_rate=NCF_CONFIG["dropout_rate"],
)

ncf_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=NCF_CONFIG["lr"]),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

ncf_callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=NCF_CONFIG["patience"],
        restore_best_weights=True, verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1,
    ),
]

t0 = time.time()
ncf_history = ncf_model.fit(
    train_ds, validation_data=val_ds,
    epochs=NCF_CONFIG["epochs"], callbacks=ncf_callbacks, verbose=1,
)
ncf_train_time = time.time() - t0

print(f"\n✓ NCF Training complete in {ncf_train_time:.0f}s ({ncf_train_time/60:.1f} min)")
print(f"  Best val_loss: {min(ncf_history.history['val_loss']):.4f}")
print(f"  Final val_auc: {ncf_history.history['val_auc'][-1]:.4f}")

# %% [markdown]
# ## Cell 9 — Evaluation (HR@K, NDCG@K, Coverage)
#
# Standard evaluation protocol from He et al. (2017):
# - For each test user, score 1 positive + 99 random negatives = 100 candidates
# - Check if the positive item ranks in top K
# - Evaluate on up to 5,000 users for tractable runtime

# %%
# ══════════════════════════════════════════════════════════════
# Evaluation Functions
# ══════════════════════════════════════════════════════════════

N_EVAL_NEGATIVES = 99
MAX_EVAL_USERS = 5000
EVAL_BATCH_SIZE = 4096
K_VALUES = [5, 10, 20]
SEED = 42


def build_user_positive_items(train_df, val_df):
    """Build dict: user_idx -> set of item_idx they've interacted with."""
    user_items = {}
    for df in [train_df, val_df]:
        for uid, iid in zip(df["user_idx"].values, df["item_idx"].values):
            uid, iid = int(uid), int(iid)
            if uid not in user_items:
                user_items[uid] = set()
            user_items[uid].add(iid)
    return user_items


def evaluate_model(model, test_df, n_items, user_positive_items,
                   k_values=K_VALUES, max_users=MAX_EVAL_USERS):
    """
    Compute HR@K and NDCG@K for a model.

    Protocol: For each test user, rank 100 items (1 true + 99 negative).
    """
    rng = np.random.RandomState(SEED)

    # One test item per user (first chronologically, since temporal split)
    test_users = test_df.groupby("user_idx")["item_idx"].first().reset_index()
    if len(test_users) > max_users:
        test_users = test_users.sample(max_users, random_state=SEED)

    print(f"Evaluating on {len(test_users):,} test users...")
    t0 = time.time()

    # Pre-allocate arrays for batch scoring
    all_hits = {k: 0 for k in k_values}
    all_ndcg = {k: [] for k in k_values}
    total = 0

    for idx, (_, row) in enumerate(test_users.iterrows()):
        user_id = int(row["user_idx"])
        true_item = int(row["item_idx"])
        positives = user_positive_items.get(user_id, set())

        # Sample 99 negatives (not in user's history)
        negs = []
        exclude = positives | {true_item}
        while len(negs) < N_EVAL_NEGATIVES:
            cands = rng.randint(0, n_items, size=N_EVAL_NEGATIVES * 2)
            for c in cands:
                if c not in exclude and c not in negs:
                    negs.append(c)
                    if len(negs) >= N_EVAL_NEGATIVES:
                        break

        candidates = negs + [true_item]  # true item at index 99
        user_ids = np.full(100, user_id, dtype=np.int32)
        item_ids = np.array(candidates, dtype=np.int32)
        inputs = np.stack([user_ids, item_ids], axis=1)
        scores = model.predict(inputs, batch_size=EVAL_BATCH_SIZE, verbose=0).flatten()

        # Rank by score (descending)
        ranked = np.argsort(scores)[::-1]
        true_pos = np.where(ranked == 99)[0][0]  # Position of true item

        for k in k_values:
            # HR@K
            if true_pos < k:
                all_hits[k] += 1
                all_ndcg[k].append(1.0 / np.log2(true_pos + 2))
            else:
                all_ndcg[k].append(0.0)

        total += 1

        if (idx + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  {idx+1:,}/{len(test_users):,} users ({elapsed:.0f}s)")

    results = {}
    for k in k_values:
        results[f"HR@{k}"] = round(all_hits[k] / total, 4) if total > 0 else 0
        results[f"NDCG@{k}"] = round(float(np.mean(all_ndcg[k])), 4) if all_ndcg[k] else 0

    elapsed = time.time() - t0
    results["eval_time_seconds"] = round(elapsed, 1)
    print(f"\n✓ Evaluation complete in {elapsed:.0f}s")
    return results


def compute_coverage(model, n_users, n_items, k=10, sample_users=500):
    """What fraction of items ever appear in any user's top-K?"""
    rng = np.random.RandomState(SEED)
    sampled = rng.choice(n_users, size=min(sample_users, n_users), replace=False)
    recommended = set()
    all_items = np.arange(n_items, dtype=np.int32)

    for i, uid in enumerate(sampled):
        user_ids = np.full(n_items, uid, dtype=np.int32)
        inputs = np.stack([user_ids, all_items], axis=1)
        scores = model.predict(inputs, batch_size=EVAL_BATCH_SIZE, verbose=0).flatten()
        top_k = np.argsort(scores)[-k:]
        recommended.update(top_k.tolist())

        if (i + 1) % 100 == 0:
            print(f"  Coverage: {i+1}/{len(sampled)} users scored")

    return round(len(recommended) / n_items, 4)


# ══════════════════════════════════════════════════════════════
# Run Evaluation
# ══════════════════════════════════════════════════════════════

print("Building user positive item lookup...")
user_pos = build_user_positive_items(train_df, val_df)

print("\n" + "=" * 70)
print("EVALUATING: Matrix Factorization Baseline")
print("=" * 70)
mf_results = evaluate_model(mf_model, test_df, n_items, user_pos)
print(f"MF Results: {mf_results}")

print("\nComputing MF coverage...")
mf_results["Coverage@10"] = compute_coverage(mf_model, n_users, n_items, k=10, sample_users=500)
print(f"MF Coverage@10: {mf_results['Coverage@10']}")

print("\n" + "=" * 70)
print("EVALUATING: Neural Collaborative Filtering")
print("=" * 70)
ncf_results = evaluate_model(ncf_model, test_df, n_items, user_pos)
print(f"NCF Results: {ncf_results}")

print("\nComputing NCF coverage...")
ncf_results["Coverage@10"] = compute_coverage(ncf_model, n_users, n_items, k=10, sample_users=500)
print(f"NCF Coverage@10: {ncf_results['Coverage@10']}")

# %% [markdown]
# ## Cell 10 — Results Comparison Table

# %%
print("\n" + "=" * 75)
print("MODEL COMPARISON — Recommendation Metrics (Full Training)")
print("=" * 75)
print(f"{'Metric':<20} {'MF Baseline':>15} {'NCF':>15} {'Δ (NCF vs MF)':>20}")
print("-" * 75)

metrics_order = ["HR@5", "HR@10", "HR@20", "NDCG@5", "NDCG@10", "NDCG@20", "Coverage@10"]

for metric in metrics_order:
    mf_val = mf_results.get(metric)
    ncf_val = ncf_results.get(metric)

    if mf_val is not None and ncf_val is not None:
        delta = ncf_val - mf_val
        delta_pct = 100 * delta / mf_val if mf_val > 0 else 0
        delta_str = f"{delta:+.4f} ({delta_pct:+.1f}%)"
    else:
        delta_str = "N/A"

    mf_str = f"{mf_val:.4f}" if mf_val is not None else "N/A"
    ncf_str = f"{ncf_val:.4f}" if ncf_val is not None else "N/A"
    print(f"{metric:<20} {mf_str:>15} {ncf_str:>15} {delta_str:>20}")

print("-" * 75)
print(f"{'Training Time':<20} {mf_train_time:>14.0f}s {ncf_train_time:>14.0f}s")
print(f"{'Epochs Trained':<20} {len(mf_history.history['loss']):>15} {len(ncf_history.history['loss']):>15}")
print(f"{'Best Val Loss':<20} {min(mf_history.history['val_loss']):>15.4f} {min(ncf_history.history['val_loss']):>15.4f}")
print(f"{'Final Val AUC':<20} {mf_history.history['val_auc'][-1]:>15.4f} {ncf_history.history['val_auc'][-1]:>15.4f}")
print("=" * 75)

# %% [markdown]
# ## Cell 11 — Save Results & Models
#
# Save everything so you can:
# 1. Update the README metrics table
# 2. Upload model weights to S3
# 3. Take a screenshot of the comparison table

# %%
import json

# Save results JSON
all_results = {
    "dataset": "movielens-32m",
    "n_users": int(n_users),
    "n_items": int(n_items),
    "train_rows": int(len(train_df)),
    "val_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
    "neg_ratio_train": 4,
    "neg_ratio_val": 1,
    "mf_config": MF_CONFIG,
    "ncf_config": NCF_CONFIG,
    "mf_results": mf_results,
    "ncf_results": ncf_results,
    "mf_training": {
        "epochs_trained": len(mf_history.history["loss"]),
        "training_time_seconds": round(mf_train_time, 1),
        "best_val_loss": round(min(mf_history.history["val_loss"]), 4),
        "final_val_auc": round(mf_history.history["val_auc"][-1], 4),
    },
    "ncf_training": {
        "epochs_trained": len(ncf_history.history["loss"]),
        "training_time_seconds": round(ncf_train_time, 1),
        "best_val_loss": round(min(ncf_history.history["val_loss"]), 4),
        "final_val_auc": round(ncf_history.history["val_auc"][-1], 4),
    },
}

os.makedirs("/content/results", exist_ok=True)

with open("/content/results/full_training_results.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("✓ Results saved to /content/results/full_training_results.json")

# Save model weights
mf_model.save_weights("/content/results/mf_baseline_weights/weights")
print("✓ MF weights saved")

ncf_model.save_weights("/content/results/ncf_v1_weights/weights")
print("✓ NCF weights saved")

# Save training history
mf_hist = {k: [float(v) for v in vals] for k, vals in mf_history.history.items()}
ncf_hist = {k: [float(v) for v in vals] for k, vals in ncf_history.history.items()}

with open("/content/results/mf_history.json", "w") as f:
    json.dump(mf_hist, f, indent=2)
with open("/content/results/ncf_history.json", "w") as f:
    json.dump(ncf_hist, f, indent=2)
print("✓ Training histories saved")

# Download results
print("\n📥 Downloading results JSON...")
files.download("/content/results/full_training_results.json")

# %% [markdown]
# ## Cell 12 — Generate README Metrics Table
#
# Copy this directly into your README.md

# %%
print("\n### Copy this into README.md:\n")
print("```")
print(f"| Model                      | HR@5   | HR@10  | HR@20  | NDCG@10 | Coverage@10 |")
print(f"|----------------------------|--------|--------|--------|---------|-------------|")

for name, res in [("Matrix Factorization (MF)", mf_results), ("Neural CF (NCF)", ncf_results)]:
    hr5 = res.get("HR@5", 0)
    hr10 = res.get("HR@10", 0)
    hr20 = res.get("HR@20", 0)
    ndcg10 = res.get("NDCG@10", 0)
    cov = res.get("Coverage@10", 0)
    print(f"| {name:<26} | {hr5:.4f} | {hr10:.4f} | {hr20:.4f} | {ndcg10:.4f}  | {cov:.4f}      |")

print("```")

# Compute NCF improvement
if mf_results.get("NDCG@10") and mf_results["NDCG@10"] > 0:
    ndcg_improvement = 100 * (ncf_results["NDCG@10"] - mf_results["NDCG@10"]) / mf_results["NDCG@10"]
    print(f"\nNCF outperforms MF by {ndcg_improvement:+.1f}% on NDCG@10")

if mf_results.get("HR@10") and mf_results["HR@10"] > 0:
    hr_improvement = 100 * (ncf_results["HR@10"] - mf_results["HR@10"]) / mf_results["HR@10"]
    print(f"NCF outperforms MF by {hr_improvement:+.1f}% on HR@10")

# %% [markdown]
# ## Cell 13 — Training Curves Plot

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curves
axes[0].plot(mf_history.history["loss"], label="MF Train", linestyle="--", color="tab:blue")
axes[0].plot(mf_history.history["val_loss"], label="MF Val", color="tab:blue")
axes[0].plot(ncf_history.history["loss"], label="NCF Train", linestyle="--", color="tab:orange")
axes[0].plot(ncf_history.history["val_loss"], label="NCF Val", color="tab:orange")
axes[0].set_title("Loss (Binary Cross-Entropy)")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# AUC curves
axes[1].plot(mf_history.history["auc"], label="MF Train", linestyle="--", color="tab:blue")
axes[1].plot(mf_history.history["val_auc"], label="MF Val", color="tab:blue")
axes[1].plot(ncf_history.history["auc"], label="NCF Train", linestyle="--", color="tab:orange")
axes[1].plot(ncf_history.history["val_auc"], label="NCF Val", color="tab:orange")
axes[1].set_title("AUC")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("AUC")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Metrics comparison bar chart
metrics = ["HR@5", "HR@10", "HR@20", "NDCG@10", "Coverage@10"]
mf_vals = [mf_results.get(m, 0) for m in metrics]
ncf_vals = [ncf_results.get(m, 0) for m in metrics]

x = np.arange(len(metrics))
width = 0.35
axes[2].bar(x - width/2, mf_vals, width, label="MF Baseline", color="tab:blue", alpha=0.8)
axes[2].bar(x + width/2, ncf_vals, width, label="NCF", color="tab:orange", alpha=0.8)
axes[2].set_title("Model Comparison")
axes[2].set_xticks(x)
axes[2].set_xticklabels(metrics, rotation=15)
axes[2].set_ylabel("Score")
axes[2].legend()
axes[2].grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("/content/results/model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("✓ Chart saved to /content/results/model_comparison.png")

# Download chart
files.download("/content/results/model_comparison.png")

# %% [markdown]
# ## Done!
#
# **Next steps back in Codespaces:**
# 1. Upload `full_training_results.json` to repo: `docs/results/`
# 2. Upload `model_comparison.png` to repo: `docs/screenshots/`
# 3. Update README.md metrics table with real numbers from Cell 12
# 4. Update resume bullet: "outperformed MF baseline by X% on NDCG@10"
# 5. Commit: `git add . && git commit -m "Full training results (T4 GPU)" && git push`
