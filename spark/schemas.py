"""
Spark Schema Definitions
========================
StructType schemas for all DataFrames in the MovieLens 32M pipeline.

Defining schemas explicitly (rather than letting Spark infer) ensures:
1. Consistent types across pipeline runs
2. Faster CSV loading (no inference pass over data)
3. Early failure on schema mismatches
4. Self-documenting data contracts
"""

from pyspark.sql.types import (
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

# ── Raw Data Schemas ───────────────────────────────────────────────────

RATINGS_SCHEMA = StructType([
    StructField("userId", IntegerType(), nullable=False),
    StructField("movieId", IntegerType(), nullable=False),
    StructField("rating", FloatType(), nullable=False),
    StructField("timestamp", LongType(), nullable=False),
])
"""
ratings.csv — 32M rows, primary table driving the entire pipeline.
- userId:    Anonymized user identifier (1 to 200948)
- movieId:   MovieLens movie identifier
- rating:    0.5 to 5.0 in 0.5 increments
- timestamp: Seconds since epoch (Unix time)
"""

MOVIES_SCHEMA = StructType([
    StructField("movieId", IntegerType(), nullable=False),
    StructField("title", StringType(), nullable=True),
    StructField("genres", StringType(), nullable=True),
])
"""
movies.csv — 87K rows, metadata enrichment.
- movieId: Matches ratings.csv
- title:   Movie title with year, e.g. "Toy Story (1995)"
- genres:  Pipe-delimited, e.g. "Adventure|Animation|Children"
"""

TAGS_SCHEMA = StructType([
    StructField("userId", IntegerType(), nullable=False),
    StructField("movieId", IntegerType(), nullable=False),
    StructField("tag", StringType(), nullable=True),
    StructField("timestamp", LongType(), nullable=False),
])
"""
tags.csv — Optional, low priority.
- Free-text user tags for movies.
"""

LINKS_SCHEMA = StructType([
    StructField("movieId", IntegerType(), nullable=False),
    StructField("imdbId", StringType(), nullable=True),
    StructField("tmdbId", IntegerType(), nullable=True),
])
"""
links.csv — Optional, for future enrichment via IMDB/TMDB APIs.
"""

# ── Engineered Feature Schemas ─────────────────────────────────────────

USER_FEATURES_SCHEMA = StructType([
    StructField("userId", IntegerType(), nullable=False),
    StructField("user_avg_rating", DoubleType(), nullable=True),
    StructField("user_rating_count", IntegerType(), nullable=True),
    StructField("user_rating_stddev", DoubleType(), nullable=True),
    StructField("user_active_days", IntegerType(), nullable=True),
    StructField("user_genre_diversity", IntegerType(), nullable=True),
    StructField("user_avg_timestamp", DoubleType(), nullable=True),
    StructField("user_positive_ratio", DoubleType(), nullable=True),
])
"""
User-level behavioral features computed by PySpark.
- user_avg_rating:      Mean rating across all rated movies
- user_rating_count:    Total number of ratings
- user_rating_stddev:   Rating standard deviation (0 for single-rating users)
- user_active_days:     Days between first and last rating
- user_genre_diversity: Count of distinct genres rated
- user_avg_timestamp:   Mean timestamp (proxy for activity recency)
- user_positive_ratio:  Fraction of ratings >= 4.0
"""

ITEM_FEATURES_SCHEMA = StructType([
    StructField("movieId", IntegerType(), nullable=False),
    StructField("item_avg_rating", DoubleType(), nullable=True),
    StructField("item_rating_count", IntegerType(), nullable=True),
    StructField("item_rating_stddev", DoubleType(), nullable=True),
    StructField("item_genre_count", IntegerType(), nullable=True),
    StructField("item_recency_score", DoubleType(), nullable=True),
    StructField("item_popularity_rank", IntegerType(), nullable=True),
])
"""
Item-level features computed by PySpark.
- item_avg_rating:      Mean rating received
- item_rating_count:    Total number of ratings received
- item_rating_stddev:   Rating standard deviation
- item_genre_count:     Number of genres assigned
- item_recency_score:   Exponential decay based on last rating time
- item_popularity_rank: Dense rank by rating count (1 = most popular)
"""


# ── Quick validation when run directly ─────────────────────────────────
if __name__ == "__main__":
    schemas = {
        "RATINGS_SCHEMA": RATINGS_SCHEMA,
        "MOVIES_SCHEMA": MOVIES_SCHEMA,
        "TAGS_SCHEMA": TAGS_SCHEMA,
        "LINKS_SCHEMA": LINKS_SCHEMA,
        "USER_FEATURES_SCHEMA": USER_FEATURES_SCHEMA,
        "ITEM_FEATURES_SCHEMA": ITEM_FEATURES_SCHEMA,
    }

    for name, schema in schemas.items():
        fields = [f"{f.name}:{f.dataType.simpleString()}" for f in schema.fields]
        print(f"✓ {name} — {len(schema.fields)} fields: {', '.join(fields)}")

    print(f"\n✓ All {len(schemas)} schemas defined successfully")
