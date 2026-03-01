"""
Spark Session Configuration
===========================
Factory function for creating a SparkSession optimized for the
MovieLens 32M recommendation pipeline.

Environment: GitHub Codespaces (2-core, 8GB RAM, 32GB disk)
"""

import logging
import os

from pyspark.sql import SparkSession

logger = logging.getLogger(__name__)

# ── Spark Configuration Constants ──────────────────────────────────────
SPARK_APP_NAME = "MovieRecommender"
SPARK_MASTER = "local[*]"          # Use all available cores (2 in Codespaces)
SPARK_DRIVER_MEMORY = "3g"         # Codespaces has 8GB; 3g for Spark + headroom for OS + Python
SPARK_SHUFFLE_PARTITIONS = "8"     # Default 200 is overkill for single-node; 8 ≈ cores × 4
SPARK_SERIALIZER = "org.apache.spark.serializer.KryoSerializer"
SPARK_PARQUET_COMPRESSION = "snappy"


def get_spark_session(
    app_name: str = SPARK_APP_NAME,
    driver_memory: str = SPARK_DRIVER_MEMORY,
    shuffle_partitions: str = SPARK_SHUFFLE_PARTITIONS,
) -> SparkSession:
    """
    Create and return a configured SparkSession.

    Why these settings:
    - local[*]: Uses all available cores in Codespaces (2-core machine)
    - 4g driver memory: Codespaces gives 8GB RAM; leave headroom for OS + Python
    - 8 shuffle partitions: Default 200 is overkill for single-node; 8 matches core count × 4
    - KryoSerializer: Faster than Java default serializer for shuffles
    - snappy compression: Fast read/write with reasonable compression ratio for Parquet

    Args:
        app_name: Spark application name (visible in Spark UI)
        driver_memory: Driver memory allocation (e.g., "4g")
        shuffle_partitions: Number of shuffle partitions (string)

    Returns:
        Configured SparkSession instance
    """
    # Ensure JAVA_HOME is set to Java 17 (Java 25 breaks Spark 3.5)
    java_home = os.environ.get("JAVA_HOME", "")
    if "java-17" not in java_home:
        os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-17-openjdk-amd64"
        logger.warning(
            "JAVA_HOME was not set to Java 17. "
            "Overriding to /usr/lib/jvm/java-17-openjdk-amd64"
        )

    logger.info(
        "Creating SparkSession: app=%s, memory=%s, partitions=%s",
        app_name, driver_memory, shuffle_partitions,
    )

    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(SPARK_MASTER)
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", shuffle_partitions)
        .config("spark.serializer", SPARK_SERIALIZER)
        .config("spark.sql.parquet.compression.codec", SPARK_PARQUET_COMPRESSION)
        .config("spark.ui.showConsoleProgress", "true")
        # Reduce logging noise in Codespaces
        .config("spark.ui.enabled", "false")
        .config("spark.driver.extraJavaOptions", "-Xss4m")
        .getOrCreate()
    )

    # Suppress noisy Spark logs (keep WARN and above)
    spark.sparkContext.setLogLevel("WARN")

    logger.info("SparkSession created successfully — Spark version: %s", spark.version)

    return spark


def stop_spark(spark: SparkSession) -> None:
    """
    Cleanly shut down a SparkSession.

    Args:
        spark: Active SparkSession to stop
    """
    if spark is not None:
        spark.stop()
        logger.info("SparkSession stopped.")


# ── Quick validation when run directly ─────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    )

    spark = get_spark_session()
    print(f"✓ SparkSession created — version {spark.version}")
    print(f"  App name:           {spark.sparkContext.appName}")
    print(f"  Master:             {spark.sparkContext.master}")
    print(f"  Driver memory:      {SPARK_DRIVER_MEMORY}")
    print(f"  Shuffle partitions: {SPARK_SHUFFLE_PARTITIONS}")
    print(f"  Serializer:         {SPARK_SERIALIZER}")
    print(f"  Parquet codec:      {SPARK_PARQUET_COMPRESSION}")

    # Smoke test: create a tiny DataFrame
    test_df = spark.createDataFrame([(1, "test")], ["id", "value"])
    assert test_df.count() == 1, "Smoke test failed"
    print(f"✓ Smoke test passed — DataFrame operations work")

    stop_spark(spark)
    print(f"✓ SparkSession stopped cleanly")