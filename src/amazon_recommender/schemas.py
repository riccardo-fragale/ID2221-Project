"""Initial Parquet/JSON contracts. Building schemas does not create a Spark JVM.

Review timestamps are Unix milliseconds; ETL must verify units and reject bad
values. Original identifiers stay strings until training-only mappings exist.
"""


def review_schema():
    from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

    return StructType([
        StructField("user_id", StringType(), True),
        StructField("parent_asin", StringType(), True),
        StructField("rating", DoubleType(), True),
        StructField("timestamp", LongType(), True),
    ])


def interaction_schema():
    from pyspark.sql.types import IntegerType, StructField, TimestampType

    # Nullable at read time: ETL validation must enforce required clean fields.
    return review_schema().add(StructField("event_time", TimestampType(), True)).add(
        StructField("year", IntegerType(), True)
    )


def metadata_schema():
    from pyspark.sql.types import StringType, StructField, StructType

    return StructType([
        StructField("parent_asin", StringType(), True),
        StructField("title", StringType(), True),
    ])


def mapping_schema(entity: str):
    from pyspark.sql.types import IntegerType, StringType, StructField, StructType

    if entity not in ("user", "item"):
        raise ValueError("mapping entity must be user or item")
    original = "user_id" if entity == "user" else "parent_asin"
    return StructType([
        StructField(original, StringType(), False),
        StructField(f"{entity}_index", IntegerType(), False),
    ])


def recommendation_schema():
    from pyspark.sql.types import DoubleType, IntegerType, StringType, StructField, StructType

    # Flat Parquet rows; the publisher later groups ordered items per user.
    return StructType([
        StructField("user_id", StringType(), False),
        StructField("parent_asin", StringType(), False),
        StructField("rank", IntegerType(), False),
        StructField("score", DoubleType(), False),
        StructField("recommendation_type", StringType(), False),
    ])
