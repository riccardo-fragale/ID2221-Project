"""Item-details stage: display fields for items whose metadata has no title.

ETL keeps only `parent_asin` and `title`, and 313,873 reviewed items have no title. This stage
reads the same raw metadata file once more (re-verified against run.json) and writes, per item,
the fields the dashboard uses to describe an untitled item. ETL and its outputs are untouched.

Normalization contract (raw `details` comes in two families):
- directors: `Directors` (a list, Prime Video records) or, if absent or empty, `Director` (a
  string, physical editions); starring: `Starring` (list) or else `Actors` (string).
- Lists are never split. Strings are split on commas. In both, a suffix token (Jr., Jr, Sr., Sr,
  II, III, IV; any case) is re-attached to the previous name: "Robert Downey, Jr., Gwyneth Paltrow"
  gives two names, and so does the list ["Robert Downey", "Jr.", "Gwyneth Paltrow"] (591 raw lists
  arrive split that way). This is a heuristic: a name containing another comma is split.
- Names are whitespace-trimmed, deduplicated in order (ignoring case and spacing; characters are
  otherwise kept as is), and capped at the first 3.
- studio: the first element of `Studio` (list) or the string itself.
- image: the first `MAIN` image (else the first image), at 360w, else 480w, large, thumb.
- An item listed more than once keeps the row with the largest rating_number (nulls last), then
  the smallest JSON of its extracted fields, so the choice is deterministic.

Output: `<run>/item_details/` (completed by the `.partial` rename) with `items/` (parent_asin,
main_category, directors, starring, studio, image, rating_number) and `stage.json` with counts.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

from amazon_recommender.artifacts import file_identity
from amazon_recommender.config import ProjectConfig
from amazon_recommender.etl import CORRUPT, timed

NAME_LIMIT = 3
SUFFIXES = {"jr.", "jr", "sr.", "sr", "ii", "iii", "iv"}
IMAGE_SIZES = ("360w", "480w", "large", "thumb")
FIELDS = ("main_category", "directors", "starring", "studio", "image", "rating_number")


def _as_list(value: str | None) -> list | None:
    """A details value from a list family, kept as JSON text by the reader; a bare string is one entry."""
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except ValueError:
        return [value]
    return [str(entry) for entry in parsed] if isinstance(parsed, list) else [str(parsed)]


def split_names(text: str) -> list[str]:
    return rejoin_suffixes(text.split(","))


def rejoin_suffixes(parts) -> list[str]:
    names = []
    for part in (piece.strip() for piece in parts):
        if not part:
            continue
        if names and part.lower() in SUFFIXES:
            names[-1] = f"{names[-1]}, {part}"
        else:
            names.append(part)
    return names


def unique_names(names: list[str], limit: int = NAME_LIMIT) -> list[str]:
    result, seen = [], set()
    for name in names:
        name = " ".join(name.split())
        if name and name.casefold() not in seen:
            seen.add(name.casefold())
            result.append(name)
    return result[:limit]


def people(details: dict, list_key: str, string_key: str) -> list[str]:
    listed = unique_names(rejoin_suffixes(_as_list(details.get(list_key)) or []))
    if listed:
        return listed  # an empty list falls back to the string family
    text = details.get(string_key)
    return unique_names(split_names(text)) if text else []


def studio(details: dict) -> str | None:
    value = _as_list(details.get("Studio"))
    names = unique_names(value or [], limit=1)
    return names[0] if names else None


def main_image(images) -> str | None:
    images = [image.asDict() if hasattr(image, "asDict") else image for image in images or [] if image]
    if not images:
        return None
    chosen = next((image for image in images if image.get("variant") == "MAIN"), images[0])
    return next((chosen[size] for size in IMAGE_SIZES if chosen.get(size)), None)


def extract(details, images) -> dict:
    details = dict(details or {})
    return {"directors": people(details, "Directors", "Director"),
            "starring": people(details, "Starring", "Actors"),
            "studio": studio(details), "image": main_image(images)}


def raw_schema():
    from pyspark.sql.types import ArrayType, LongType, MapType, StringType, StructField, StructType

    image = StructType([StructField(name, StringType(), True) for name in ("variant", *IMAGE_SIZES)])
    return StructType([
        StructField("parent_asin", StringType(), True),
        StructField("main_category", StringType(), True),
        StructField("rating_number", LongType(), True),
        StructField("details", MapType(StringType(), StringType()), True),  # list values arrive as JSON text
        StructField("images", ArrayType(image), True),
        StructField(CORRUPT, StringType(), True),
    ])


def item_details(raw):
    """One row per parent_asin with the extracted fields."""
    from pyspark.sql import Window, functions as F
    from pyspark.sql.types import ArrayType, StringType, StructField, StructType

    extracted = StructType([StructField("directors", ArrayType(StringType()), False),
                            StructField("starring", ArrayType(StringType()), False),
                            StructField("studio", StringType(), True), StructField("image", StringType(), True)])
    udf = F.udf(extract, extracted)
    rows = (raw.filter(F.col(CORRUPT).isNull() & F.col("parent_asin").isNotNull() & (F.trim("parent_asin") != ""))
            .withColumn("x", udf("details", "images"))
            .select("parent_asin", "main_category", "x.directors", "x.starring", "x.studio", "x.image",
                    "rating_number"))
    order = Window.partitionBy("parent_asin").orderBy(F.col("rating_number").desc_nulls_last(),
                                                      F.to_json(F.struct(*FIELDS)))
    return rows.withColumn("_n", F.row_number().over(order)).filter("_n = 1").drop("_n")


def run_item_details(config: ProjectConfig, run_directory: Path) -> Path:
    """Write `<run>/item_details/` from the run's raw metadata. Starts Spark."""
    run_directory = Path(run_directory).resolve(strict=True)
    manifest = json.loads((run_directory / "run.json").read_text())
    final, partial = run_directory / "item_details", run_directory / "item_details.partial"
    if final.exists():
        raise FileExistsError(f"Item details already written for this run: {final}")
    recorded = manifest["inputs"]["metadata"]
    started = time.monotonic()
    timings = {}
    with timed(timings, "input_hash"):
        identity = file_identity(Path(recorded["path"]))
    if (identity["bytes"], identity["sha256"]) != (recorded["bytes"], recorded["sha256"]):
        raise ValueError(f"Metadata input changed since init-run: {recorded['path']}")
    if partial.exists():
        shutil.rmtree(partial)  # an interrupted attempt; never marked complete

    from pyspark.sql import functions as F

    from amazon_recommender.spark.session import create_spark_session, describe_resources

    with timed(timings, "spark_start"):
        spark = create_spark_session(config, app_name="amazon-recommender-item-details")
    try:
        raw = spark.read.schema(raw_schema()).option("mode", "PERMISSIVE") \
            .option("columnNameOfCorruptRecord", CORRUPT).json(recorded["path"])
        with timed(timings, "extract_write"):
            item_details(raw).write.mode("error").parquet(str(partial / "items"))
        written = spark.read.parquet(str(partial / "items"))
        with timed(timings, "verification_counts"):
            counts = written.agg(
                F.count("*").alias("items"),
                *[F.sum((F.size(column) > 0).cast("long")).alias(f"with_{column}") for column in ("directors", "starring")],
                *[F.count(column).alias(f"with_{column}") for column in ("main_category", "studio", "image",
                                                                         "rating_number")],
            ).first().asDict()
            counts["rows_read"] = raw.count()
        timings["total"] = round(time.monotonic() - started, 3)
        stage = {
            "stage": "item-details",
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_id": manifest["run_id"],
            "input": identity,
            "contract": __doc__.split("Normalization contract")[1].split("Output:")[0].strip(),
            "counts": {key: int(value or 0) for key, value in counts.items()},
            "resources": describe_resources(spark, config),
            "timings_seconds": timings,
            "outputs": {"items": "items"},
        }
    finally:
        spark.stop()
    (partial / "stage.json").write_text(json.dumps(stage, indent=2, allow_nan=False) + "\n")
    partial.rename(final)
    return final
