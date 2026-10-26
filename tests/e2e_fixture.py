"""Seeded synthetic category for the end-to-end test (P9.2), written as raw JSONL in the same
fields and types as tests/fixtures/*.ndjson. Three of the four untitled items also carry raw
metadata fields shaped like the real category's (main_category, rating_number, details in both
families, images); the fourth has none, so the dashboard falls back to its parent_asin.

Interactions are laid out in time phases, so the global 80/10/10 split and its refit
(train + validation up to the validation cutoff) leave every user type of the serving view:
- `core` users rate 8 distinct items early, then 3 more later. After the k=5 core they are
  known users, and the later ratings feed the validation and test windows;
- `sparse` users rate 2 items early, so the k-core removes them (`removed_by_k_core`);
- `late` users appear only at the very end (`new`), after the refit's train cutoff.
Every item is rated early by at least 5 core users, so the k-core keeps the whole catalog, and
each known user has at least ITEMS - 11 unseen items, more than k=20. Some items have a null title.
"""

import json
from pathlib import Path
import random

ITEMS = 40
CORE_USERS = 50
SPARSE_USERS = 10
LATE_USERS = 10
UNTITLED_ITEMS = 4
EARLY_RATINGS, LATER_RATINGS = 8, 3
START_MS = 1_420_070_400_000  # 2015-01-01
MINUTE_MS = 60_000


# Raw fields of the first three (untitled) items, and what the dashboard should show for each untitled item.
DETAILS = [
    {"main_category": "Prime Video", "rating_number": 120,
     "details": {"Directors": ["Ada Park"], "Starring": ["Robert Downey, Jr.", "Bea Cole", "Cy Dunn", "Di Ek"],
                 "Studio": ["North"]},
     "images": [{"variant": "PT01", "360w": "https://img.example/side.jpg"},
                {"variant": "MAIN", "360w": "https://img.example/e2e-00.jpg", "480w": "https://img.example/x.jpg"}]},
    {"main_category": "Prime Video", "rating_number": 7, "details": {"Starring": ["Eve Fox"]},
     "images": [{"variant": "MAIN", "480w": "https://img.example/e2e-01.jpg"}]},
    {"main_category": "Movies & TV", "rating_number": 3,
     "details": {"Director": "Gil Ho", "Actors": "Ian Jo, Jr., Kim Lu", "Studio": "South"},
     "images": [{"variant": "MAIN", "thumb": "https://img.example/t.jpg", "large": "https://img.example/e2e-02.jpg",
                 "hi_res": "https://img.example/h.jpg"}]},
]
LABELS = ["Prime Video · dir. Ada Park · Robert Downey, Jr., Bea Cole, Cy Dunn", "Prime Video · Eve Fox",
          "Movies & TV · dir. Gil Ho · Ian Jo, Jr., Kim Lu", None]
IMAGES = ["https://img.example/e2e-00.jpg", "https://img.example/e2e-01.jpg", "https://img.example/e2e-02.jpg", None]


def item(n: int) -> str:
    return f"e2e-item-{n:02d}"


def generate(directory: Path, seed: int = 2221) -> dict:
    """Write reviews.jsonl and metadata.jsonl; return their paths and the user ids by intended type."""
    rng = random.Random(seed)
    catalog = [item(n) for n in range(ITEMS)]
    core = [f"e2e-core-{n:02d}" for n in range(CORE_USERS)]
    sparse = [f"e2e-sparse-{n:02d}" for n in range(SPARSE_USERS)]
    late = [f"e2e-late-{n:02d}" for n in range(LATE_USERS)]
    # Core user i rates the block of items 8i..8i+7 (mod ITEMS), so each item gets exactly 10 early ratings.
    early = {user: {catalog[(index * EARLY_RATINGS + offset) % ITEMS] for offset in range(EARLY_RATINGS)}
             for index, user in enumerate(core)}
    phases = [
        [(user, asin) for user in core for asin in sorted(early[user])] + [
            (user, asin) for user in sparse for asin in rng.sample(catalog, 2)],
        *[[(user, rng.choice([asin for asin in catalog if asin not in early[user]])) for user in core]
          for _ in range(LATER_RATINGS)],
        [(user, asin) for user in late for asin in rng.sample(catalog, 2)],
    ]
    rows, clock = [], START_MS
    for phase in phases:
        rng.shuffle(phase)
        for user, asin in phase:
            clock += rng.randint(1, 600) * MINUTE_MS  # strictly increasing: no timestamp ties
            rating = float(rng.choices([5, 4, 3, 2, 1], weights=[45, 30, 12, 8, 5])[0])
            rows.append({"user_id": user, "parent_asin": asin, "rating": rating, "timestamp": clock})
    # Later phases may repeat an item for a user; ETL keeps the latest (user, item) pair.
    untitled = set(catalog[:UNTITLED_ITEMS])
    metadata = [{"parent_asin": asin, "title": None if asin in untitled else f"Synthetic Movie {asin[-2:]}"}
                for asin in catalog]
    for row, extra in zip(metadata, DETAILS):
        row.update(extra)
    directory.mkdir(parents=True, exist_ok=True)
    reviews, items = directory / "reviews.jsonl", directory / "metadata.jsonl"
    reviews.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
    items.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in metadata))
    return {"reviews": reviews, "metadata": items, "users": {"core": core, "sparse": sparse, "late": late},
            "untitled": sorted(untitled), "labels": dict(zip(catalog, LABELS)),
            "images": dict(zip(catalog, IMAGES))}
