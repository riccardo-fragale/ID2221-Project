"""User recommendation lookup (P8.1) against the publication named by the active-run pointer.

The pointer is read again on every lookup, and the user document and the item titles are read
from the collections that pointer names, so a lookup never mixes two publications. Only pymongo
is used here; nothing imports Spark.
"""

from dataclasses import dataclass, field

from amazon_recommender.publish import ACTIVE_COLLECTION, ACTIVE_ID

OK = "ok"
NO_ACTIVE_RUN = "no_active_run"
UNKNOWN_USER = "unknown_user"
EMPTY_ID = "empty_id"


@dataclass(frozen=True)
class LookupResult:
    status: str
    user_id: str
    publication: dict | None = None
    document: dict | None = None
    rows: list[dict] = field(default_factory=list)

    @property
    def short_list(self) -> bool:
        return self.document is not None and len(self.rows) < self.publication["k"]

    @property
    def missing_titles(self) -> int:
        return sum(row["title_missing"] for row in self.rows)


def active_publication(database) -> dict | None:
    return database[ACTIVE_COLLECTION].find_one({"_id": ACTIVE_ID})


def item_titles(collection, parent_asins: list[str]) -> dict:
    """One query for all items of a list; items without a document are absent from the result."""
    if not parent_asins:
        return {}
    return {item["_id"]: item.get("title") for item in collection.find({"_id": {"$in": list(parent_asins)}})}


def display_rows(items: list[dict], titles: dict) -> list[dict]:
    """Ranked rows; an item without a title is shown by its parent_asin and flagged."""
    rows = []
    for item in sorted(items, key=lambda item: item["rank"]):
        title = titles.get(item["parent_asin"])
        missing = not (isinstance(title, str) and title.strip())
        rows.append({"rank": item["rank"], "title": item["parent_asin"] if missing else title,
                     "parent_asin": item["parent_asin"], "score": item["score"], "title_missing": missing})
    return rows


def lookup_user(database, user_id: str) -> LookupResult:
    """Exact lookup by original user id. Raises pymongo errors; the caller reports outages."""
    user_id = (user_id or "").strip()
    if not user_id:
        return LookupResult(EMPTY_ID, user_id)
    publication = active_publication(database)
    if publication is None:
        return LookupResult(NO_ACTIVE_RUN, user_id)
    document = database[publication["recommendations"]].find_one({"_id": user_id})
    if document is None:
        return LookupResult(UNKNOWN_USER, user_id, publication)
    items = document.get("items") or []
    titles = item_titles(database[publication["items"]], [item["parent_asin"] for item in items])
    return LookupResult(OK, user_id, publication, document, display_rows(items, titles))
