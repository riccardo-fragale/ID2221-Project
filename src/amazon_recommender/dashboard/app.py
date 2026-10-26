"""Streamlit entry point: `amazon-recommender dashboard --config <profile>` (or
`streamlit run app.py -- --config <profile>`). Reads MongoDB ($MONGODB_URI) and the saved stage
records of the active publication; never starts Spark."""

import argparse
from pathlib import Path
import sys
import tomllib

import altair as alt
from pymongo import MongoClient
from pymongo.errors import PyMongoError
import streamlit as st

from amazon_recommender.config import load_config
from amazon_recommender.dashboard import tables
from amazon_recommender.dashboard.evidence import EvidenceError, load_evidence
from amazon_recommender.dashboard.lookup import EMPTY_ID, NO_ACTIVE_RUN, UNKNOWN_USER, active_publication, lookup_user
from amazon_recommender.publish import mongo_uri

TIMEOUT_MS = 2000


@st.cache_resource
def client(uri: str) -> MongoClient:
    """One client per URI for the whole server; results are never cached with it."""
    return MongoClient(uri, serverSelectionTimeoutMS=TIMEOUT_MS)


@st.cache_data(max_entries=4)
def evidence(source: str, version: str) -> dict:
    """Keyed on the publication version too, so a new publication never reuses old evidence."""
    return load_evidence(source)


def publication_caption(publication: dict) -> str:
    return (f"Active publication `{publication['version']}`: model {publication['model_label']}, fallback "
            f"{publication['fallback_label']}, top-{publication['k']}, run {publication['run_id']}, split "
            f"{publication['split']}, published {publication['published_at']}")


def lookup_tab(database) -> None:
    user_id = st.text_input("Original user ID (case-sensitive)")
    if not st.button("Look up") and not user_id:
        return
    try:
        result = lookup_user(database, user_id)
    except PyMongoError as exc:
        st.error(f"MongoDB is unavailable, no recommendations can be shown: {exc}")
        return
    if result.status == EMPTY_ID:
        st.info("Enter a user ID.")
        return
    if result.status == NO_ACTIVE_RUN:
        st.warning("No publication is active yet: run `amazon-recommender publish` first.")
        return
    st.caption(publication_caption(result.publication))
    if result.status == UNKNOWN_USER:
        st.warning(f"User `{result.user_id}` is not in the active publication (IDs are exact and case-sensitive).")
        return
    document = result.document
    st.write(f"Recommendation type: **{document['rec_type']}** · user type: **{document['user_type']}**"
             + (" · approximate retrieval" if document["approximate"] else ""))
    if not result.rows:
        st.warning(f"Empty list: {document.get('short_list_reason') or 'no reason recorded'}.")
        return
    if result.short_list:
        st.info(f"{len(result.rows)} of {result.publication['k']} items: "
                f"{document.get('short_list_reason') or 'no reason recorded'}.")
    if result.missing_titles:
        st.info(f"{result.missing_titles} of {len(result.rows)} items have no title in the category metadata: "
                f"{result.derived_labels} are described from their details (category, directors, cast), "
                f"the others are shown by parent_asin.")
    st.dataframe(tables.recommendation_table(result.rows), hide_index=True, use_container_width=True,
                 column_config={
                     "image": st.column_config.ImageColumn("image", help="Loaded from Amazon's image CDN"),
                     "link": st.column_config.LinkColumn("link", display_text="amazon.com",
                                                         help="amazon.com/dp/<parent_asin>; not verified"),
                 })


def definitions_section(definitions: dict) -> None:
    k = definitions.get("k")
    cohort = definitions.get("cohort") or {}
    bootstrap = definitions.get("bootstrap") or {}
    st.markdown(
        f"- **Relevant item:** a held-out rating >= {definitions.get('relevant_rating')}.\n"
        f"- **Cohort:** held-out users with at least one relevant item: {cohort.get('eligible_users')} of "
        f"{cohort.get('held_out_users')} held-out users ({cohort.get('users_without_relevant')} without a relevant "
        "item are excluded and counted).\n"
        f"- **Ranking:** top-{k} over the {definitions.get('catalog_items')} training items, excluding the user's "
        "training items.\n"
        f"- **P@{k}:** relevant items in the list divided by {k}, short lists included. **NDCG@{k}:** binary gains. "
        "Both are macro averages over the cohort.\n"
        f"- **95% CI:** user-level bootstrap, {bootstrap.get('resamples')} resamples, seed {bootstrap.get('seed')}.\n"
        f"- **Random reference:** {definitions.get('random_reference_note')}.\n"
        f"- **p (vs random):** {definitions.get('signal_test')}.\n"
        "- **RMSE:** on all held-out ratings, predictions clipped to the rating scale; raw RMSE unclipped.")


def quality_section(data: dict) -> None:
    views = data["views"]
    headline = views.get("test")
    st.subheader("Four-system comparison")
    st.caption(data["headline_label"] + ", standard view (headline)")
    if headline is None:
        st.warning("No standard test-view evaluation records were found for this publication.")
    else:
        st.dataframe(tables.quality_table(headline["rows"], data["selected"]), hide_index=True,
                     use_container_width=True)
        st.dataframe(tables.random_row(headline["random_reference"]), hide_index=True, use_container_width=True)
        for model, mark in sorted(data["selected"].items()):
            if mark["status"] != "selected on development":
                st.caption(f"{model}: {mark['status']}")
        with st.expander("Metric definitions (from the evaluation records)"):
            definitions_section(data["definitions"])
    for name, view in sorted(views.items()):
        if name == "test":
            continue
        with st.expander(f"Additional view: {name}" + (f" ({view['view']['label']})" if view["view"]["label"] else "")):
            st.dataframe(tables.quality_table(view["rows"], {}), hide_index=True, use_container_width=True)
            st.dataframe(tables.random_row(view["random_reference"]), hide_index=True, use_container_width=True)
            if view["user_types"]:
                st.dataframe(tables.user_type_table(view["user_types"]), hide_index=True, use_container_width=True)


def development_section(selections: list[dict]) -> None:
    st.subheader("Development selections (not headline results)")
    if not selections:
        st.info("No selection records in this run.")
    for selection in selections:
        with st.expander(f"{selection['model']}: chosen {selection['chosen']}"):
            st.caption(selection["label"])
            st.write(f"Rule: {selection['rule']} (written {selection['rule_written']}); {selection['users']} users.")
            st.dataframe(tables.selection_table(selection), hide_index=True, use_container_width=True)
            st.caption(selection["path"])


def retention_section(retention: dict) -> None:
    st.subheader("Data retention and serving coverage")
    st.caption(retention["label"])
    st.json({"sample": retention["sample"], "split": retention["split"]}, expanded=False)
    coverage = retention.get("serving_coverage") or {}
    st.write(f"Serving coverage: {coverage.get('users')} users, {coverage.get('rows')} recommendation rows.")
    st.dataframe([{"user type / recommendation type": key, "users": value}
                  for key, value in (coverage.get("users_by_type_and_rec_type") or {}).items()], hide_index=True)
    st.json({"short_lists": coverage.get("short_lists"), "checks": coverage.get("checks")}, expanded=False)


def scalability_section(benchmarks: list[dict]) -> None:
    st.subheader("Scalability (fixed resources)")
    if not benchmarks:
        st.info("No benchmark report in this run.")
    for benchmark in benchmarks:
        st.markdown(f"**{benchmark['label']}** ({benchmark['status']}): {benchmark['measurement_type']}; "
                    f"{benchmark['scaling']}; repetitions: {benchmark['repetitions']}.")
        spark, host = benchmark["spark"] or {}, benchmark["host"] or {}
        st.caption(f"Resources: master {spark.get('master')}, driver memory {spark.get('driver_memory')}, "
                   f"{spark.get('shuffle_partitions')} shuffle partitions, identical for every stage: "
                   f"{benchmark['spark_resources_identical']}; host {host.get('logical_cpus')} logical CPUs, "
                   f"{(host.get('mem_total_bytes') or 0) / 2**30:.1f} GiB; cache {(benchmark['cache'] or {}).get('state')}.")
        points = tables.scaling_points(benchmark)
        if points:
            chart = alt.Chart(alt.Data(values=points)).mark_line(point=True).encode(
                x=alt.X("fraction:Q", title="input fraction"),
                y=alt.Y("work seconds (median):Q", title="work seconds (median)"),
                color=alt.Color("step:N"), tooltip=["step:N", "fraction:Q", "work seconds (median):Q",
                                                    "input rows:Q", "label:N"])
            st.altair_chart(chart, use_container_width=True)
            st.dataframe(points, hide_index=True, use_container_width=True)
        unmeasured = tables.unmeasured_points(benchmark)
        if unmeasured:
            st.caption("Not measured: " + "; ".join(f"{p['step']} at {p['fraction']} ({p['statuses']})"
                                                    for p in unmeasured))
        for step, note in benchmark["notes"].items():
            st.caption(f"{step}: {note}")
        if benchmark["figure"]:
            with st.expander("Saved figure (scaling.svg)"):
                st.image(Path(benchmark["figure"]).read_text())
        st.caption(benchmark["path"])


def evidence_tab(database) -> None:
    try:
        publication = active_publication(database)
    except PyMongoError as exc:
        st.error(f"MongoDB is unavailable, so the active publication is unknown: {exc}")
        return
    if publication is None:
        st.warning("No publication is active yet.")
        return
    st.caption(publication_caption(publication))
    try:
        data = evidence(publication["source"], publication["version"])
    except (EvidenceError, OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"The saved records of this publication cannot be read: {exc}")
        return
    quality_section(data)
    development_section(data["selections"])
    retention_section(data["retention"])
    scalability_section(data["benchmarks"])
    for problem in data["problems"]:
        st.caption(f"Note: {problem}")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    st.set_page_config(page_title="Amazon recommender", layout="wide")
    st.title("Amazon Reviews 2023 recommender")
    try:
        config = load_config(args.config)
        database = client(mongo_uri(config))[config.mongo.database]
    except (OSError, ValueError, TypeError, tomllib.TOMLDecodeError) as exc:
        st.error(f"Cannot configure the dashboard: {exc}")
        return
    lookup, results = st.tabs(["Recommendations", "Experiment evidence"])
    with lookup:
        lookup_tab(database)
    with results:
        evidence_tab(database)


main(sys.argv[1:])
