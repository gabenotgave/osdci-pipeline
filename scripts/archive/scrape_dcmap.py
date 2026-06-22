#!/usr/bin/env python3
"""Scrape datacentermap.com US data centers, geocode, and assign H3 res-6 indices."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import h3
import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
CHECKPOINT_PATH = RAW_DIR / "dcmap_checkpoint.jsonl"
OUTPUT_PATH = RAW_DIR / "datacenters_dcmap.parquet"
FAILURES_PATH = RAW_DIR / "dcmap_geocode_failures.csv"

BASE_URL = "https://www.datacentermap.com"
USA_INDEX_URL = f"{BASE_URL}/usa/"
H3_RESOLUTION = 6

# Site returns 429 for research-bot; browser UA is used after the first 429.
RESEARCH_BOT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; research-bot/1.0)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

US_STATE_SLUGS = [
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
    "connecticut", "delaware", "district-of-columbia", "florida", "georgia",
    "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky",
    "louisiana", "maine", "maryland", "massachusetts", "michigan", "minnesota",
    "mississippi", "missouri", "montana", "nebraska", "nevada", "new-hampshire",
    "new-jersey", "new-mexico", "new-york", "north-carolina", "north-dakota",
    "ohio", "oklahoma", "oregon", "pennsylvania", "rhode-island", "south-carolina",
    "south-dakota", "tennessee", "texas", "utah", "vermont", "virginia",
    "washington", "west-virginia", "wisconsin", "wyoming",
]

NOMINATIM_HEADERS = {"User-Agent": "osdci-research/1.0"}
CENSUS_GEOCODE_URL = "https://geocoding.geo.census.gov/geocoder/locations/address"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

REQUEST_SLEEP_SEC = 2
NOMINATIM_SLEEP_SEC = 1
PROGRESS_EVERY = 50
MAX_RETRIES = 3
MARKET_RETRIES = 4
VALID_MAPTYPES = frozenset({"dc", "dc-unmapped"})

OUTPUT_COLUMNS = [
    "name",
    "address",
    "city",
    "state",
    "zip_code",
    "lat",
    "lon",
    "h3_index",
    "geocode_source",
    "source",
]


def log(message: str) -> None:
    print(message, flush=True)


def _geo_to_h3(lat: float, lon: float, resolution: int) -> str:
    if hasattr(h3, "geo_to_h3"):
        return h3.geo_to_h3(lat, lon, resolution)
    return h3.latlng_to_cell(lat, lon, resolution)


def _normalize_slug(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _format_bytes(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{size} B"
        size /= 1024
    return f"{size:.1f} GB"


class Scraper:
    """Scrape datacentermap.com using market-page bulk JSON (mapdata.dcs)."""

    def __init__(self, use_research_bot: bool = False, facilities_only: bool = False) -> None:
        self.session = requests.Session()
        self.using_browser_ua = not use_research_bot
        self.headers = dict(RESEARCH_BOT_HEADERS if use_research_bot else BROWSER_HEADERS)
        self.facilities_only = facilities_only
        self.dc_count = 0

    def _sleep(self) -> None:
        time.sleep(REQUEST_SLEEP_SEC)

    def _switch_to_browser_ua(self) -> None:
        if not self.using_browser_ua:
            log("  research-bot User-Agent blocked (429); switching to browser User-Agent")
            self.headers = dict(BROWSER_HEADERS)
            self.using_browser_ua = True

    def fetch(self, url: str) -> str | None:
        """Fetch a URL with rate limiting and minimal retries."""
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self.session.get(url, headers=self.headers, timeout=60)
                if response.status_code == 429:
                    self._switch_to_browser_ua()
                    if attempt < MAX_RETRIES:
                        wait = 5 * (attempt + 1)
                        log(f"  Rate limited (429) for {url}; retrying in {wait}s...")
                        time.sleep(wait)
                        continue
                    log(f"  Giving up on {url} after repeated 429 responses")
                    return None
                response.raise_for_status()
                self._sleep()
                return response.text
            except requests.RequestException as exc:
                log(f"  Request failed for {url}: {exc}")
                if attempt < MAX_RETRIES:
                    time.sleep(5)
        return None

    @staticmethod
    def parse_next_data(html: str) -> dict[str, Any] | None:
        soup = BeautifulSoup(html, "html.parser")
        script = soup.find("script", id="__NEXT_DATA__")
        if not script or not script.string:
            return None
        try:
            payload = json.loads(script.string)
            return payload.get("props", {}).get("pageProps", {})
        except json.JSONDecodeError:
            return None

    @staticmethod
    def extract_hrefs(html: str, pattern: re.Pattern[str]) -> list[str]:
        return sorted(set(pattern.findall(html)))

    @staticmethod
    def expected_facility_count(page_props: dict[str, Any]) -> int | None:
        stats = page_props.get("geodata", {}).get("meta_stats", {})
        facilities = stats.get("dcs_facilities")
        if isinstance(facilities, dict) and "total" in facilities:
            return int(facilities["total"])
        return None

    @staticmethod
    def market_paths_from_state(page_props: dict[str, Any], state_slug: str) -> list[str]:
        """Build market URLs from state page JSON (primary) and HTML is not needed here."""
        paths: list[str] = []
        for geo in page_props.get("mapdata", {}).get("geos") or []:
            link = (geo.get("properties") or {}).get("link")
            if link:
                paths.append(f"/usa/{state_slug}/{link}/")
        return sorted(set(paths))

    @staticmethod
    def records_from_mapdata(page_props: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract facilities from mapdata.dcs on a market page."""
        features = page_props.get("mapdata", {}).get("dcs") or []
        records: list[dict[str, Any]] = []
        for feature in features:
            props = feature.get("properties") or {}
            if props.get("maptype") not in VALID_MAPTYPES:
                continue
            dc_id = props.get("id")
            if dc_id is None:
                continue
            geometry = feature.get("geometry") or {}
            coordinates = geometry.get("coordinates") or [None, None]
            lon, lat = coordinates[0], coordinates[1]
            records.append(
                {
                    "name": props.get("name"),
                    "address": props.get("address"),
                    "city": props.get("city"),
                    "state": props.get("state"),
                    "zip_code": props.get("postal"),
                    "lat": float(lat) if lat is not None else None,
                    "lon": float(lon) if lon is not None else None,
                    "source_url": urljoin(BASE_URL, props.get("url", "")),
                    "dc_id": dc_id,
                    "listingtype": props.get("listingtype"),
                }
            )
        return records

    @staticmethod
    def dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        by_id: dict[int, dict[str, Any]] = {}
        for record in records:
            dc_id = record.get("dc_id")
            if dc_id is not None:
                by_id[dc_id] = record
        return list(by_id.values())

    def _track_scraped(self, count: int) -> None:
        previous = self.dc_count
        self.dc_count += count
        if self.dc_count // PROGRESS_EVERY > previous // PROGRESS_EVERY:
            log(f"  Scraped {self.dc_count:,} data centers...")

    def scrape_market_page(
        self, market_path: str, market_label: str = "", expected: int | None = None
    ) -> list[dict[str, Any]]:
        page_url = urljoin(BASE_URL, market_path)
        label = market_label or market_path

        for attempt in range(MARKET_RETRIES):
            html = self.fetch(page_url)
            if not html:
                if attempt < MARKET_RETRIES - 1:
                    log(f"  {label}: fetch failed, retrying...")
                    time.sleep(10 * (attempt + 1))
                    continue
                log(f"  {label}: fetch failed, skipping")
                return []

            page_props = self.parse_next_data(html)
            if not page_props:
                log(f"  No __NEXT_DATA__ on {page_url}")
                return []

            records = self.records_from_mapdata(page_props)
            if records:
                self._track_scraped(len(records))
                return records

            # Fallback: single facility page (no market listing JSON)
            dc = page_props.get("dc")
            if dc:
                record = {
                    "name": dc.get("name"),
                    "address": dc.get("address"),
                    "city": dc.get("city"),
                    "state": dc.get("state"),
                    "zip_code": dc.get("postal"),
                    "lat": float(dc["latitude"]) if dc.get("latitude") is not None else None,
                    "lon": float(dc["longitude"]) if dc.get("longitude") is not None else None,
                    "source_url": page_url,
                    "dc_id": dc.get("id"),
                    "listingtype": dc.get("listingtype"),
                }
                self._track_scraped(1)
                return [record]

            if expected and attempt < MARKET_RETRIES - 1:
                wait = 10 * (attempt + 1)
                log(
                    f"  {label}: expected ~{expected} facilities but got 0; "
                    f"retrying in {wait}s..."
                )
                time.sleep(wait)
                continue
            break

        if expected:
            log(f"  {label}: expected ~{expected} facilities but got 0")
        return []

    def scrape_state(self, state_slug: str) -> list[dict[str, Any]]:
        log(f"Scraping state: {state_slug}")
        records: list[dict[str, Any]] = []
        state_path = f"/usa/{state_slug}/"
        state_url = urljoin(BASE_URL, state_path)
        html = self.fetch(state_url)
        if not html:
            log(f"  Skipping {state_slug}: could not load state page")
            return records

        page_props = self.parse_next_data(html)
        if not page_props:
            log(f"  Skipping {state_slug}: no __NEXT_DATA__ on state page")
            return records

        expected_total = self.expected_facility_count(page_props)
        market_paths = self.market_paths_from_state(page_props, state_slug)
        if not market_paths:
            child_pattern = re.compile(rf'href="(/usa/{re.escape(state_slug)}/[^"/]+/)"')
            market_paths = [
                path
                for path in self.extract_hrefs(html, child_pattern)
                if "/quote/" not in path.lower()
            ]

        market_expected: dict[str, int] = {}
        for geo in page_props.get("mapdata", {}).get("geos") or []:
            props = geo.get("properties") or {}
            link = props.get("link")
            count = props.get("datacenters")
            if link and count is not None:
                market_expected[f"/usa/{state_slug}/{link}/"] = int(count)

        log(f"  {len(market_paths)} markets" + (
            f" (site reports {expected_total:,} facilities)" if expected_total else ""
        ))

        for market_path in market_paths:
            market_name = market_path.strip("/").split("/")[-1]
            expected = market_expected.get(market_path)
            batch = self.scrape_market_page(
                market_path, market_label=market_name, expected=expected
            )
            records.extend(batch)
            if expected and len(batch) < expected // 2:
                log(
                    f"  Warning: {market_name} returned {len(batch)} records "
                    f"(site map shows {expected})"
                )

        records = self.dedupe_records(records)
        if self.facilities_only:
            records = [r for r in records if r.get("listingtype") == "Facility"]
        if expected_total and len(records) < expected_total * 0.9:
            log(
                f"  Warning: scraped {len(records):,} unique facilities but "
                f"datacentermap.com reports {expected_total:,} for {state_slug}"
            )
        log(f"  {state_slug}: {len(records):,} unique data centers")
        return records

    def get_state_slugs(self) -> list[str]:
        html = self.fetch(USA_INDEX_URL)
        if html:
            pattern = re.compile(r'href="(/usa/([a-z0-9-]+)/)"')
            slugs = sorted(
                {
                    slug
                    for path, slug in pattern.findall(html)
                    if slug not in {"quote", "usa"}
                }
            )
            if slugs:
                return slugs
            log("  Warning: no state links parsed from /usa/; using built-in list")

        log(f"  Warning: could not load {USA_INDEX_URL}; using built-in state list")
        return list(US_STATE_SLUGS)


def load_checkpoint_states() -> set[str]:
    if not CHECKPOINT_PATH.exists():
        return set()
    states: set[str] = set()
    with CHECKPOINT_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            state = record.get("_state")
            if state:
                states.add(state)
    return states


def append_state_checkpoint(state_slug: str, records: list[dict[str, Any]]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("a", encoding="utf-8") as handle:
        for record in records:
            row = {**record, "_state": state_slug}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_all_scraped_records() -> list[dict[str, Any]]:
    if not CHECKPOINT_PATH.exists():
        return []
    seen: set[int | str] = set()
    records: list[dict[str, Any]] = []
    with CHECKPOINT_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            record.pop("_state", None)
            key = record.get("dc_id") or record.get("source_url")
            if key in seen:
                continue
            seen.add(key)
            records.append(record)
    return records


def census_geocode(
    street: str | None,
    city: str | None,
    state: str | None,
    zip_code: str | None,
) -> tuple[float, float] | None:
    params: dict[str, str] = {
        "benchmark": "Public_AR_Current",
        "format": "json",
    }
    if street:
        params["street"] = street
    if city:
        params["city"] = city
    if state:
        params["state"] = state
    if zip_code:
        params["zip"] = zip_code

    if not any(params.get(k) for k in ("street", "city", "state")):
        return None

    try:
        response = requests.post(CENSUS_GEOCODE_URL, params=params, timeout=30)
        response.raise_for_status()
        matches = response.json().get("result", {}).get("addressMatches", [])
        if matches:
            coords = matches[0]["coordinates"]
            return float(coords["y"]), float(coords["x"])
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        log(f"  Census geocode error: {exc}")
    return None


def nominatim_geocode(
    street: str | None,
    city: str | None,
    state: str | None,
) -> tuple[float, float] | None:
    parts = [part for part in (street, city, state, "USA") if part]
    if len(parts) < 2:
        return None

    time.sleep(NOMINATIM_SLEEP_SEC)
    params = {"q": ", ".join(parts), "format": "json", "limit": 1}
    try:
        response = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=NOMINATIM_HEADERS,
            timeout=30,
        )
        response.raise_for_status()
        results = response.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        log(f"  Nominatim geocode error: {exc}")
    return None


def geocode_record(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("lat") is not None and record.get("lon") is not None:
        record["geocode_source"] = None
        return record

    street = record.get("address") or None
    city = record.get("city") or None
    state = record.get("state") or None
    zip_code = record.get("zip_code") or None

    coords = census_geocode(street, city, state, zip_code)
    if coords:
        record["lat"], record["lon"] = coords
        record["geocode_source"] = "census"
        return record

    coords = census_geocode(street, city, state, None)
    if coords:
        record["lat"], record["lon"] = coords
        record["geocode_source"] = "census"
        return record

    coords = nominatim_geocode(street, city, state)
    if coords:
        record["lat"], record["lon"] = coords
        record["geocode_source"] = "nominatim"
        return record

    record["lat"] = None
    record["lon"] = None
    record["geocode_source"] = None
    return record


def geocode_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    need_geocode = sum(1 for r in records if r.get("lat") is None or r.get("lon") is None)
    log(f"\nGeocoding {need_geocode:,} records without coordinates ({len(records):,} total)...")
    geocoded: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    geocoded_n = 0

    for index, record in enumerate(records, start=1):
        result = geocode_record(record.copy())
        geocoded.append(result)
        if result["lat"] is None or result["lon"] is None:
            failures.append(result)
        elif result.get("geocode_source"):
            geocoded_n += 1
        if index % PROGRESS_EVERY == 0:
            log(f"  Processed {index:,}/{len(records):,}...")

    return geocoded, failures


def assign_h3_indices(records: list[dict[str, Any]]) -> None:
    for record in records:
        lat, lon = record.get("lat"), record.get("lon")
        if lat is not None and lon is not None:
            record["h3_index"] = _geo_to_h3(lat, lon, H3_RESOLUTION)
        else:
            record["h3_index"] = None


def build_output_dataframe(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        rows.append(
            {
                "name": record.get("name"),
                "address": record.get("address"),
                "city": record.get("city"),
                "state": record.get("state"),
                "zip_code": record.get("zip_code"),
                "lat": record.get("lat"),
                "lon": record.get("lon"),
                "h3_index": record.get("h3_index"),
                "geocode_source": record.get("geocode_source"),
                "source": "datacentermap.com",
            }
        )
    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    return df.drop_duplicates(subset=["name", "address", "city", "state"], keep="first")


def print_summary(df: pd.DataFrame, failures: list[dict[str, Any]]) -> None:
    total = len(df)
    geocoded = df["lat"].notna().sum()
    pct = (geocoded / total * 100) if total else 0.0
    census = (df["geocode_source"] == "census").sum()
    nominatim = (df["geocode_source"] == "nominatim").sum()
    from_site = int((df["lat"].notna() & df["geocode_source"].isna()).sum())

    log("\n" + "=" * 60)
    log("Scrape complete")
    log("=" * 60)
    log(f"Total data centers scraped: {total:,}")
    log(f"Successfully geocoded:      {geocoded:,} ({pct:.1f}%)")
    log(f"  Coordinates from site:    {from_site:,}")
    log(f"  Census geocoder:          {census:,}")
    log(f"  Nominatim fallback:       {nominatim:,}")
    log(f"Failed geocoding:           {len(failures):,}")
    if OUTPUT_PATH.exists():
        log(f"Output: {OUTPUT_PATH} ({_format_bytes(OUTPUT_PATH.stat().st_size)})")
    if failures:
        log(f"Failures log: {FAILURES_PATH}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape US data centers from datacentermap.com"
    )
    parser.add_argument(
        "--state",
        type=str,
        default=None,
        help="Scrape a single state slug only (e.g. virginia, delaware)",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip scraping; geocode and build output from checkpoint only",
    )
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Delete checkpoint before scraping",
    )
    parser.add_argument(
        "--use-research-bot",
        action="store_true",
        help="Use research-bot User-Agent (often blocked with 429; browser UA is default)",
    )
    parser.add_argument(
        "--facilities-only",
        action="store_true",
        help="Keep only listingtype=Facility rows (~matches site facility counts)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if args.reset_checkpoint and CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        log(f"Removed checkpoint: {CHECKPOINT_PATH}")

    if not args.skip_scrape:
        scraper = Scraper(
            use_research_bot=args.use_research_bot,
            facilities_only=args.facilities_only,
        )

        if args.state:
            target = _normalize_slug(args.state)
            if target in US_STATE_SLUGS:
                state_slugs = [target]
            else:
                state_slugs = scraper.get_state_slugs()
                state_slugs = [slug for slug in state_slugs if slug == target]
                if not state_slugs:
                    log(f"Error: state '{args.state}' not found on datacentermap.com/usa/")
                    return 1
        else:
            state_slugs = scraper.get_state_slugs()

        completed_states = load_checkpoint_states()
        if completed_states:
            log(f"Resuming: {len(completed_states)} state(s) already in checkpoint")

        for state_slug in state_slugs:
            if state_slug in completed_states:
                log(f"Skipping {state_slug} (already checkpointed)")
                continue
            records = scraper.scrape_state(state_slug)
            append_state_checkpoint(state_slug, records)

    records = load_all_scraped_records()
    if not records:
        log("No scraped records found. Run without --skip-scrape first.")
        return 1

    geocoded, failures = geocode_records(records)
    assign_h3_indices(geocoded)

    df = build_output_dataframe(geocoded)
    try:
        df.to_parquet(OUTPUT_PATH, index=False)
    except OSError as exc:
        log(f"Error: failed to write {OUTPUT_PATH}: {exc}")
        return 1

    if failures:
        failure_df = pd.DataFrame(failures)
        try:
            failure_df.to_csv(FAILURES_PATH, index=False)
        except OSError as exc:
            log(f"Warning: failed to write failures CSV: {exc}")

    print_summary(df, failures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
