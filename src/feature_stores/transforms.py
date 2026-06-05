import json
import logging
import os
import re
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Iterator, List, Optional, Tuple

from pyflink.datastream.functions import (
    FilterFunction,
    FlatMapFunction,
    KeyedProcessFunction,
    MapFunction,
    ReduceFunction,
)
from pyflink.common.watermark_strategy import TimestampAssigner
from pyflink.datastream.state import ListStateDescriptor
from pyflink.common.typeinfo import Types as FlinkTypes

from avro_deserializer import AvroDecoder
from config import FEAST_REPO_PATH

logger = logging.getLogger("PyFlinkCDCFeast")


class CDCParser(FlatMapFunction):
    def __init__(self):
        self._avro = AvroDecoder()

    def flat_map(self, raw: str) -> Iterator[Tuple]:
        if not raw:
            return
        try:
            raw = self._avro.decode(raw)
            if not raw:
                return
            envelope = json.loads(raw)
            msg   = envelope.get("payload", envelope)
            op    = msg.get("op")
            after = msg.get("after")
            src   = msg.get("source", {})

            if op not in ("c", "u") or not after:
                return

            table = src.get("table", "")
            if "interactions" in table:
                topic_type = "interaction"
                entity_id  = str(after.get("user_id", ""))
            elif "users" in table:
                topic_type = "user"
                entity_id  = str(after.get("user_id", ""))
            elif "items" in table:
                topic_type = "item"
                entity_id  = str(after.get("item_id", ""))
            else:
                return

            if not entity_id:
                return

            ts_ms = src.get("ts_ms") or int(datetime.utcnow().timestamp() * 1000)
            yield (topic_type, entity_id, json.dumps(after), ts_ms)

        except Exception as exc:
            logger.warning("CDC parse error: %s | %s", raw[:120], exc)


class InteractionFilter(FilterFunction):
    def filter(self, v: Tuple) -> bool:
        return v[0] == "interaction"

class UserFilter(FilterFunction):
    def filter(self, v: Tuple) -> bool:
        return v[0] == "user"

class ItemFilter(FilterFunction):
    def filter(self, v: Tuple) -> bool:
        return v[0] == "item"


def _year_from_title(title: Optional[str]) -> int:
    if not title:
        return 0
    m = re.search(r"\((\d{4})\)\s*$", str(title))
    if not m:
        return 0
    try:
        y = int(m.group(1))
        return y if 1800 <= y <= 2100 else 0
    except ValueError:
        return 0


class CDCTimestampAssigner(TimestampAssigner):
    def extract_timestamp(self, value: Tuple, record_timestamp: int) -> int:
        return value[3]


class UserRatingReducer(ReduceFunction):
    def reduce(self, value1: Tuple, value2: Tuple) -> Tuple:
        _, user_id, acc1_json, _ = value1
        _, _, acc2_json, ts2 = value2
        try:
            acc1 = json.loads(acc1_json)
            acc2 = json.loads(acc2_json)
            merged = {
                "rating_sum":   acc1.get("rating_sum", 0.0)  + acc2.get("rating_sum", 0.0),
                "rating_count": acc1.get("rating_count", 0)  + acc2.get("rating_count", 0),
                "rating_sum_sq": acc1.get("rating_sum_sq", 0.0) + acc2.get("rating_sum_sq", 0.0),
            }
        except Exception:
            merged = {"rating_sum": 0.0, "rating_count": 0, "rating_sum_sq": 0.0}
        return ("interaction", user_id, json.dumps(merged), ts2)


class ItemRatingReducer(ReduceFunction):
    def reduce(self, value1: Tuple, value2: Tuple) -> Tuple:
        _, item_id, acc1_json, _ = value1
        _, _, acc2_json, ts2 = value2
        try:
            acc1 = json.loads(acc1_json)
            acc2 = json.loads(acc2_json)
            merged = {
                "rating_sum":   acc1.get("rating_sum", 0.0)  + acc2.get("rating_sum", 0.0),
                "rating_count": acc1.get("rating_count", 0)  + acc2.get("rating_count", 0),
                "rating_sum_sq": acc1.get("rating_sum_sq", 0.0) + acc2.get("rating_sum_sq", 0.0),
            }
        except Exception:
            merged = {"rating_sum": 0.0, "rating_count": 0, "rating_sum_sq": 0.0}
        return ("interaction", item_id, json.dumps(merged), ts2)


class UserToFeature(MapFunction):
    def map(self, value: Tuple) -> str:
        _, user_id, acc_json, _ = value
        try:
            acc   = json.loads(acc_json)
            count = acc.get("rating_count", 0)
            total = acc.get("rating_sum", 0.0)
            sum_sq = acc.get("rating_sum_sq", 0.0)
            avg = round(total / count, 4) if count else 0.0
            std = round((sum_sq / count - avg * avg) ** 0.5, 4) if count > 1 else 0.0
            feature = {
                "user_id":           int(user_id),
                "avg_rating":        avg,
                "interaction_count": count,
                "rating_std":        std,
                "event_timestamp":   datetime.now(timezone.utc),
            }
        except Exception:
            feature = {
                "user_id": int(user_id),
                "avg_rating": 0.0,
                "interaction_count": 0,
                "rating_std": 0.0,
                "event_timestamp": datetime.now(timezone.utc),
            }
        return json.dumps(feature, default=str)


class ItemToFeature(MapFunction):
    def map(self, value: Tuple) -> str:
        _, item_id, acc_json, _ = value
        try:
            acc   = json.loads(acc_json)
            count = acc.get("rating_count", 0)
            total = acc.get("rating_sum", 0.0)
            sum_sq = acc.get("rating_sum_sq", 0.0)
            avg = round(total / count, 4) if count else 0.0
            std = round((sum_sq / count - avg * avg) ** 0.5, 4) if count > 1 else 0.0
            # Align names with Feast movie_stats_fv: popularity = # distinct ratings in aggregate
            feature = {
                "movie_id":        int(item_id),
                "year":            0,
                "popularity":      int(count),
                "avg_rating":      avg,
                "rating_std":      std,
                "event_timestamp": datetime.now(timezone.utc),
            }
        except Exception:
            feature = {
                "movie_id": int(item_id),
                "year": 0,
                "popularity": 0,
                "avg_rating": 0.0,
                "rating_std": 0.0,
                "event_timestamp": datetime.now(timezone.utc),
            }
        return json.dumps(feature, default=str)


def interaction_to_user_acc(value: Tuple) -> Tuple:
    _, user_id, payload_json, ts = value
    try:
        row = json.loads(payload_json)
        r   = float(row.get("rating", 0)) if row.get("rating") is not None else 0.0
    except Exception:
        r = 0.0
    return ("interaction", user_id, json.dumps({"rating_sum": r, "rating_count": 1, "rating_sum_sq": r * r}), ts)


def interaction_to_item_acc(value: Tuple) -> Tuple:
    _, _, payload_json, ts = value
    try:
        row     = json.loads(payload_json)
        item_id = str(row.get("item_id", ""))
        r       = float(row.get("rating", 0)) if row.get("rating") is not None else 0.0
    except Exception:
        item_id = ""
        r = 0.0
    return ("interaction", item_id, json.dumps({"rating_sum": r, "rating_count": 1, "rating_sum_sq": r * r}), ts)


def interaction_to_recent_acc(value: Tuple) -> Tuple:
    """Convert a CDC interaction event to a recent-interactions accumulator (1 item)."""
    _, user_id, payload_json, ts = value
    try:
        row     = json.loads(payload_json)
        item_id = int(row.get("item_id", 0))
        r       = float(row.get("rating", 0)) if row.get("rating") is not None else 0.0
    except Exception:
        item_id = 0
        r = 0.0
    acc = {"movie_ids": [item_id], "ratings": [r]}
    return ("interaction", user_id, json.dumps(acc), ts)


class RecentInteractionsReducer(KeyedProcessFunction):
    """Keep the 5 most recent movie interactions per user using managed ListState."""
    MAX_RECENT = 10000

    def __init__(self):
        self._state = None

    def open(self, runtime_context):
        descriptor = ListStateDescriptor(
            "recent_interactions",
            FlinkTypes.TUPLE([FlinkTypes.INT(), FlinkTypes.FLOAT()])
        )
        self._state = runtime_context.get_list_state(descriptor)

    def process_element(self, value: Tuple, ctx):
        _, user_id, acc_json, ts = value
        try:
            acc     = json.loads(acc_json)
            new_ids = acc.get("movie_ids", [])
            new_rat = acc.get("ratings", [])

            existing = list(self._state.get() or [])
            existing.extend(zip(new_ids, new_rat))

            # Deduplicate by movie_id, keep latest occurrence
            seen: dict = {}
            for mid, rat in existing:
                seen[mid] = rat
            deduped = list(seen.items())[-self.MAX_RECENT:]

            self._state.clear()
            self._state.add_all(deduped)

            ids     = [d[0] for d in deduped]
            ratings = [d[1] for d in deduped]
            merged  = {"movie_ids": ids, "ratings": ratings}
        except Exception:
            merged = {"movie_ids": [], "ratings": []}

        yield ("interaction", user_id, json.dumps(merged), ts)


class UserRecentToFeature(MapFunction):
    def map(self, value: Tuple) -> str:
        _, user_id, acc_json, _ = value
        try:
            acc = json.loads(acc_json)
            feature = {
                "user_id":           int(user_id),
                "recent_movie_ids":  acc.get("movie_ids", []),
                "recent_ratings":    acc.get("ratings",   []),
                "event_timestamp":   datetime.now(timezone.utc),
            }
        except Exception:
            feature = {
                "user_id":          int(user_id),
                "recent_movie_ids": [],
                "recent_ratings":   [],
                "event_timestamp":  datetime.now(timezone.utc),
            }
        return json.dumps(feature, default=str)


def _normalize_jsonb_list(raw) -> list:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        if s.startswith("["):
            try:
                return [str(x) for x in json.loads(s)]
            except json.JSONDecodeError:
                return [s]
        if "|" in s:
            return [x.strip() for x in s.split("|") if x.strip()]
        return [s]
    return [str(raw)]


class ItemCatalogFromCdc(MapFunction):
    """Build offline/online row for item catalog from Debezium ``items`` ``after`` payload."""

    def map(self, value: Tuple) -> str:
        _, _, payload_json, ts_ms = value
        try:
            row = json.loads(payload_json)
            title = row.get("title") or ""
            genres = _normalize_jsonb_list(row.get("genre"))
            tags_raw = row.get("tags")
            if isinstance(tags_raw, list):
                tags_list = [str(x) for x in tags_raw]
            elif isinstance(tags_raw, str):
                try:
                    tags_list = [str(x) for x in json.loads(tags_raw)] if tags_raw.strip().startswith("[") else [tags_raw]
                except json.JSONDecodeError:
                    tags_list = [tags_raw]
            else:
                tags_list = []
            evt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            feat = {
                "movie_id": int(row.get("item_id", 0)),
                "title": title,
                "genres": genres,
                "tags": tags_list,
                "release_year": _year_from_title(title),
                "event_timestamp": evt,
            }
        except Exception as exc:
            logger.warning("ItemCatalogFromCdc failed: %s", exc)
            feat = {
                "movie_id": 0,
                "title": "",
                "genres": [],
                "tags": [],
                "release_year": 0,
                "event_timestamp": datetime.now(timezone.utc),
            }
        return json.dumps(feat, default=str)


class UserCatalogFromCdc(MapFunction):
    """Build offline row for user catalog from Debezium ``users`` ``after`` payload."""

    def map(self, value: Tuple) -> str:
        _, _, payload_json, ts_ms = value
        try:
            row = json.loads(payload_json)
            meta = row.get("metadata")
            if meta is None:
                meta_json = "{}"
            elif isinstance(meta, (dict, list)):
                meta_json = json.dumps(meta)
            else:
                meta_json = str(meta)
            evt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
            feat = {
                "user_id": int(row.get("user_id", 0)),
                "metadata_json": meta_json,
                "event_timestamp": evt,
            }
        except Exception as exc:
            logger.warning("UserCatalogFromCdc failed: %s", exc)
            feat = {
                "user_id": 0,
                "metadata_json": "{}",
                "event_timestamp": datetime.now(timezone.utc),
            }
        return json.dumps(feat, default=str)


class S3BatchSink(MapFunction):
    """Buffer feature JSON strings; flush to S3 Parquet when buffer reaches flush_every.

    Layout (Hive-style, physical partitions only; FileSource path stays the parent prefix):
      ``{s3_prefix}/date=YYYY-MM-DD/part-{time_us}_{uuid}.parquet``

    Rows are grouped by **UTC calendar date** of ``event_timestamp`` before write so one flush
    can emit several files if the buffer spans midnight. Missing/unparseable timestamps use
    the current UTC date.

    **Migration:** Existing flat objects ``{prefix}/part-*.parquet`` (no ``date=``) are left
    as-is; Feast / DuckDB+ibis still read them as long as they stay under the same
    ``FileSource`` prefix. New data uses the ``date=`` subfolders only.
    """

    def __init__(self, s3_prefix: str, flush_every: Optional[int] = None):
        super().__init__()
        self.s3_prefix = s3_prefix.rstrip("/")
        raw = flush_every if flush_every is not None else os.environ.get("S3_PARQUET_FLUSH_EVERY", "")
        try:
            n = int(raw) if str(raw).strip() else 1000
        except ValueError:
            n = 1000
        self._flush_every = max(1, n)
        self._buf: list = []
        self._s3 = None

    def _get_s3(self):
        if self._s3 is None:
            import boto3
            self._s3 = boto3.client("s3")
        return self._s3

    def map(self, record: str) -> str:
        self._buf.append(record)
        if len(self._buf) >= self._flush_every:
            self._flush()
        return record

    @staticmethod
    def _partition_date_str(record: dict) -> str:
        """UTC ``YYYY-MM-DD`` for Hive-style ``date=`` prefix from ``event_timestamp``."""
        import pandas as pd

        raw = record.get("event_timestamp")
        if raw is None:
            return datetime.now(timezone.utc).date().isoformat()
        try:
            dt = pd.to_datetime(raw, utc=True)
            return dt.date().isoformat()
        except Exception:
            return datetime.now(timezone.utc).date().isoformat()

    def _group_records_by_partition_date(self, records: List[dict]) -> Dict[str, List[dict]]:
        groups: Dict[str, List[dict]] = defaultdict(list)
        for rec in records:
            groups[self._partition_date_str(rec)].append(rec)
        return groups

    def _flush(self):
        if not self._buf:
            return
        import io

        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        from urllib.parse import urlparse

        parsed = urlparse(self.s3_prefix)
        bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        base_ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        try:
            records = [json.loads(r) for r in self._buf]
            by_date = self._group_records_by_partition_date(records)
            for date_str, group in sorted(by_date.items()):
                df = pd.DataFrame(group)
                if "event_timestamp" in df.columns:
                    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True)
                table = pa.Table.from_pandas(df, preserve_index=False)
                buf = io.BytesIO()
                pq.write_table(table, buf)
                suffix = f"{base_ts}-{uuid.uuid4().hex[:12]}"
                key = f"{prefix}/date={date_str}/part-{suffix}.parquet"
                self._get_s3().put_object(Bucket=bucket, Key=key, Body=buf.getvalue())
                logger.info(
                    "S3BatchSink flushed %d records → s3://%s/%s (date=%s)",
                    len(group),
                    bucket,
                    key,
                    date_str,
                )
        except Exception as exc:
            logger.error("S3BatchSink flush failed: %s", exc)
        self._buf.clear()

    def close(self):
        self._flush()


class FeastPushFunction(MapFunction):
    def __init__(self, entity_type: str, push_source_name: str):
        super().__init__()
        self.entity_type       = entity_type
        self.push_source_name  = push_source_name
        self._store            = None
        self._store_init_attempted = False

    def _get_store(self):
        if not self._store_init_attempted:
            self._store_init_attempted = True
            try:
                from feast import FeatureStore
                self._store = FeatureStore(repo_path=FEAST_REPO_PATH)
                logger.info("FeatureStore ready for %s", self.entity_type)
            except Exception as exc:
                logger.error("FeatureStore init failed: %s", exc)
        return self._store

    def map(self, feature_json: str) -> str:
        store = self._get_store()
        if store is None:
            logger.error("FeatureStore not ready, skipping push.")
            return feature_json
        try:
            import pandas as pd
            record = json.loads(feature_json)
            record["event_timestamp"] = pd.Timestamp(record["event_timestamp"], tz="UTC")
            df = pd.DataFrame([record])
            store.push(push_source_name=self.push_source_name, df=df)
            logger.info("Pushed %s to Feast: %s", self.entity_type, record)
        except Exception as exc:
            logger.error("Feast push failed (%s): %s", self.entity_type, exc)
        return feature_json
