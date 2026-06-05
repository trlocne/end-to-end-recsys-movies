import os
import time
import requests
import psycopg2.errors
import torch
import numpy as np
import pandas as pd
import logging
from typing import Any, Callable, Dict, List, Optional

from src.inference.v1.core.retrieval import (
    recommend_home,
    recommend_similar_items,
    recommend_search,
    mmr_rerank,
)
from src.inference.v1.core.cache import build_cache
from src.inference.v1.monitoring.metrics import (
    REC_REQUESTS, REC_LATENCY, CACHE_HIT_TOTAL, CACHE_MISS_TOTAL,
    FEAST_LATENCY, FEAST_ERRORS, MODEL_INFERENCE_LATENCY,
    ACTIVE_USERS, ACTIVE_ITEMS
)
from src.models.reranker import DeepFMFeatureExtractor
from src.inference.v1.core.triton_client import TritonRerankerClient
from src.inference.v1.core.db import (
    fetch_recent_interactions,
    insert_interaction,
    fetch_items_metadata,
)

logger = logging.getLogger(__name__)

class RecommendationService:
    def __init__(
        self,
        user_emb: np.ndarray,
        item_emb: np.ndarray,
        reranker: Optional[torch.nn.Module] = None,
        feature_extractor: Optional[Any] = None,
        milvus_collection: Optional[Any] = None,
        es_client: Optional[Any] = None,
        top_k_candidates: int = 100,
        top_k_final: int = 10,
        diversity_lambda: float = 0.7,
        es_index_name: str = "movies",
        feast_fs: Optional[Any] = None,
        metadata_df: Optional[pd.DataFrame] = None,
        use_mmr: bool = True,
        triton_client: Optional[TritonRerankerClient] = None,
        embedding_source: str = "parquet",
        embed_dim: int = 64,
    ):
        self.embedding_source = embedding_source
        self.embed_dim = embed_dim
        self._feast_only = self.embedding_source.strip().lower() == "feast_only"
        self.user_emb, self.item_emb = user_emb, item_emb
        self.reranker, self.feature_extractor = reranker, feature_extractor
        self.triton_client = triton_client
        self.milvus_collection, self.es_client = milvus_collection, es_client
        self.top_k_candidates, self.top_k_final = top_k_candidates, top_k_final
        self.diversity_lambda, self.es_index_name = diversity_lambda, es_index_name
        self.fs = feast_fs # Feast FeatureStore SDK
        self.metadata_df = metadata_df
        self.use_mmr = use_mmr
        self.cache = build_cache()
        self.session_history: Dict[int, Dict[str, List]] = {}

        if len(self.user_emb) > 0:
            sample_size = min(3000, len(self.user_emb))
            idx = np.random.choice(len(self.user_emb), size=sample_size, replace=False)
            self.mean_user_emb = np.mean(self.user_emb[idx], axis=0)
        else:
            self.mean_user_emb = np.zeros(self.embed_dim, dtype=np.float32)

        # Set Gauge values
        ACTIVE_USERS.set(len(self.user_emb))
        ACTIVE_ITEMS.set(len(self.item_emb))

    @staticmethod
    def _np_emb(raw: Any, dim: int) -> np.ndarray:
        if raw is None:
            return np.zeros(dim, dtype=np.float32)
        a = np.asarray(raw, dtype=np.float32).ravel()
        out = np.zeros(dim, dtype=np.float32)
        n = min(a.size, dim)
        if n > 0:
            out[:n] = a[:n]
        return out

    def _get_feast_user_embedding_vector(self, user_id: int) -> np.ndarray:
        if not self.fs:
            return np.zeros(self.embed_dim, dtype=np.float32)
        try:
            r = self.fs.get_online_features(
                features=["user_embeddings:embedding"],
                entity_rows=[{"user_id": int(user_id)}],
                full_feature_names=True,
            ).to_dict()
            return self._np_emb(
                (r.get("user_embeddings__embedding") or [None])[0],
                self.embed_dim,
            )
        except Exception as e:
            FEAST_ERRORS.labels(operation="get_user_embedding").inc()
            logger.warning("Feast user embedding lookup failed: %s", e)
            return np.zeros(self.embed_dim, dtype=np.float32)

    def _get_feast_movie_embeddings_batch(self, movie_ids: List[int]) -> Dict[int, np.ndarray]:
        if not self.fs or not movie_ids:
            return {}
        uniq: List[int] = []
        seen = set()
        for mid in movie_ids:
            if mid is None:
                continue
            m = int(mid)
            if m not in seen:
                seen.add(m)
                uniq.append(m)
        try:
            entity_rows = [{"movie_id": mid} for mid in uniq]
            batch = self.fs.get_online_features(
                features=["movie_embeddings:embedding"],
                entity_rows=entity_rows,
                full_feature_names=True,
            ).to_dict()
            vecs = batch.get("movie_embeddings__embedding", [])
            out: Dict[int, np.ndarray] = {}
            for i, mid in enumerate(uniq):
                if i < len(vecs) and vecs[i] is not None:
                    out[mid] = self._np_emb(vecs[i], self.embed_dim)
            return out
        except Exception as e:
            FEAST_ERRORS.labels(operation="get_movie_embeddings_batch").inc()
            logger.warning("Feast movie embedding batch failed: %s", e)
            return {}

    def _get_user_emb(self, user_id: int) -> np.ndarray:
        if self._feast_only:
            return self._get_feast_user_embedding_vector(user_id)
        if user_id < len(self.user_emb):
            return self.user_emb[user_id]
        return self.mean_user_emb

    def _get_hybrid_user_emb(self, user_id: int) -> np.ndarray:
        """Blend static profile with recent interaction history (70/30)."""
        base_emb = self._get_user_emb(user_id)
        recent_ids, _ = self._get_user_recent_interactions(user_id)
        if not recent_ids:
            return base_emb
        if self._feast_only:
            feast_map = self._get_feast_movie_embeddings_batch(
                [int(r) for r in recent_ids if r is not None]
            )
            vectors = [
                feast_map[int(r)]
                for r in recent_ids
                if r is not None and int(r) in feast_map
            ]
            if not vectors:
                return base_emb
            recent_mean_emb = np.mean(np.stack(vectors), axis=0)
        else:
            valid_ids = [rid for rid in recent_ids if rid < len(self.item_emb)]
            if not valid_ids:
                return base_emb
            recent_mean_emb = np.mean(self.item_emb[valid_ids], axis=0)
        return 0.7 * base_emb + 0.3 * recent_mean_emb

    def _get_user_recent_interactions(
        self, user_id: int, *, prefer_session: bool = True
    ) -> tuple[List[int], List[float]]:
        # Same-pod clicks before background Postgres write: merge/rerank still use RAM.
        if prefer_session and user_id in self.session_history:
            h = self.session_history[user_id]
            return h["ids"], h["ratings"]

        try:
            ids, ratings = fetch_recent_interactions(user_id, limit=5)
            return ids, ratings
        except Exception as e:
            logger.warning("Recent interactions Postgres fetch failed: %s", e)

        if not self.fs:
            return [], []

        try:
            res = self.fs.get_online_features(
                features=["user_recent_interactions:recent_movie_ids", "user_recent_interactions:recent_ratings"],
                entity_rows=[{"user_id": user_id}]
            ).to_dict()

            ids = res.get("recent_movie_ids", [[]])[0] or []
            ratings = res.get("recent_ratings", [[]])[0] or []
            return [int(x) for x in ids], [float(x) for x in ratings]
        except Exception as e:
            FEAST_ERRORS.labels(operation="get_recent").inc()
            logger.error(f"Interaction fetch failed (SDK): {e}")
            return [], []

    def _get_feast_features(self, user_id: int, item_ids: List[int]) -> Dict:
        if not self.fs: return {}
        t_feast = time.time()
        entity_rows = [{"user_id": user_id, "movie_id": mid} for mid in item_ids]
        try:
            batch = self.fs.get_online_features(
                features=[
                    "user_stats:avg_rating", "user_stats:interaction_count", "user_stats:rating_std",
                    "movie_stats:popularity", "movie_stats:avg_rating", "movie_stats:rating_std",
                    "user_genres:genre_vector", "user_embeddings:embedding", "movie_embeddings:embedding",
                    "user_recent_interactions:recent_movie_ids", "user_recent_interactions:recent_ratings",
                ],
                entity_rows=entity_rows,
                full_feature_names=True,
            ).to_dict()
        except Exception as e:
            FEAST_ERRORS.labels(operation="get_features").inc()
            logger.error(f"Batch features fetch failed: {e}")
            batch = {}

        try:
            # CDC features: align names with cdc_feature_view (user: interaction_count; item: rating_count)
            cdc = self.fs.get_online_features(
                features=[
                    "user_cdc_features:avg_rating",
                    "user_cdc_features:interaction_count",
                    "item_cdc_features:avg_rating",
                    "item_cdc_features:rating_count",
                ],
                entity_rows=entity_rows,
                full_feature_names=True,
            ).to_dict()
            if cdc.get("user_cdc_features__avg_rating"):
                batch["user_stats__avg_rating"] = cdc["user_cdc_features__avg_rating"]
            if cdc.get("user_cdc_features__interaction_count"):
                batch["user_stats__interaction_count"] = cdc["user_cdc_features__interaction_count"]
            if cdc.get("item_cdc_features__avg_rating"):
                batch["movie_stats__avg_rating"] = cdc["item_cdc_features__avg_rating"]
            if cdc.get("item_cdc_features__rating_count"):
                batch["movie_stats__popularity"] = cdc["item_cdc_features__rating_count"]
        except Exception as e:
            logger.warning(f"CDC features fetch failed (non-critical): {e}")

        FEAST_LATENCY.labels(operation="get_features").observe(time.time() - t_feast)
        return batch

    def _stage2_rerank(self, user_id: int, candidates: List[Dict], top_k: int) -> List[Dict]:
        if not (self.reranker or self.triton_client) or not candidates: return candidates[:top_k]
        
        if self.feature_extractor:
            item_ids = [c["item_id"] for c in candidates]
            u_tensor = torch.tensor([user_id] * len(item_ids), device=self.feature_extractor.device)
            i_tensor = torch.tensor(item_ids, device=self.feature_extractor.device)
            
            with torch.no_grad():
                batch_tensor = self.feature_extractor.extract_features_batch(u_tensor, i_tensor)
                t_model = time.time()
                if self.triton_client:
                    scores = self.triton_client.rerank(batch_tensor.cpu().numpy()).flatten()
                else:
                    scores = self.reranker(batch_tensor).flatten().cpu().numpy()
                MODEL_INFERENCE_LATENCY.observe(time.time() - t_model)
        else:
            data = self._get_feast_features(user_id, [c["item_id"] for c in candidates])
            if not data: return candidates[:top_k]

            def get_v(name, idx, default):
                vals = data.get(name, [])
                return vals[idx] if idx < len(vals) and vals[idx] is not None else default

            all_recent: set = set()
            if self._feast_only:
                for i in range(len(candidates)):
                    for r in get_v("user_recent_interactions__recent_movie_ids", i, []) or []:
                        if r is not None:
                            all_recent.add(int(r))
            recent_emb_map = (
                self._get_feast_movie_embeddings_batch(list(all_recent))
                if self._feast_only
                else {}
            )

            batch = []
            for i, c in enumerate(candidates):
                u = self._np_emb(get_v("user_embeddings__embedding", i, None), self.embed_dim)
                m = self._np_emb(get_v("movie_embeddings__embedding", i, None), self.embed_dim)
                r_ids, r_stats = get_v("user_recent_interactions__recent_movie_ids", i, []), get_v("user_recent_interactions__recent_ratings", i, [])
                r_emb = np.zeros(self.embed_dim, dtype=np.float32)
                v_ids = 0
                for rid in r_ids or []:
                    if rid is None:
                        continue
                    ri = int(rid)
                    if self._feast_only:
                        if ri in recent_emb_map:
                            r_emb += recent_emb_map[ri]
                            v_ids += 1
                    elif ri < len(self.item_emb):
                        r_emb += self.item_emb[ri]
                        v_ids += 1
                if v_ids > 0: r_emb /= v_ids

                # Individual Recent Interactions (5 * embed_dim + 5 ratings)
                r_5_emb = np.zeros(5 * self.embed_dim, dtype=np.float32)
                r_5_rating = np.zeros(5, dtype=np.float32)
                for j in range(5):
                    if j < len(r_ids or []):
                        rid = r_ids[j]
                        rating = r_stats[j] if j < len(r_stats) else 0.0
                        if rid is not None:
                            ri = int(rid)
                            if self._feast_only and ri in recent_emb_map:
                                r_5_emb[j*self.embed_dim:(j+1)*self.embed_dim] = recent_emb_map[ri]
                            elif not self._feast_only and ri < len(self.item_emb):
                                r_5_emb[j*self.embed_dim:(j+1)*self.embed_dim] = self.item_emb[ri]
                            r_5_rating[j] = float(rating)

                feat = DeepFMFeatureExtractor.combine_features(
                    u_emb=u, i_emb=m, dot=float(np.dot(u, m)),
                    u_st=np.array([
                        get_v("user_stats__avg_rating", i, 0.0) / 5.0, 
                        np.log1p(get_v("user_stats__interaction_count", i, 0.0)), 
                        get_v("user_stats__rating_std", i, 0.0)
                    ]),
                    i_st=np.array([
                        0.0,
                        np.log1p(get_v("movie_stats__popularity", i, 0.0)), 
                        get_v("movie_stats__avg_rating", i, 0.0) / 5.0, 
                        get_v("movie_stats__rating_std", i, 0.0)
                    ]),
                    genres=np.array(get_v("user_genres__genre_vector", i, np.zeros(18))),
                    h_emb=r_emb, h_rating=float(np.mean(r_stats)) if r_stats else 0.0,
                    r5_emb=r_5_emb, r5_rating=np.array(r_5_rating)
                )
                batch.append(feat)

            t_model = time.time()
            if self.triton_client:
                scores = self.triton_client.rerank(np.array(batch)).flatten()
            elif self.reranker:
                with torch.no_grad():
                    scores = self.reranker(torch.FloatTensor(np.array(batch))).flatten().numpy()
            else:
                 return candidates[:top_k]
            MODEL_INFERENCE_LATENCY.observe(time.time() - t_model)

        for item, score in zip(candidates, scores): 
            item["rerank_score"] = float(score)
        return sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)

    def recommend_home(self, user_id: int, top_k: Optional[int] = None) -> Dict:
        scenario = "home"
        top_k = top_k or self.top_k_final
        REC_REQUESTS.labels(scenario=scenario).inc()
        t0 = time.time()
        cache_key = f"home:{user_id}:{top_k}:v1"
        cached = self.cache.get(cache_key)
        if cached:
            CACHE_HIT_TOTAL.labels(scenario=scenario).inc()
            cached.update({"cached": True, "latency_ms": round((time.time() - t0) * 1000, 2)})
            REC_LATENCY.labels(scenario=scenario).observe(time.time() - t0)
            return cached

        CACHE_MISS_TOTAL.labels(scenario=scenario).inc()
        u_emb_hybrid = self._get_hybrid_user_emb(user_id)
        candidates = recommend_home(user_id=0, user_emb=np.array([u_emb_hybrid]), item_emb=self.item_emb, milvus_collection=self.milvus_collection, k=self.top_k_candidates)
        candidates = self._stage2_rerank(user_id, candidates, top_k * 2)

        if self.use_mmr:
            if self._feast_only and len(self.item_emb) == 0:
                mids = [c["item_id"] for c in candidates]
                emb_m = self._get_feast_movie_embeddings_batch(mids)

                def _mmr_emb(iid: int) -> np.ndarray:
                    return emb_m.get(int(iid), np.zeros(self.embed_dim, dtype=np.float32))

                final = mmr_rerank(
                    candidates,
                    self.item_emb,
                    lambda_param=self.diversity_lambda,
                    top_k=top_k,
                    get_item_emb=_mmr_emb,
                )
            else:
                final = mmr_rerank(candidates, self.item_emb, lambda_param=self.diversity_lambda, top_k=top_k)
        else:
            final = candidates[:top_k]
        res = {"user_id": user_id, "items": final, "scenario": "home", "latency_ms": round((time.time() - t0) * 1000, 2)}
        self.cache.set(cache_key, res, ttl=60)
        REC_LATENCY.labels(scenario=scenario).observe(time.time() - t0)
        return res

    def recommend_similar(self, user_id: int, item_id: int, alpha: float = 0.7, top_k: Optional[int] = None) -> Dict:
        scenario = "item_detail"
        top_k = top_k or self.top_k_final
        REC_REQUESTS.labels(scenario=scenario).inc()
        t0 = time.time()
        kw = dict(
            item_id=item_id,
            user_id=user_id,
            user_emb=self.user_emb,
            item_emb=self.item_emb,
            milvus_collection=self.milvus_collection,
            alpha=alpha,
            k=self.top_k_candidates,
            fallback_user_emb=self.mean_user_emb,
        )
        if self._feast_only and len(self.item_emb) == 0:
            kw["batch_item_embs"] = self._get_feast_movie_embeddings_batch
        candidates = recommend_similar_items(**kw)
        candidates = self._stage2_rerank(user_id, candidates, top_k * 2)
        if self._feast_only and len(self.item_emb) == 0:
            mids = [c["item_id"] for c in candidates]
            emb_m = self._get_feast_movie_embeddings_batch(mids)

            def _mmr_sim(iid: int) -> np.ndarray:
                return emb_m.get(int(iid), np.zeros(self.embed_dim, dtype=np.float32))

            final = mmr_rerank(candidates, self.item_emb, top_k=top_k, get_item_emb=_mmr_sim)
        else:
            final = mmr_rerank(candidates, self.item_emb, top_k=top_k)
        latency = time.time() - t0
        REC_LATENCY.labels(scenario=scenario).observe(latency)
        return {"user_id": user_id, "source_item_id": item_id, "items": final, "scenario": scenario, "latency_ms": round(latency * 1000, 2)}

    def recommend_search(self, user_id: int, query: str, alpha: float = 0.5, top_k: Optional[int] = None) -> Dict:
        scenario = "search"
        top_k = top_k or self.top_k_final
        REC_REQUESTS.labels(scenario=scenario).inc()
        t0 = time.time()
        u_emb_hybrid = self._get_hybrid_user_emb(user_id)
        rs_kw = dict(
            query=query,
            user_id=user_id,
            es_client=self.es_client,
            user_emb=np.array([u_emb_hybrid]),
            item_emb=self.item_emb,
            milvus_collection=self.milvus_collection,
            index_name=self.es_index_name,
            alpha=alpha,
            k=self.top_k_candidates,
            metadata_df=self.metadata_df,
            embed_dim=self.embed_dim,
        )
        if self._feast_only and len(self.item_emb) == 0:
            rs_kw["batch_resolve_item_embs"] = self._get_feast_movie_embeddings_batch
        candidates = recommend_search(**rs_kw)
        candidates = self._stage2_rerank(user_id, candidates, top_k * 2)

        if self.use_mmr:
            if self._feast_only and len(self.item_emb) == 0:
                mids = [c["item_id"] for c in candidates]
                emb_m = self._get_feast_movie_embeddings_batch(mids)

                def _mmr_se(iid: int) -> np.ndarray:
                    return emb_m.get(int(iid), np.zeros(self.embed_dim, dtype=np.float32))

                final = mmr_rerank(
                    candidates,
                    self.item_emb,
                    lambda_param=self.diversity_lambda,
                    top_k=top_k,
                    get_item_emb=_mmr_se,
                )
            else:
                final = mmr_rerank(candidates, self.item_emb, lambda_param=self.diversity_lambda, top_k=top_k)
        else:
            final = candidates[:top_k]

        latency = time.time() - t0
        REC_LATENCY.labels(scenario=scenario).observe(latency)
        return {"user_id": user_id, "query": query, "items": final, "scenario": scenario, "latency_ms": round(latency * 1000, 2)}

    def get_movies_metadata(self, item_ids: List[int]) -> List[Dict[str, Any]]:
        """Title/genres/tag: prefer ``public.items`` (Postgres), then movies_metadata parquet."""
        if not item_ids:
            return []

        def placeholder(mid: int) -> Dict[str, Any]:
            return {"movie_id": mid, "title": f"Movie {mid}", "genres": "", "tag": ""}

        wanted = [int(i) for i in item_ids]

        pg_by_id: Dict[int, Dict[str, str]] = {}
        try:
            pg_by_id = fetch_items_metadata(wanted)
        except Exception as e:
            logger.warning("Postgres items metadata failed: %s", e)

        df = self.metadata_df
        parquet_ok = df is not None and not getattr(df, "empty", True) and "movie_id" in df.columns
        parquet_by_id: Dict[int, Any] = {}
        if parquet_ok:
            sub = df[df["movie_id"].isin(set(wanted))]
            for _, row in sub.iterrows():
                parquet_by_id[int(row["movie_id"])] = row

        out: List[Dict[str, Any]] = []
        for mid in wanted:
            pg = pg_by_id.get(mid)
            pq = parquet_by_id.get(mid)
            if pg is None and pq is None:
                out.append(placeholder(mid))
                continue

            title = (pg.get("title", "") if pg else "").strip()
            genres = (pg.get("genres", "") if pg else "").strip()
            tag = (pg.get("tag", "") if pg else "").strip()
            if pq is not None:
                if not title:
                    title = str(pq.get("title", "") or "").strip()
                if not genres:
                    genres = str(pq.get("genres", "") or "").strip()
                if not tag:
                    tag = str(pq.get("tag", "") or "").strip()

            if not title:
                title = f"Movie {mid}"
            out.append(
                {
                    "movie_id": mid,
                    "title": title,
                    "genres": genres,
                    "tag": tag,
                }
            )
        return out

    def merge_interaction_session(self, user_id: int, item_id: int, rating: float = 5.0) -> Dict[str, List]:
        self.cache.invalidate_user(user_id)
        old_ids, old_ratings = self._get_user_recent_interactions(user_id)

        if item_id in old_ids:
            idx = old_ids.index(item_id)
            old_ids.pop(idx)
            if idx < len(old_ratings):
                old_ratings.pop(idx)

        new_ids = (old_ids + [int(item_id)])[-5:]
        new_ratings = (old_ratings + [float(rating)])[-5:]

        self.session_history[user_id] = {"ids": new_ids, "ratings": new_ratings}
        if self.feature_extractor:
            self.feature_extractor.watch_history[user_id] = [
                {"item_id": i, "rating": r} for i, r in zip(new_ids, new_ratings)
            ]

        return {"recent_movie_ids": new_ids, "recent_ratings": new_ratings}

    def persist_interaction_db(
        self,
        user_id: int,
        item_id: int,
        rating: float,
        recent_movie_ids: List[int],
        recent_ratings: List[float],
    ) -> None:
        """Write interaction to Postgres (CDC). Flink ingests Debezium → Feast; serving does not push recent."""
        try:
            insert_interaction(user_id, item_id, float(rating), interaction_type="rating")
        except psycopg2.errors.ForeignKeyViolation as e:
            logger.warning(
                "Postgres interaction insert failed (FK, background): %s", e
            )
            return
        except Exception as e:
            logger.exception("Postgres interaction insert failed (background): %s", e)
            return

    def get_recent_interactions(self, user_id: int) -> Dict[str, Any]:
        """Last up to 5 interactions: Postgres (shared), else Feast online — not pod session RAM."""
        ids, ratings = self._get_user_recent_interactions(user_id, prefer_session=False)
        return {
            "user_id": user_id,
            "recent_movie_ids": ids,
            "recent_ratings": ratings,
        }
