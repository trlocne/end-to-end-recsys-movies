import numpy as np
import torch
import pandas as pd
from typing import Callable, Dict, List, Optional, Tuple

def recommend_home(
    user_id: int,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    milvus_collection=None,
    k: int = 500,
    exclude_items: Optional[set] = None,
) -> List[Dict]:
    user_vec = user_emb[user_id].astype(np.float32).tolist()

    if milvus_collection is not None:
        search_params = {"metric_type": "IP", "params": {"ef": 64}}
        results = milvus_collection.search(
            data=[user_vec],
            anns_field="embedding",
            param=search_params,
            limit=k + (len(exclude_items) if exclude_items else 0),
            expr=None
        )
        candidates = []
        for hit in results[0]:
            idx = int(hit.id)
            if exclude_items is None or idx not in exclude_items:
                candidates.append({"item_id": idx, "ann_score": float(hit.distance), "source": "cf"})
    else:
        # Fallback: brute-force full matrix (in-memory item_emb). For Feast-only serving, use Milvus.
        if len(item_emb) == 0:
            return []
        user_vec_np = np.array(user_vec).reshape(1, -1)
        scores = (item_emb @ user_vec_np.T).ravel()
        top_k  = np.argsort(-scores)[:k]
        candidates = [
            {"item_id": int(idx), "ann_score": float(scores[idx]), "source": "cf"}
            for idx in top_k
            if exclude_items is None or int(idx) not in exclude_items
        ]

    return candidates[:k]

def recommend_similar_items(
    item_id: int,
    user_id: int,
    user_emb: np.ndarray,
    item_emb: np.ndarray,
    milvus_collection=None,
    alpha: float = 0.7,
    k: int = 500,
    fallback_user_emb: Optional[np.ndarray] = None,
    source_item_vec: Optional[np.ndarray] = None,
    get_item_emb: Optional[Callable[[int], np.ndarray]] = None,
    batch_item_embs: Optional[Callable[[List[int]], Dict[int, np.ndarray]]] = None,
) -> List[Dict]:
    emb_map: Dict[int, np.ndarray] = {}
    if len(item_emb) == 0 and batch_item_embs is not None:
        emb_map = batch_item_embs([int(item_id)]) or {}
        src = emb_map.get(int(item_id))
        if src is None:
            return []
        item_vec = np.asarray(src, dtype=np.float32).ravel().tolist()
    elif source_item_vec is not None:
        item_vec = np.asarray(source_item_vec, dtype=np.float32).ravel().tolist()
    elif len(item_emb) == 0 or item_id >= len(item_emb):
        if get_item_emb is None:
            return []
        item_vec = np.asarray(get_item_emb(item_id), dtype=np.float32).ravel().tolist()
    else:
        item_vec = item_emb[item_id].astype(np.float32).tolist()

    if milvus_collection is not None:
        search_params = {"metric_type": "IP", "params": {"ef": 64}}
        results = milvus_collection.search(
            data=[item_vec],
            anns_field="embedding",
            param=search_params,
            limit=k + 1,
            expr=None
        )
        candidates_ids = [hit.id for hit in results[0] if hit.id != item_id]
    else:
        if len(item_emb) == 0:
            return []
        item_vec_np = np.array(item_vec).reshape(1, -1)
        scores = item_emb @ item_vec_np.T
        candidates_ids = np.argsort(-scores.ravel()).tolist()
        candidates_ids = [i for i in candidates_ids if i != item_id][:k]

    # Stage 1: pure item similarity — DeepFM will personalize in stage 2
    results = [
        {"item_id": int(idx), "ann_score": 1.0 - (i / max(len(candidates_ids), 1)), "source": "item2item"}
        for i, idx in enumerate(candidates_ids[:k])
    ]
    return results

def local_keyword_search(
    query: str,
    metadata_df: Optional[pd.DataFrame],
    k: int = 100
) -> List[int]:
    """Fallback: Search for query in local metadata DataFrame titles."""
    if metadata_df is None or "title" not in metadata_df.columns:
        return []
    
    matches = metadata_df[metadata_df["title"].str.contains(query, case=False, na=False)]
    return matches.head(k)["item_idx"].tolist() if "item_idx" in matches.columns else matches.head(k).index.tolist()

def recommend_search(
    query: str,
    user_id: int,
    es_client=None,
    user_emb: np.ndarray = None,
    item_emb: np.ndarray = None,
    milvus_collection=None,
    index_name: str = "movies",
    alpha: float = 0.5,
    k: int = 500,
    metadata_df: Optional[pd.DataFrame] = None,
    get_item_emb: Optional[Callable[[int], np.ndarray]] = None,
    embed_dim: int = 64,
    batch_resolve_item_embs: Optional[Callable[[List[int]], Dict[int, np.ndarray]]] = None,
) -> List[Dict]:
    """BM25-first search: ES retrieves candidates by keyword, DeepFM reranks for personalization."""

    candidates: List[Dict] = []

    if es_client is not None:
        try:
            body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["title^3", "genres^2", "tags^1"]
                    }
                },
                "size": k
            }
            res = es_client.search(index=index_name, body=body)
            hits = res["hits"]["hits"]
            max_score = res["hits"]["max_score"] or 1.0
            for hit in hits:
                idx_str = hit["_source"].get("item_idx", hit["_id"])
                candidates.append({
                    "item_id":    int(idx_str),
                    "bm25_score": hit["_score"] / max_score,
                    "ann_score":  hit["_score"] / max_score,
                    "source":     "bm25",
                })
        except Exception as e:
            logger.warning("ES search failed: %s", e)

    if not candidates and metadata_df is not None:
        fallback_ids = local_keyword_search(query, metadata_df, k=k)
        for idx in fallback_ids:
            candidates.append({
                "item_id":    idx,
                "bm25_score": 1.0,
                "ann_score":  1.0,
                "source":     "local_fallback",
            })

    return candidates[:k]

def weighted_score_fusion(
    ranked_lists: Dict[str, List[Dict]],
    weights: Optional[Dict[str, float]] = None,
) -> List[Dict]:
    """
    Merge bằng weighted sum của scores từ nhiều nguồn.
    """
    if weights is None:
        weights = {src: 1.0 for src in ranked_lists}

    scores: Dict[int, Dict] = {}
    for source, items in ranked_lists.items():
        w = weights.get(source, 1.0)
        max_s = max((d["ann_score"] for d in items), default=1.0)
        for item in items:
            iid   = item["item_id"]
            norm  = item["ann_score"] / (max_s + 1e-8)
            if iid not in scores:
                scores[iid] = {"item_id": iid, "score": 0.0, "sources": []}
            scores[iid]["score"]   += w * norm
            scores[iid]["sources"].append(source)

    result = sorted(scores.values(), key=lambda d: d["score"], reverse=True)
    return result


def reciprocal_rank_fusion(
    ranked_lists: Dict[str, List[Dict]],
    k: int = 60,
) -> List[Dict]:
    scores: Dict[int, Dict] = {}
    for source, items in ranked_lists.items():
        for rank, item in enumerate(items):
            iid = item["item_id"]
            if iid not in scores:
                scores[iid] = {"item_id": iid, "score": 0.0, "sources": []}
            scores[iid]["score"]   += 1.0 / (k + rank + 1)
            scores[iid]["sources"].append(source)

    return sorted(scores.values(), key=lambda d: d["score"], reverse=True)


def cascade_merge(
    ranked_lists: Dict[str, List[Dict]],
    rerank_fn=None,
    top_k: int = 500,
) -> List[Dict]:
    all_ids     = set()
    inter_ids   = None

    for items in ranked_lists.values():
        ids = {item["item_id"] for item in items}
        all_ids |= ids
        inter_ids = ids if inter_ids is None else inter_ids & ids

    id_count: Dict[int, int] = {}
    for items in ranked_lists.values():
        for item in items:
            id_count[item["item_id"]] = id_count.get(item["item_id"], 0) + 1

    all_items  = [{**item, "source_count": id_count[item["item_id"]]}
                  for items in ranked_lists.values() for item in items]
    seen: set = set()
    unique: List[Dict] = []
    for item in all_items:
        if item["item_id"] not in seen:
            seen.add(item["item_id"])
            unique.append(item)

    unique.sort(key=lambda d: (-d["source_count"], -d.get("ann_score", 0)))
    return unique[:top_k]

def mmr_rerank(
    candidates: List[Dict],
    item_emb: np.ndarray,
    lambda_param: float = 0.5,
    top_k: int = 20,
    get_item_emb: Optional[Callable[[int], np.ndarray]] = None,
) -> List[Dict]:
    """
    Maximal Marginal Relevance (MMR) re-ranking.
    MMR_score = λ * relevance - (1-λ) * max_similarity_to_selected
    """
    if not candidates:
        return []

    if len(item_emb) == 0 and get_item_emb is None:
        return sorted(
            candidates,
            key=lambda x: x.get("rerank_score", x.get("ann_score", 0.0)),
            reverse=True,
        )[:top_k]

    # rerank_score is already in [0,1] (sigmoid output) — use directly.
    # ann_score from Milvus Inner Product is unbounded — normalize to [0,1].
    has_rerank = any("rerank_score" in c for c in candidates)
    if has_rerank:
        for c in candidates:
            c["_norm_score"] = float(c.get("rerank_score", 0.0))
    else:
        raw_scores = [c.get("ann_score", 0.0) for c in candidates]
        min_s, max_s = min(raw_scores), max(raw_scores)
        score_range = max_s - min_s if max_s > min_s else 1.0
        for c, s in zip(candidates, raw_scores):
            c["_norm_score"] = (s - min_s) / score_range

    def _emb(item_id: int) -> np.ndarray:
        if get_item_emb is not None:
            return np.asarray(get_item_emb(item_id), dtype=np.float32).ravel()
        return item_emb[item_id]

    selected: List[Dict] = []
    remaining = list(candidates)

    while len(selected) < top_k and remaining:
        best_score = -float("inf")
        best_item  = None

        for item in remaining:
            relevance = item["_norm_score"]

            if selected:
                i_emb = _emb(item["item_id"])
                sims  = [
                    float(np.dot(i_emb, _emb(s["item_id"])) /
                          (np.linalg.norm(i_emb) * np.linalg.norm(_emb(s["item_id"])) + 1e-8))
                    for s in selected
                ]
                max_sim = max(sims)
            else:
                max_sim = 0.0

            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_item  = item

        if best_item is not None:
            selected.append(best_item)
            remaining.remove(best_item)

    for item in selected:
        item.pop("_norm_score", None)

    return selected