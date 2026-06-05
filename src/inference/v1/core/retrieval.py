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

    if len(item_emb) == 0 and batch_item_embs is not None:
        need = [int(item_id)] + [int(x) for x in candidates_ids]
        emb_map = batch_item_embs(need) or {}

    results  = []
    user_vec = None
    if user_id is not None:
        if user_emb is not None and len(user_emb) > 0 and user_id < len(user_emb):
            user_vec = user_emb[user_id]
        elif fallback_user_emb is not None:
            # Pre-computed mean passed from caller — no recompute per request
            user_vec = fallback_user_emb
        elif user_emb is not None and len(user_emb) > 0:
            user_vec = np.mean(user_emb, axis=0)

    def _item_vec_at(idx: int) -> np.ndarray:
        if emb_map:
            v = emb_map.get(int(idx))
            if v is None:
                return np.array([], dtype=np.float32)
            return np.asarray(v, dtype=np.float32).ravel()
        if get_item_emb is not None:
            return np.asarray(get_item_emb(idx), dtype=np.float32).ravel()
        return item_emb[idx]

    src_emb = _item_vec_at(item_id)
    if src_emb.size == 0:
        return []

    for idx in candidates_ids:
        i_emb = _item_vec_at(int(idx))
        # Item similarity (cosine)
        item_sim  = float(
            np.dot(src_emb, i_emb) /
            (np.linalg.norm(src_emb) * np.linalg.norm(i_emb) + 1e-8)
        )
        # User preference (if user exists)
        user_pref = float(np.dot(user_vec, i_emb)) if user_vec is not None else 0.0
        combined  = alpha * item_sim + (1 - alpha) * user_pref

        results.append({
            "item_id":          idx,
            "item_similarity":  item_sim,
            "user_preference":  user_pref,
            "ann_score":        combined,
            "source":           "item2item",
        })

    results.sort(key=lambda d: d["ann_score"], reverse=True)
    return results[:k]

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

    search_candidates: set = set()
    bm25_scores_map: Dict[int, float] = {}

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
                idx = int(idx_str)
                bm25_scores_map[idx] = hit["_score"] / max_score
                search_candidates.add(idx)
        except Exception as e:
            print(f"[Search] ES failed: {e}")

    if not search_candidates and metadata_df is not None:
        fallback_ids = local_keyword_search(query, metadata_df, k=k)
        for idx in fallback_ids:
            bm25_scores_map[idx] = 1.0 # Give high score to exact title matches
            search_candidates.add(idx)

    cf_candidates = set()
    if milvus_collection is not None and user_emb is not None:
        if user_id is not None and user_id < len(user_emb):
            user_vec = user_emb[user_id].astype(np.float32).tolist()
        else:
            user_vec = np.mean(user_emb, axis=0).astype(np.float32).tolist()
        search_params = {"metric_type": "IP", "params": {"ef": 64}}
        results = milvus_collection.search(
            data=[user_vec],
            anns_field="embedding",
            param=search_params,
            limit=k,
        )
        cf_candidates = {hit.id for hit in results[0]}
    elif item_emb is not None and len(item_emb) > 0 and user_emb is not None:
        if user_id is not None and user_id < len(user_emb):
            user_vec_np = user_emb[user_id].reshape(1, -1)
        else:
            user_vec_np = np.mean(user_emb, axis=0).reshape(1, -1)
        cf_scores = item_emb @ user_vec_np.T
        cf_candidates = set(np.argsort(-cf_scores.ravel())[:k].tolist())


    all_candidates = search_candidates | cf_candidates

    item_vec_by_id: Optional[Dict[int, np.ndarray]] = None
    if batch_resolve_item_embs is not None and (item_emb is None or len(item_emb) == 0) and all_candidates:
        item_vec_by_id = batch_resolve_item_embs(list(all_candidates))

    if user_emb is not None and len(user_emb) > 0:
        if user_id is not None and user_id < len(user_emb):
            user_vec_flat = user_emb[user_id]
        else:
            user_vec_flat = np.mean(user_emb, axis=0)
    else:
        user_vec_flat = np.zeros(embed_dim, dtype=np.float32)
    results = []
    for idx in all_candidates:
        bm25_score = bm25_scores_map.get(idx, 0.0)
        if item_vec_by_id is not None:
            i_vec = item_vec_by_id.get(idx)
            if i_vec is None:
                continue
        elif get_item_emb is not None:
            i_vec = np.asarray(get_item_emb(idx), dtype=np.float32).ravel()
        elif item_emb is not None and len(item_emb) > idx:
            i_vec = item_emb[idx]
        else:
            continue
        cf_score   = float(np.dot(user_vec_flat, i_vec))
        cf_norm    = (cf_score + 1) / 2   # normalize to ~[0,1]
        final      = alpha * bm25_score + (1 - alpha) * cf_norm
        results.append({
            "item_id":      idx,
            "bm25_score":   bm25_score,
            "cf_score":     cf_norm,
            "ann_score":    final,
            "source":       "hybrid",
        })

    results.sort(key=lambda d: d["ann_score"], reverse=True)
    return results[:k]

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

    # Normalize relevance scores to [0, 1] so they are on the same scale as cosine similarity
    raw_scores = [c.get("rerank_score", c.get("ann_score", 0.0)) for c in candidates]
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