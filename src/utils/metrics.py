import numpy as np
from typing import List, Dict, Union


def recall_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """Recall@K: tỷ lệ items liên quan được gợi ý trong top-k."""
    if not relevant:
        return 0.0
    recommended_k = set(recommended[:k])
    hits = len(recommended_k & set(relevant))
    return hits / len(relevant)


def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """Precision@K: tỷ lệ items được gợi ý là liên quan."""
    if not recommended:
        return 0.0
    recommended_k = recommended[:k]
    hits = sum(1 for item in recommended_k if item in set(relevant))
    return hits / k


def ndcg_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """NDCG@K: Normalized Discounted Cumulative Gain."""
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant_set
    )
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """Hit Rate@K: 1 nếu có ít nhất 1 item liên quan trong top-k."""
    return float(bool(set(recommended[:k]) & set(relevant)))


def mrr_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
    """Mean Reciprocal Rank@K."""
    relevant_set = set(relevant)
    for i, item in enumerate(recommended[:k]):
        if item in relevant_set:
            return 1.0 / (i + 1)
    return 0.0


def coverage(all_recommended: List[List[int]], num_items: int) -> float:
    """Item coverage: % items được gợi ý ít nhất một lần."""
    unique_items = set(item for recs in all_recommended for item in recs)
    return len(unique_items) / num_items


def intra_list_diversity(recommended: List[int], item_embeddings: np.ndarray) -> float:
    """Đo sự đa dạng nội bộ danh sách gợi ý."""
    if len(recommended) < 2:
        return 0.0
    embs = item_embeddings[recommended]
    # Cosine similarity giữa các cặp
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normalized = embs / (norms + 1e-8)
    sim_matrix = normalized @ normalized.T
    n = len(recommended)
    total_sim = (np.sum(sim_matrix) - n) / (n * (n - 1))  # loại diagonal
    return 1.0 - total_sim  # diversity = 1 - avg_similarity


def evaluate_batch(
    recommended_lists: Dict[int, List[int]],
    relevant_lists: Dict[int, List[int]],
    ks: Union[int, List[int]] = 20
) -> Dict[str, float]:
    """Đánh giá toàn bộ tập test và trả về metrics trung bình cho một hoặc nhiều giá trị k."""
    if isinstance(ks, int):
        ks = [ks]
    
    metric_results = {k: {"recalls": [], "precisions": [], "ndcgs": [], "hits": [], "mrrs": []} for k in ks}

    for user_id, recommended in recommended_lists.items():
        relevant = relevant_lists.get(user_id, [])
        if not relevant:
            continue
        
        for k in ks:
            metric_results[k]["recalls"].append(recall_at_k(recommended, relevant, k))
            metric_results[k]["precisions"].append(precision_at_k(recommended, relevant, k))
            metric_results[k]["ndcgs"].append(ndcg_at_k(recommended, relevant, k))
            metric_results[k]["hits"].append(hit_rate_at_k(recommended, relevant, k))
            metric_results[k]["mrrs"].append(mrr_at_k(recommended, relevant, k))

    # Average metrics across all users
    final_metrics = {}
    for k in ks:
        results = metric_results[k]
        final_metrics.update({
            f"Recall@{k}": float(np.mean(results["recalls"])) if results["recalls"] else 0.0,
            f"Precision@{k}": float(np.mean(results["precisions"])) if results["precisions"] else 0.0,
            f"NDCG@{k}": float(np.mean(results["ndcgs"])) if results["ndcgs"] else 0.0,
            f"HitRate@{k}": float(np.mean(results["hits"])) if results["hits"] else 0.0,
            f"MRR@{k}": float(np.mean(results["mrrs"])) if results["mrrs"] else 0.0,
        })

    return final_metrics