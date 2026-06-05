import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Optional
import mlflow


def build_mlp_block(in_dim: int, out_dim: int,
                    dropout: float = 0.3, 
                    batch_norm: bool = True) -> nn.Sequential:
    """Build một MLP block: Linear → BatchNorm (optional) → ReLU → Dropout"""
    layers: List[nn.Module] = [nn.Linear(in_dim, out_dim)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_dim))
    layers.extend([nn.ReLU(), nn.Dropout(dropout)])
    return nn.Sequential(*layers)


class FMLayer(nn.Module):
    """
    Factorization Machine: First-order + Second-order interactions cho continuous features.
    
    First-order:  ∑ w_i * x_i
    Second-order: 0.5 * ∑_f ( (∑_i v_if * x_i)² - ∑_i (v_if * x_i)² )
    """

    def __init__(self, input_dim: int, embed_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.w = nn.Linear(input_dim, 1)
        self.v = nn.Parameter(torch.randn(input_dim, embed_dim, dtype=torch.float32) * 0.01)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, input_dim) - float features
        
        Returns:
            fm_out: (B, 2) - [first_order, second_order]
            x:      (B, input_dim) - input features cho Deep component
        """
        first_order = self.w(x) # (B, 1)

        sum_xv = torch.matmul(x, self.v)
        sum_xv_sq = sum_xv ** 2

        sum_x_sq_v_sq = torch.matmul(x ** 2, self.v ** 2)
        
        second_order = 0.5 * (sum_xv_sq - sum_x_sq_v_sq).sum(dim=1, keepdim=True) # (B, 1)
        
        fm_out = torch.cat([first_order, second_order], dim=1)
        return fm_out, x


class DeepFMReRanker(nn.Module):
    """
    DeepFM: Kết hợp FM + Deep networks.
    
    Architecture:
        Input → [FM path: 1st + 2nd order] ─┐
                [Deep path: MLP network]    ├→ Output layer → Score
                                            ─┘
    
    Ưu điểm:
      - Captures cả low-order (FM) và high-order (MLP) interactions
      - Hiệu quả với categorical features thưa
      - Training ổn định
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 64,
        mlp_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            input_dim:       Số lượng features đầu vào
            embed_dim:       Embedding dimension cho FM (default: 64)
            mlp_dims:        MLP hidden dimensions (default: [256, 128, 64])
            dropout:         Dropout rate (default: 0.3)
            use_batch_norm:  Sử dụng batch normalization (default: True)
        """
        super().__init__()
        
        if mlp_dims is None:
            mlp_dims = [256, 128, 64]
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        self.fm = FMLayer(input_dim, embed_dim)
        
        deep_layers: List[nn.Module] = []
        in_dim = input_dim
        
        for hidden_dim in mlp_dims:
            deep_layers.append(
                build_mlp_block(in_dim, hidden_dim, dropout, use_batch_norm)
            )
            in_dim = hidden_dim
        
        self.deep_network = nn.Sequential(*deep_layers)
        
        # Log architecture to MLflow if run is active
        if mlflow.active_run():
            mlflow.log_params({
                "reranker_input_dim": input_dim,
                "reranker_embed_dim": embed_dim,
                "reranker_mlp_dims": str(mlp_dims),
                "reranker_dropout": dropout
            })
        
        self.output_layer = nn.Linear(2 + in_dim, 1)
        
        self.to(torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) - float tensor
        
        Returns:
            scores: (B, 1) - predicted scores
        """
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        
        fm_out, features = self.fm(x)  # (B, 2), (B, input_dim)
        deep_out = self.deep_network(features)  # (B, mlp_dims[-1])
        combined = torch.cat([fm_out, deep_out], dim=1)  # (B, 2 + mlp_dims[-1])
        scores = self.output_layer(combined)  # (B, 1)
        
        return scores

    @torch.no_grad()
    def rerank(self, x: torch.Tensor, candidates: List[Dict],
               top_k: int = 20) -> List[Dict]:
        """
        Predict scores và rerank candidates.
        
        Args:
            x:          (B, num_fields) - input tensor
            candidates: Danh sách candidates (dicts)
            top_k:      Số kết quả trả về
        
        Returns:
            Sorted list of candidates với rerank_score
        """
        self.eval()
        scores = self(x).cpu().numpy().flatten()
        s_min, s_max = float(scores.min()), float(scores.max())
        score_range = s_max - s_min if s_max > s_min else 1.0
        for item, score in zip(candidates, scores):
            item["rerank_score"] = (float(score) - s_min) / score_range
        return sorted(
            candidates,
            key=lambda d: d["rerank_score"],
            reverse=True
        )[:top_k]
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        torch.save(self.state_dict(), path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        self.load_state_dict(torch.load(path))

class DeepFMFeatureExtractor:
    """
    Trích xuất feature vectors cho DeepFM re-ranker.
    Đã được đơn giản hóa: không cache stats cồng kềnh, chỉ giữ lại embedding và data gốc.
    
    Total feature dims: 8*embed_dim + 32
    """

    def __init__(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        user_features: Dict,
        item_features: Dict,
        user_watch_history: Dict,
        embed_dim: int = 64,
        **kwargs
    ):
        self.user_emb = user_emb.to(torch.float32)
        self.item_emb = item_emb.to(torch.float32)
        self.user_feat = user_features
        self.item_feat = item_features
        self.watch_history = user_watch_history
        self.embed_dim = embed_dim
        self.device = user_emb.device
        
        # Mean embedding for fallback
        self.mean_user_emb = self.user_emb.mean(dim=0)

    @property
    def feature_dim(self) -> int:
        return 8 * self.embed_dim + 32

    def extract_features_batch(
        self, 
        user_ids: torch.Tensor, 
        item_ids: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized feature extraction on GPU/CPU."""
        device = user_ids.device
        batch_size = user_ids.size(0)
        
        nu, ni = self.user_emb.size(0), self.item_emb.size(0)
        
        u_idx = torch.clamp(user_ids, 0, nu - 1)
        i_idx = torch.clamp(item_ids, 0, ni - 1)
        
        u_emb = self.user_emb[u_idx].to(device)
        i_emb = self.item_emb[i_idx].to(device)
        
        u_emb[user_ids >= nu] = self.mean_user_emb.to(device)
        i_emb[item_ids >= ni] = 0.0

        dot = (u_emb * i_emb).sum(dim=1, keepdim=True)

        u_stats_raw = [self.user_feat.get(int(u), [0.0]*21) for u in user_ids]
        u_stats_tensor = torch.from_numpy(np.array(u_stats_raw, dtype=np.float32)).to(device)
        
        u_st = u_stats_tensor[:, :3].clone()
        u_st[:, 0] = u_st[:, 0] / 5.0           # Normalize rating to [0, 1]
        u_st[:, 1] = torch.log1p(u_st[:, 1])   # Log-scale interaction count
        
        genres_batch = u_stats_tensor[:, 3:21]
        
        i_stats_list = [self.item_feat.get(int(i), [0.0]*4)[:4] for i in item_ids]
        i_st = torch.from_numpy(np.array(i_stats_list, dtype=np.float32)).to(device)
        i_st = i_st.clone()
        i_st[:, 0] = (i_st[:, 0] - 1950) / 100.0   # Normalize year (roughly 0..1 for modern films)
        i_st[:, 1] = torch.log1p(i_st[:, 1])       # Log-scale popularity
        i_st[:, 2] = i_st[:, 2] / 5.0              # Normalize rating to [0, 1]
        
        unique_users, inverse_indices = user_ids.unique(return_inverse=True)
        u_recent_emb = torch.zeros((len(unique_users), self.embed_dim), device=device)
        u_recent_rating = torch.zeros((len(unique_users), 1), device=device)
        u_recent_5_emb = torch.zeros((len(unique_users), 5 * self.embed_dim), device=device)
        u_recent_5_rating = torch.zeros((len(unique_users), 5), device=device)

        item_emb_dev = self.item_emb.to(device)

        for i, uid_tensor in enumerate(unique_users):
            uid = int(uid_tensor)
            history = self.watch_history.get(uid, [])
            recent = history[-5:]
            if recent:
                r_ids = [h["item_id"] for h in recent if h["item_id"] < ni]
                r_ratings = [float(h.get("rating", 0.0)) for h in recent]
                if r_ids:
                    u_recent_emb[i] = item_emb_dev[r_ids].mean(dim=0)
                    u_recent_rating[i] = torch.tensor(r_ratings, device=device).mean()
                    
                for idx, h in enumerate(recent):
                    h_id = h.get("item_id")
                    if h_id is not None and h_id < ni:
                        u_recent_5_emb[i, idx * self.embed_dim : (idx + 1) * self.embed_dim] = item_emb_dev[h_id]
                        u_recent_5_rating[i, idx] = float(h.get("rating", 0.0))
        
        # Combine everything
        return self.combine_features(
            u_emb=u_emb, i_emb=i_emb, dot=dot,
            u_st=u_st, i_st=i_st,
            genres=genres_batch,
            h_emb=u_recent_emb[inverse_indices], h_rating=u_recent_rating[inverse_indices],
            r5_emb=u_recent_5_emb[inverse_indices], r5_rating=u_recent_5_rating[inverse_indices]
        )

    @staticmethod
    def combine_features(
        u_emb,
        i_emb,
        dot,
        u_st,
        i_st,
        genres,
        h_emb,
        h_rating,
        r5_emb,
        r5_rating
    ):
        """
        Gộp các thành phần feature thành một vector duy nhất.
        Làm việc trên cả Tensor (Batch) và Numpy (Single).
        """
        if torch.is_tensor(u_emb):
            return torch.cat([
                u_emb, i_emb, dot,
                u_st, i_st,
                genres,
                h_emb, h_rating,
                r5_emb, r5_rating
            ], dim=1)
        else:
            return np.concatenate([
                u_emb, i_emb, [dot],
                u_st, i_st,
                genres,
                h_emb, [h_rating],
                r5_emb, r5_rating
            ])

    def extract_features(
        self, 
        user_id: int, 
        item_id: int,
    ) -> np.ndarray:
        """Single pair feature extraction (CPU-friendly)."""
        # Shortcut: using extract_features_batch for consistency
        u_tensor = torch.tensor([user_id], device=self.device)
        i_tensor = torch.tensor([item_id], device=self.device)
        feat = self.extract_features_batch(u_tensor, i_tensor)
        return feat.detach().cpu().numpy().flatten()