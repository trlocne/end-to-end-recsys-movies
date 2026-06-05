import torch
import torch.nn as nn
from torch_geometric.nn.models import LightGCN as PyGLightGCN

from typing import Optional, Tuple, Set
 
class LightGCN(nn.Module):
 
    def __init__(self, num_users: int, num_items: int,
                 embedding_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_nodes = num_users + num_items

        self.model = PyGLightGCN(
            num_nodes = self.num_nodes,
            embedding_dim = embedding_dim,
            num_layers = num_layers
        )

    def user_embedding(self, user_ids: torch.Tensor):
        return self.model.embedding(user_ids)

    def item_embedding(self, item_ids: torch.Tensor):
        return self.model.embedding(item_ids + self.num_users)

    def forward(self, edge_index: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            user_emb: (num_users, dim)
            item_emb: (num_items, dim)
        """
        emb = self.model.get_embedding(edge_index, edge_weight=edge_weight)
        user_emb = emb[:self.num_users]
        item_emb = emb[self.num_users:]
        return user_emb, item_emb
 
   
    def get_user_embedding(self, user_ids: torch.Tensor,
                           edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        if edge_index is not None:
            user_emb, _ = self.forward(edge_index)
            return user_emb[user_ids]
        return self.user_embedding(user_ids)
 
    def get_item_embedding(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self.item_embedding(item_ids)
 
    @torch.no_grad()
    def recommend(
        self,
        user_id: int,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        exclude_items: Optional[Set[int]] = None,
        top_k: int = 20,
    ):
        """
        Recommend top-k item local ids cho 1 user.
        """
        self.eval()

        src_index = torch.tensor([user_id], device=edge_index.device)
        dst_index = torch.arange(
            self.num_users,
            self.num_users + self.num_items,
            device=edge_index.device,
        )

        rec_global = self.model.recommend(
            edge_index=edge_index,
            edge_weight=edge_weight,
            src_index=src_index,
            dst_index=dst_index,
            k=min(top_k + (len(exclude_items) if exclude_items else 0), self.num_items),
            sorted=True,
        ).squeeze(0)  # shape: (k,)

        # chuyển global item id -> local item id
        rec_local = (rec_global - self.num_users).tolist()

        if exclude_items:
            rec_local = [i for i in rec_local if i not in exclude_items]

        return rec_local[:top_k]