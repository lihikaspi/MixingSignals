import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_remaining_self_loops
from config import Config


class EmbeddingLayer(nn.Module):
    """Encapsulates all large, trainable embedding tables."""
    def __init__(self, num_users, embed_dim, artist_ids, album_ids, item_audio_emb, config):
        super().__init__()

        # User Embedding
        self.user_emb = nn.Embedding(num_users, embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)

        # Item Meta Embeddings
        num_artists = artist_ids.max().item() + 1
        num_albums = album_ids.max().item() + 1
        self.artist_emb = nn.Embedding(num_artists, embed_dim)
        self.album_emb = nn.Embedding(num_albums, embed_dim)
        nn.init.xavier_uniform_(self.artist_emb.weight)
        nn.init.xavier_uniform_(self.album_emb.weight)

        # Projections
        # UNIFIED (Request 3): Item projection layer
        self.item_project = nn.Linear(embed_dim * 2, embed_dim)
        self.audio_proj = None  # Set to None to save memory

        # Scales (non-trainable, but needed for forward)
        self.audio_scale = config.gnn.audio_scale
        self.metadata_scale = config.gnn.metadata_scale

        # Non-trainable buffers (kept on CPU/host memory)
        self.register_buffer('item_audio_emb', item_audio_emb)
        self.register_buffer('artist_ids', artist_ids)
        self.register_buffer('album_ids', album_ids)

    def get_user_embedding(self, user_nodes):
        # Returns raw user embedding for normalization in LightGCN
        return self.user_emb(user_nodes)

    def get_item_embeddings(self, item_nodes, device):
        """Combines audio and metadata embeddings using projection."""
        item_nodes_cpu = item_nodes.cpu()

        # Move required data to the target device
        item_audio = self.item_audio_emb[item_nodes_cpu].to(device)
        artist_ids_batch = self.artist_ids[item_nodes_cpu].to(device)
        album_ids_batch = self.album_ids[item_nodes_cpu].to(device)

        if self.audio_proj is not None:
            item_audio = self.audio_proj(item_audio)

        artist_emb = self.artist_emb(artist_ids_batch)
        album_emb = self.album_emb(album_ids_batch)

        audio_part = item_audio * self.audio_scale
        metadata_part = (artist_emb + album_emb) * self.metadata_scale

        item_embed = torch.cat([audio_part, metadata_part], dim=-1)
        item_embed = self.item_project.to(device)(item_embed)

        item_embed = F.normalize(item_embed, p=2, dim=-1, eps=1e-12)

        return item_embed


def print_model_info(model):
    """
    Prints the total number of parameters, trainable parameters,
    and estimated model size in megabytes (MB) and gigabytes (GB).

    This calculation assumes parameters are stored in float32 format (4 bytes per param).

    Args:
        model (torch.nn.Module): The PyTorch model.
    """

    total_params = 0
    trainable_params = 0

    # Iterate over all parameters in the model
    # self.parameters() is available because model is an nn.Module
    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    # --- Calculate Model Size ---
    bytes_per_param = 4  # Assuming float32
    total_bytes = total_params * bytes_per_param
    total_mb = total_bytes / (1024 ** 2)
    total_gb = total_bytes / (1024 ** 3)

    # Print the information
    print(f"--- Model '{model.__class__.__name__}' Info (from __init__) ---")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-Trainable Params: {total_params - trainable_params:,}")
    print("-----------------------------------")
    print(f"Estimated Size (float32):")
    print(f"  {total_mb:.4f} MB")
    print(f"  {total_gb:.6f} GB")
    print("-----------------------------------")


class LightGCN(nn.Module):
    def __init__(self, data: HeteroData, config: Config):
        print('>>> starting GNN init')
        super().__init__()
        self.config = config
        self.num_layers = config.gnn.num_layers
        self.lambda_align = config.gnn.lambda_align
        self.embed_dim = config.gnn.embed_dim

        # Node counts
        self.num_users = data['user'].num_nodes
        self.num_items = data['item'].num_nodes
        self.num_nodes_total = self.num_users + self.num_items

        self.user_emb = nn.Embedding(self.num_users, self.embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)

        # --- NEW: Sanitize Audio Embeddings ---
        raw_audio_emb = data['item'].x.cpu()
        if torch.isnan(raw_audio_emb).any() or torch.isinf(raw_audio_emb).any():
            print(">>> WARNING: NaNs or Infs detected in raw audio embeddings. Replacing with 0.")
            raw_audio_emb = torch.nan_to_num(raw_audio_emb, nan=0.0, posinf=0.0, neginf=0.0)

        self.embedding_layer = EmbeddingLayer(
            num_users=self.num_users,
            embed_dim=self.embed_dim,
            artist_ids=data['item'].artist_id.cpu(),
            album_ids=data['item'].album_id.cpu(),
            item_audio_emb=raw_audio_emb,
            config=config  # Pass config to set scales and get projection layer
        )
        # Inherit necessary buffers for evaluation and mapping
        self.register_buffer('user_original_ids', data['user'].uid.cpu())
        self.register_buffer('item_original_ids', data['item'].item_id.cpu())

        # --- Remove redundant scales, user_emb, artist_emb, album_emb, item_project, audio_scale/metadata_scale definitions here ---

        # --- Update accessors to point to the new layer ---
        self.user_emb = self.embedding_layer.user_emb  # Expose for legacy access
        self.artist_ids = self.embedding_layer.artist_ids
        self.album_ids = self.embedding_layer.album_ids
        self.item_audio_emb = self.embedding_layer.item_audio_emb
        self.item_project = self.embedding_layer.item_project
        self.audio_scale = self.embedding_layer.audio_scale
        self.metadata_scale = self.embedding_layer.metadata_scale
        # --- END UNIFIED ---

        # Edge features
        # --- MODIFICATION: Create Homogeneous Edge Index ---
        edge_index_bipartite = data['user', 'interacts', 'item'].edge_index.cpu()
        edge_weight_init_bipartite = data['user', 'interacts', 'item'].edge_weight_init.cpu()

        # --- NEW: Sanitize Edge Weights ---
        if torch.isnan(edge_weight_init_bipartite).any() or torch.isinf(edge_weight_init_bipartite).any():
            print(">>> WARNING: NaNs or Infs detected in raw edge weights. Replacing with 0.")
            edge_weight_init_bipartite = torch.nan_to_num(edge_weight_init_bipartite, nan=0.0, posinf=0.0, neginf=0.0)
        # --- END SANITIZE ---

        # Forward edges (user -> item)
        fwd_edge_index = edge_index_bipartite.clone()
        fwd_edge_index[1] += self.num_users  # Offset item IDs

        # Backward edges (item -> user)
        bwd_edge_index = torch.stack([fwd_edge_index[1], fwd_edge_index[0]], dim=0)

        # Combine
        edge_index_full_homo = torch.cat([fwd_edge_index, bwd_edge_index], dim=1)

        # Combine weights (they are the same for both directions)
        edge_weight_init_full_homo = torch.cat([edge_weight_init_bipartite, edge_weight_init_bipartite], dim=0)

        # Register the new homogeneous buffers
        # *** NOTE: We do NOT add self-loops here ***
        self.register_buffer('edge_index', edge_index_full_homo)
        self.register_buffer('edge_weight_init', edge_weight_init_full_homo)

        # Ensure all edge weights are positive
        self.edge_weight_init.data = torch.clamp(self.edge_weight_init.data, min=1e-6)

        # --- THE FIX: Enable normalization, but do NOT add self-loops here ---
        # Self-loops will be added on-the-fly in the forward passes
        self.convs = nn.ModuleList([
            LGConv(normalize=True)
            for _ in range(self.num_layers)
        ])

        # print_model_info(self)

        print(">>> finished GNN init")

    def _get_item_embeddings(self, item_nodes, device):
        """
        UNIFIED (Request 3): Combine audio + metadata embeddings
        using Concat + Linear Projection.

        This function expects `item_nodes` to be on the *target device*.
        """
        # `item_nodes` is already on the target device.
        # We need to get the corresponding IDs from the CPU buffers.
        item_nodes_cpu = item_nodes.cpu()

        # Move required data to the target device
        item_audio = self.item_audio_emb[item_nodes_cpu].to(device)
        artist_ids_batch = self.artist_ids[item_nodes_cpu].to(device)
        album_ids_batch = self.album_ids[item_nodes_cpu].to(device)

        if self.audio_proj is not None:
            item_audio = self.audio_proj(item_audio)

        artist_emb = self.artist_emb(artist_ids_batch)
        album_emb = self.album_emb(album_ids_batch)

        audio_part = item_audio * self.audio_scale
        metadata_part = (artist_emb + album_emb) * self.metadata_scale

        # --- UNIFIED (Request 3): Concatenate and project ---
        item_embed = torch.cat([audio_part, metadata_part], dim=-1)
        # Move project layer to the correct device
        item_embed = self.item_project.to(device)(item_embed)
        # --- END UNIFIED ---

        item_embed = F.normalize(item_embed, p=2, dim=-1, eps=1e-12)

        return item_embed

    def forward(self, batch_nodes, edge_index_sub, edge_weight_init_sub):
        """
        Primary forward pass for a sampled subgraph (mini-batch training).

        Args:
            batch_nodes (Tensor): Homogeneous node indices for the subgraph.
            edge_index_sub (Tensor): Edge index for the subgraph.
            edge_weight_init_sub (Tensor): Edge weights for the subgraph.
        """
        device = next(self.parameters()).device

        # Move base subgraph data to device
        batch_nodes = batch_nodes.to(device)
        edge_index_sub = edge_index_sub.to(device)
        edge_weight_sub = edge_weight_init_sub.to(device)

        # --- DYNAMIC SELF-LOOP FIX ---
        # Add self-loops *to the subgraph* to ensure no 0-degree nodes
        # `num_nodes` here is the size of the subgraph (len(batch_nodes))
        edge_index_sub_loops, edge_weight_sub_loops = add_remaining_self_loops(
            edge_index_sub, edge_weight_sub, fill_value=1.0, num_nodes=batch_nodes.size(0)
        )
        # --- END FIX ---

        # Identify users vs items
        user_mask = batch_nodes < self.num_users
        item_mask = ~user_mask

        # Get the original user_idx and the zero-indexed item_idx
        user_nodes = batch_nodes[user_mask]
        item_nodes = batch_nodes[item_mask] - self.num_users  # Zero-indexed item ID

        # Get embeddings
        # FSDP manages parameter access, this is fine
        user_embed = F.normalize(self.user_emb(user_nodes), p=2, dim=-1, eps=1e-12)
        item_embed = self._get_item_embeddings(item_nodes, device)

        # Concatenate in subgraph order
        x_sub = torch.zeros((len(batch_nodes), self.embed_dim), device=device)
        x_sub[user_mask] = user_embed
        x_sub[item_mask] = item_embed

        all_emb = [x_sub]
        for conv in self.convs:
            # Pass the subgraph tensors *with self-loops* to the conv layer
            x_sub = conv(x_sub, edge_index_sub_loops, edge_weight=edge_weight_sub_loops)
            all_emb.append(x_sub)

        x_sub = torch.stack(all_emb, dim=0).mean(dim=0)
        x_sub = F.normalize(x_sub, p=2, dim=-1, eps=1e-12)

        # Return the final node embeddings (subgraph order), the user/item indices in the subgraph
        # and the original user/item indices for loss calculation
        return x_sub, user_nodes, item_nodes  # item_nodes are zero-indexed item IDs
