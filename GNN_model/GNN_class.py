import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_remaining_self_loops
from config import Config


def print_model_info(model):
    """
    Prints the total number of parameters, trainable parameters,
    and estimated model size in megabytes (MB) and gigabytes (GB).
    """
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    bytes_per_param = 4  # Assuming float32
    total_bytes = total_params * bytes_per_param
    total_mb = total_bytes / (1024 ** 2)
    total_gb = total_bytes / (1024 ** 3)

    print(f"--- Model '{model.__class__.__name__}' Info (from __init__) ---")
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-Trainable Params: {total_params - trainable_params:,}")
    print("-----------------------------------")
    print(f"Estimated Size (float32):")
    print(f"  {total_mb:.4f} MB")
    print(f"  {total_gb:.6f} GB")
    print("-----------------------------------")


class EdgeWeightMLP(nn.Module):
    """
    A small MLP to learn edge weights from edge attributes.
    Input: [batch_size, input_dim] -> Output: [batch_size, 1] (sigmoid)
    """
    def __init__(self, config: Config):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(config.gnn.edge_mlp_input_dim, config.gnn.edge_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gnn.edge_mlp_hidden_dim, 1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def forward(self, x):
        return self.mlp(x)


class LightGCN(nn.Module):
    def __init__(self, data: HeteroData, config: Config):
        """
        Args:
            data: HeteroData object containing the graph structure and node attributes.
            config: Config object containing model hyperparameters.
        """
        print('>>> starting GNN init')

        super().__init__()
        self.config = config
        self.num_layers = config.gnn.num_layers
        self.embed_dim = config.gnn.embed_dim

        # Node counts
        self.num_users = data['user'].num_nodes
        self.num_items = data['item'].num_nodes
        self.num_nodes_total = self.num_users + self.num_items

        # 1. Initialize Node Embeddings & Projections
        self._init_node_embeddings(data)
        # 2. Register Item Data (Audio & IDs)
        self._register_item_data(data)
        # 3. Initialize Edge Weight Module
        self.edge_mlp = EdgeWeightMLP(config)
        # 4. Build Homogeneous Graph Structure
        self._build_homogeneous_graph(data)
        # 5. Initialize GNN Layers
        self.convs = nn.ModuleList([
            LGConv(normalize=True)
            for _ in range(self.num_layers)
        ])

        print(">>> finished GNN init")


    def _init_node_embeddings(self, data):
        """
        Initializes user, artist, and album embeddings.

        Args:
            data: HeteroData object containing the graph structure and node attributes.
        """
        # User Embeddings
        self.user_emb = nn.Embedding(self.num_users, self.embed_dim)
        nn.init.xavier_uniform_(self.user_emb.weight)

        # Metadata Embeddings
        # Note: We need to access max IDs from data to size the embeddings
        # Using CPU to avoid GPU sync/OOM during init if graph is huge
        artist_ids = data['item'].artist_id.cpu()
        album_ids = data['item'].album_id.cpu()

        num_artists = artist_ids.max().item() + 1
        num_albums = album_ids.max().item() + 1

        self.artist_emb = nn.Embedding(num_artists, self.embed_dim)
        self.album_emb = nn.Embedding(num_albums, self.embed_dim)

        nn.init.xavier_uniform_(self.artist_emb.weight)
        nn.init.xavier_uniform_(self.album_emb.weight)

        # Projections & Scales
        self.audio_scale = self.config.gnn.audio_scale
        self.metadata_scale = self.config.gnn.metadata_scale
        self.audio_proj = None
        self.item_project = nn.Linear(self.embed_dim * 2, self.embed_dim)


    def _register_item_data(self, data):
        """
        Registers item-related static tensors as buffers.

        Args:
            data: HeteroData object containing the graph structure and node attributes.
        """
        # Sanitize Audio Embeddings
        raw_audio_emb = data['item'].x.cpu()
        if torch.isnan(raw_audio_emb).any() or torch.isinf(raw_audio_emb).any():
            print(">>> WARNING: NaNs or Infs detected in raw audio embeddings. Replacing with 0.")
            raw_audio_emb = torch.nan_to_num(raw_audio_emb, nan=0.0, posinf=0.0, neginf=0.0)

        self.register_buffer('item_audio_emb', raw_audio_emb)
        self.register_buffer('artist_ids', data['item'].artist_id.cpu())
        self.register_buffer('album_ids', data['item'].album_id.cpu())
        self.register_buffer('user_original_ids', data['user'].uid.cpu())
        self.register_buffer('item_original_ids', data['item'].item_id.cpu())


    def _build_homogeneous_graph(self, data):
        """
        Converts bipartite edge index/attr to homogeneous structure.

        Args:
            data: HeteroData object containing the graph structure and node attributes.
        """
        # Edge features
        edge_index_bipartite = data['user', 'interacts', 'item'].edge_index.cpu()
        edge_attr_bipartite = data['user', 'interacts', 'item'].edge_attr.cpu()

        # Forward edges (user -> item)
        fwd_edge_index = edge_index_bipartite.clone()
        fwd_edge_index[1] += self.num_users  # Offset item IDs

        # Backward edges (item -> user)
        bwd_edge_index = torch.stack([fwd_edge_index[1], fwd_edge_index[0]], dim=0)

        # Combine
        edge_index_full_homo = torch.cat([fwd_edge_index, bwd_edge_index], dim=1)

        # Combine attributes (same for both directions)
        edge_attr_full_homo = torch.cat([edge_attr_bipartite, edge_attr_bipartite], dim=0)

        # Register buffers
        self.register_buffer('edge_index', edge_index_full_homo)
        self.register_buffer('edge_attr_init', edge_attr_full_homo)


    def _get_item_embeddings(self, item_nodes, device):
        """
        Combine audio + metadata embeddings using Concat + Linear Projection.

        Args:
            item_nodes: Tensor containing item IDs.
            device: Device to run the operation on.

        Returns:
            item embeddings of shape [num_items, embed_dim]
        """
        item_nodes_cpu = item_nodes.cpu()

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


    def _compute_edge_weights(self, device):
        """
        Helper to run MLP on edge attributes to get learnable weights.

        Args:
            device: Device to run the operation on.

        Returns:
            MLP-learned edge weights of shape [num_edges, 1]
        """
        # Ensure inputs are on the target device
        attr = self.edge_attr_init.to(device)

        # Ensure MLP is on the target device
        self.edge_mlp.to(device)

        # Run MLP
        weights = self.edge_mlp(attr).squeeze(-1)

        # Enforce minimum weight to keep graph connected
        weights = torch.clamp(weights, min=1e-6)

        return weights


    def forward(self):
        """
        Full-graph forward (GPU version).

        Returns:
            model-learned user and item embeddings
        """
        device = next(self.parameters()).device

        # --- 1. Compute Dynamic Weights ---
        edge_weight = self._compute_edge_weights(device)
        edge_index = self.edge_index.to(device)

        # --- 2. Dynamic Self-Loops ---
        edge_index_with_loops, edge_weight_with_loops = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=self.num_nodes_total
        )

        # Initial embeddings
        user_nodes = torch.arange(self.num_users, device=device)
        item_nodes = torch.arange(self.num_items, device=device)

        user_embed = F.normalize(self.user_emb(user_nodes), p=2, dim=-1, eps=1e-12)
        item_embed = self._get_item_embeddings(item_nodes, device)

        x = torch.cat([user_embed, item_embed], dim=0)
        all_emb_sum = x

        for conv in self.convs:
            x = conv(x, edge_index_with_loops, edge_weight=edge_weight_with_loops)
            all_emb_sum = all_emb_sum + x

        x = all_emb_sum / (self.num_layers + 1)
        x = F.normalize(x, p=2, dim=-1, eps=1e-12)

        user_emb = x[:self.num_users]
        item_emb = x[self.num_users:]

        return user_emb, item_emb


    def forward_cpu(self):
        """
        Performs the full-graph forward pass with GNN propagation on CPU.

        Returns:
            model-learned user and item embeddings
        """
        # --- 1. Set up devices ---
        cpu_device = torch.device('cpu')
        gpu_device = cpu_device  # Force CPU for OOM prevention

        model_device = next(self.parameters()).device

        # --- 2. Compute Dynamic Weights on CPU ---
        edge_weight_cpu = self._compute_edge_weights(cpu_device)
        edge_index_cpu = self.edge_index.to(cpu_device)

        edge_index_with_loops, edge_weight_with_loops = add_remaining_self_loops(
            edge_index_cpu, edge_weight_cpu, fill_value=1.0, num_nodes=self.num_nodes_total
        )

        # --- 3. Get L0 User Embeddings ---
        user_emb_gpu = self.user_emb.weight.to(gpu_device)
        user_embed_gpu_norm = F.normalize(user_emb_gpu, p=2, dim=-1, eps=1e-12)
        user_embed = user_embed_gpu_norm.to(cpu_device)
        del user_emb_gpu, user_embed_gpu_norm
        if gpu_device != cpu_device:
            torch.cuda.empty_cache()

        # --- 4. Get L0 Item Embeddings ---
        self.artist_emb.to(gpu_device)
        self.album_emb.to(gpu_device)
        self.item_project.to(gpu_device)
        if self.audio_proj is not None:
            self.audio_proj.to(gpu_device)

        batch_size = 10000
        item_embeds_cpu_list = []

        for i in range(0, self.num_items, batch_size):
            end_idx = min(i + batch_size, self.num_items)
            batch_item_nodes_gpu = torch.arange(i, end_idx, device=gpu_device)
            item_embed_batch_gpu = self._get_item_embeddings(batch_item_nodes_gpu, gpu_device)
            item_embeds_cpu_list.append(item_embed_batch_gpu.cpu())

        # Return params to model device
        self.artist_emb.to(model_device)
        self.album_emb.to(model_device)
        self.item_project.to(model_device)
        if self.audio_proj is not None:
            self.audio_proj.to(model_device)
        if gpu_device != cpu_device:
            torch.cuda.empty_cache()

        item_embed = torch.cat(item_embeds_cpu_list, dim=0)
        del item_embeds_cpu_list

        # --- 5. Concatenate L0 Embeddings (on CPU) ---
        x = torch.cat([user_embed, item_embed], dim=0)
        del user_embed, item_embed

        # --- 6. Run GNN Propagation (on CPU) ---
        all_emb_sum = x

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index_with_loops, edge_weight=edge_weight_with_loops)
            all_emb_sum = all_emb_sum + x

        # --- 7. Final Aggregation & Normalization (on CPU) ---
        x = all_emb_sum / (self.num_layers + 1)
        del all_emb_sum

        x = F.normalize(x, p=2, dim=-1, eps=1e-12)

        user_emb = x[:self.num_users]
        item_emb = x[self.num_users:]

        align_loss_placeholder = torch.tensor(0.0, device=cpu_device)

        return user_emb, item_emb, align_loss_placeholder

