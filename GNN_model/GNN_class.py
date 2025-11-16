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
        # Check for NaNs and Infs
        if torch.isnan(raw_audio_emb).any() or torch.isinf(raw_audio_emb).any():
            print(">>> WARNING: NaNs or Infs detected in raw audio embeddings. Replacing with 0.")
            # Replace NaNs with 0
            raw_audio_emb = torch.nan_to_num(raw_audio_emb, nan=0.0, posinf=0.0, neginf=0.0)

        # Keep on CPU, move to GPU only when needed
        self.register_buffer('item_audio_emb', raw_audio_emb)
        # --- END SANITIZE ---

        self.register_buffer('artist_ids', data['item'].artist_id.cpu())
        self.register_buffer('album_ids', data['item'].album_id.cpu())
        self.register_buffer('user_original_ids', data['user'].uid.cpu())
        self.register_buffer('item_original_ids', data['item'].item_id.cpu())

        num_artists = self.artist_ids.max().item() + 1
        num_albums = self.album_ids.max().item() + 1
        self.artist_emb = nn.Embedding(num_artists, self.embed_dim)
        self.album_emb = nn.Embedding(num_albums, self.embed_dim)

        nn.init.xavier_uniform_(self.artist_emb.weight)
        nn.init.xavier_uniform_(self.album_emb.weight)

        self.audio_scale = config.gnn.audio_scale
        self.metadata_scale = config.gnn.metadata_scale

        self.audio_proj = None  # Set to None to save memory

        # --- UNIFIED (Request 3): Add projection layer for concatenated item embeddings ---
        # Input dim is embed_dim (audio) + embed_dim (metadata) = embed_dim * 2
        # Output dim is embed_dim
        self.item_project = nn.Linear(self.embed_dim * 2, self.embed_dim)
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

    def forward(self, return_projections=False):
        """
        Full-graph forward (used for evaluation / saving final embeddings).
        Assumes model and data are moved to device *before* calling.
        """
        device = next(self.parameters()).device

        # Move edge data to device
        edge_index = self.edge_index.to(device)
        edge_weight = self.edge_weight_init.to(device)

        # --- DYNAMIC SELF-LOOP FIX ---
        # Add self-loops to the edge index *and* weights for normalization stability
        edge_index_with_loops, edge_weight_with_loops = add_remaining_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=self.num_nodes_total
        )
        # --- END FIX ---

        # Initial embeddings
        user_nodes = torch.arange(self.num_users, device=device)
        item_nodes = torch.arange(self.num_items, device=device)

        user_embed = F.normalize(self.user_emb(user_nodes), p=2, dim=-1, eps=1e-12)
        item_embed = self._get_item_embeddings(item_nodes, device)

        x = torch.cat([user_embed, item_embed], dim=0)
        all_emb_sum = x

        for conv in self.convs:
            # Pass the tensors *with self-loops* to the conv layer
            x = conv(x, edge_index_with_loops, edge_weight=edge_weight_with_loops)
            all_emb_sum = all_emb_sum + x

        x = all_emb_sum / (self.num_layers + 1)
        x = F.normalize(x, p=2, dim=-1, eps=1e-12)

        user_emb = x[:self.num_users]
        item_emb = x[self.num_users:]

        align_loss = torch.tensor(0.0, device=device)
        return user_emb, item_emb, align_loss

    def forward_cpu(self):
        """
        Performs the full-graph forward pass with GNN propagation on CPU.

        --- HYBRID OPTIMIZATION ---
        If a GPU is available, it will be used to accelerate the
        L0 item embedding creation (which is computationally heavy) in batches.
        The main GNN propagation (memory heavy) will still run on the CPU.
        """

        # --- 1. Set up devices ---
        # The main computation device (for GNN conv) is CPU.
        cpu_device = torch.device('cpu')

        # The model's parameters (e.g., user_emb) should be on the CPU,
        # as set by the GNNTrainer.
        model_device = next(self.parameters()).device
        if model_device != cpu_device:
            print(
                f"    > forward_cpu: Warning: Model parameters are on {model_device}, expected {cpu_device}. Moving to CPU.")
            self.to(cpu_device)
            model_device = cpu_device

        # We will try to use a GPU *just* for the L0 item embedding calc.
        gpu_device = None
        if torch.cuda.is_available():
            try:
                gpu_device = torch.device(self.config.gnn.device)  # e.g., 'cuda:0'
                print(f"    > forward_cpu: Found GPU. Accelerating L0 embedding creation on {gpu_device}.")
            except Exception as e:
                print(
                    f"    > forward_cpu: Error setting GPU device {self.config.gnn.device}: {e}. Falling back to CPU.")
                gpu_device = cpu_device
        else:
            print("    > forward_cpu: No GPU found. Running L0 embedding creation on CPU.")
            gpu_device = cpu_device  # Fallback to CPU

        # --- 2. Get edge data (all on CPU) ---
        edge_weight_cpu = self.edge_weight_init.to(cpu_device)
        edge_index_cpu = self.edge_index.to(cpu_device)

        edge_index_with_loops, edge_weight_with_loops = add_remaining_self_loops(
            edge_index_cpu, edge_weight_cpu, fill_value=1.0, num_nodes=self.num_nodes_total
        )

        # --- 3. Get L0 User Embeddings (Hybrid CPU/GPU) ---
        # --- MODIFIED: Offload user norm to GPU if available ---
        print("    > forward_cpu: Normalizing L0 User Embeddings...")
        user_emb_gpu = self.user_emb.weight.to(gpu_device)
        user_embed_gpu_norm = F.normalize(user_emb_gpu, p=2, dim=-1, eps=1e-12)
        user_embed = user_embed_gpu_norm.to(cpu_device)
        del user_emb_gpu, user_embed_gpu_norm  # Free VRAM
        if gpu_device != cpu_device:
            torch.cuda.empty_cache()
        print("    > forward_cpu: L0 User Embeddings complete.")
        # --- END MODIFIED ---

        # --- 4. Get L0 Item Embeddings (Hybrid CPU/GPU) ---
        print("    > forward_cpu: Calculating L0 Item Embeddings (Hybrid)...")
        # Move relevant parameters *temporarily* to the GPU (if available)
        self.artist_emb.to(gpu_device)
        self.album_emb.to(gpu_device)
        self.item_project.to(gpu_device)  # UNIFIED (Request 3)
        if self.audio_proj is not None:
            self.audio_proj.to(gpu_device)

        batch_size = 10000
        item_embeds_cpu_list = []

        # We iterate over items, compute embeddings on GPU, move back to CPU
        for i in range(0, self.num_items, batch_size):
            end_idx = min(i + batch_size, self.num_items)

            # Create the batch of node indices *on the GPU*
            batch_item_nodes_gpu = torch.arange(i, end_idx, device=gpu_device)

            # This calculation now runs on the GPU
            item_embed_batch_gpu = self._get_item_embeddings(batch_item_nodes_gpu, gpu_device)

            # Move the result *back to the CPU*
            item_embeds_cpu_list.append(item_embed_batch_gpu.cpu())

        # Move parameters *back to the model's original device* (CPU)
        self.artist_emb.to(model_device)
        self.album_emb.to(model_device)
        self.item_project.to(model_device)  # UNIFIED (Request 3)
        if self.audio_proj is not None:
            self.audio_proj.to(model_device)
        if gpu_device != cpu_device:
            torch.cuda.empty_cache()  # Clean up VRAM

        # Concatenate all CPU batches
        item_embed = torch.cat(item_embeds_cpu_list, dim=0)
        del item_embeds_cpu_list  # Free memory
        print("    > forward_cpu: L0 Item Embeddings complete.")

        # --- 5. Concatenate L0 Embeddings (on CPU) ---
        x = torch.cat([user_embed, item_embed], dim=0)
        del user_embed, item_embed

        # --- 6. Run GNN Propagation (on CPU) ---
        print("    > forward_cpu: Starting GNN propagation on CPU...")
        all_emb_sum = x

        for i, conv in enumerate(self.convs):
            print(f"    > forward_cpu: GNN Layer {i + 1}/{self.num_layers}...")
            # This is the slow part that runs on CPU (as intended)
            x = conv(x, edge_index_with_loops, edge_weight=edge_weight_with_loops)
            all_emb_sum = all_emb_sum + x

        print("    > forward_cpu: GNN propagation complete.")

        # --- 7. Final Aggregation & Normalization (on CPU) ---
        # Stays on CPU to avoid massive H2D/D2H copies
        print("    > forward_cpu: Final aggregation and normalization on CPU...")
        x = all_emb_sum / (self.num_layers + 1)
        del all_emb_sum

        x = F.normalize(x, p=2, dim=-1, eps=1e-12)

        user_emb = x[:self.num_users]
        item_emb = x[self.num_users:]

        align_loss_placeholder = torch.tensor(0.0, device=cpu_device)

        print("    > forward_cpu: Complete.")
        return user_emb, item_emb, align_loss_placeholder

    def forward_subgraph(self, batch_nodes, edge_index_sub, edge_weight_init_sub):
        """
        Forward pass on a subgraph with proper embedding combination.
        (Not used in the CPU-based training loop, but kept for completeness
         and potential future use)
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

        user_nodes = batch_nodes[user_mask]
        item_nodes = batch_nodes[item_mask] - self.num_users

        # Get embeddings
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

        return x_sub, user_nodes, item_nodes