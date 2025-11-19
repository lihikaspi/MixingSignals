import duckdb
import torch
from torch_geometric.data import HeteroData
import numpy as np
import pandas as pd


class GraphBuilder:
    """
    Class to construct the graph used for the GNN model.
    """
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    def _fetch_and_filter_edges(self) -> pd.DataFrame:
        """Fetches edge data from DB and filters for valid train items."""
        query = """
                SELECT user_id,
                       COALESCE(item_train_idx, -1) AS item_train_idx,
                       item_id                      AS original_item_idx,
                       edge_count,
                       edge_avg_played_ratio,
                       edge_type,
                       edge_weight
                FROM agg_edges_artist_album
                         JOIN agg_edges USING (item_id)
                """
        edges_df = self.con.execute(query).fetch_df()

        # Filter out edges pointing to items not in train
        return edges_df[edges_df['item_train_idx'] >= 0].copy()


    def _create_edge_tensors(self, edges_df: pd.DataFrame):
        """Constructs PyG edge index and edge attribute tensors."""
        # Edge Index: [2, num_edges]
        edge_index_np = np.vstack((edges_df['user_id'].values, edges_df['item_train_idx'].values))
        edge_index = torch.from_numpy(edge_index_np).long()

        # Edge Attributes: [num_edges, 4]
        # Order: edge_type, edge_count, edge_avg_played_ratio, edge_weight
        edge_attr = torch.tensor(
            edges_df[['edge_type', 'edge_count', 'edge_avg_played_ratio', 'edge_weight']].fillna(0).values,
            dtype=torch.float
        )

        return edge_index, edge_attr


    def _fetch_item_features(self) -> pd.DataFrame:
        """Fetches item embeddings and metadata."""
        query = """
                SELECT item_train_idx,
                       item_normalized_embed,
                       artist_idx,
                       album_idx,
                       item_id AS original_item_idx
                FROM agg_edges_artist_album
                WHERE item_train_idx >= 0
                ORDER BY item_train_idx
                """
        return self.con.execute(query).fetch_df()


    def _fetch_user_data(self) -> pd.DataFrame:
        """Fetches user mapping data."""
        query = """
                SELECT DISTINCT uid, user_id
                FROM events_with_idx
                ORDER by uid
                """
        return self.con.execute(query).fetch_df()


    def _build_graph(self) -> HeteroData:
        """Orchestrates the graph construction process."""
        print("Fetching and processing edges...")
        edges_df = self._fetch_and_filter_edges()
        edge_index, edge_attr = self._create_edge_tensors(edges_df)
        num_users_from_edges = edges_df['user_id'].max() + 1

        print("Fetching and processing item nodes...")
        item_df = self._fetch_item_features()

        print("Fetching user nodes...")
        user_df = self._fetch_user_data()

        # Initialize HeteroData
        data = HeteroData()

        # --- Assign Edge Data ---
        data['user', 'interacts', 'item'].edge_index = edge_index
        data['user', 'interacts', 'item'].edge_attr = edge_attr

        # --- Assign Item Node Data ---
        # Feature matrix X (audio embeddings)
        data['item'].x = torch.tensor(np.vstack(item_df['item_normalized_embed'].values), dtype=torch.float)
        # Metadata ID tensors
        data['item'].artist_id = torch.tensor(item_df['artist_idx'].values, dtype=torch.long)
        data['item'].album_id = torch.tensor(item_df['album_idx'].values, dtype=torch.long)
        # Original ID for mapping back
        data['item'].item_id = torch.tensor(item_df['original_item_idx'].astype(np.int64).values, dtype=torch.long)
        data['item'].num_nodes = len(item_df)

        # --- Assign User Node Data ---
        data['user'].user_id = torch.tensor(user_df['user_id'].values, dtype=torch.long)
        data['user'].uid = torch.tensor(user_df['uid'].astype(np.int64).values, dtype=torch.long)
        data['user'].num_nodes = max(num_users_from_edges, len(user_df))

        print("Finished constructing graph")
        return data


    def build_graph(self, output_path: str):
        """
        Construct the train graph and save it.

        Args:
            output_path: path to save the graph
        """
        data = self._build_graph()
        torch.save(data, output_path)
        print(f"Graph saved to {output_path}")