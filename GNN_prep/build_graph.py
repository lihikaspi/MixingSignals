import duckdb
import torch
from torch_geometric.data import HeteroData
import numpy as np


class GraphBuilder:
    """
    Class to construct the graph used for the GNN model.
    """
    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    def _build_graph(self):
        """
        Construct the train graph used for the GNN model.
        """
        edges_df = self.con.execute("""
                                    SELECT user_id,
                                           COALESCE(item_train_idx, -1) AS item_train_idx,
                                           item_id                      AS original_item_idx,
                                           edge_count,
                                           edge_avg_played_ratio,
                                           edge_type,
                                           edge_weight
                                    FROM agg_edges_artist_album
                                             JOIN agg_edges USING (item_id)
                                    """).fetch_df()

        # Filter out edges pointing to items not in train (optional: keep them with -1 if needed)
        edges_df = edges_df[edges_df['item_train_idx'] >= 0].copy()

        # Edge index
        edge_index_np = np.vstack((edges_df['user_id'].values, edges_df['item_train_idx'].values))
        edge_index = torch.from_numpy(edge_index_np).long()

        # --- UPDATED: Consolidate all attributes into one tensor ---
        # Order matches config.gnn.edge_attr_indices:
        # 0: edge_type
        # 1: edge_count
        # 2: edge_avg_played_ratio
        # 3: edge_weight (heuristic)
        edge_attr = torch.tensor(
            edges_df[['edge_type', 'edge_count', 'edge_avg_played_ratio', 'edge_weight']].fillna(0).values,
            dtype=torch.float
        )

        # Load item node features
        item_embeddings_df = self.con.execute("""
                                              SELECT item_train_idx,
                                                     item_normalized_embed,
                                                     artist_idx,
                                                     album_idx,
                                                     item_id AS original_item_idx
                                              FROM agg_edges_artist_album
                                              WHERE item_train_idx >= 0
                                              ORDER BY item_train_idx
                                              """).fetch_df()

        # Build HeteroData
        data = HeteroData()
        data['user', 'interacts', 'item'].edge_index = edge_index
        data['user', 'interacts', 'item'].edge_attr = edge_attr

        data['item'].x = torch.tensor(np.vstack(item_embeddings_df['item_normalized_embed'].values), dtype=torch.float)
        data['item'].artist_id = torch.tensor(item_embeddings_df['artist_idx'].values, dtype=torch.long)
        data['item'].album_id = torch.tensor(item_embeddings_df['album_idx'].values, dtype=torch.long)
        data['item'].item_id = torch.tensor(item_embeddings_df['original_item_idx'].astype(np.int64).values,
                                            dtype=torch.long)  # keep original for mapping back
        data['item'].num_nodes = len(item_embeddings_df)

        users_df = self.con.execute("""
                                    SELECT DISTINCT uid, user_id
                                    FROM events_with_idx
                                    ORDER by uid
                                    """).fetch_df()

        # Users
        num_users = edges_df['user_id'].max() + 1
        data['user'].user_id = torch.tensor(users_df['user_id'].values, dtype=torch.long)
        data['user'].uid = torch.tensor(users_df['uid'].astype(np.int64).values, dtype=torch.long)
        data['user'].num_nodes = num_users

        print("Finished constructing graph")

        return data

    def build_graph(self, output_path: str):
        """
        Construct the train graph and save it

        Args:
            output_path: path to save the graph
        """
        data = self._build_graph()
        torch.save(data, output_path)
        print(f"Graph saved to {output_path}")