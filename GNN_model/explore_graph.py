import torch
from config import config

# Path to the saved graph
graph_path = config.paths.train_graph_file

print(f"Loading graph from: {graph_path}")
# Load the graph
data = torch.load(graph_path)

# Number of nodes
num_users = data['user'].num_nodes
num_items = data['item'].num_nodes

# Number of edges
num_edges = data['user', 'interacts', 'item'].edge_index.size(1)

print("=" * 60)
print("GRAPH STATISTICS")
print("=" * 60)
print(f"Number of user nodes: {num_users}")
print(f"Number of item (song) nodes: {num_items}")
print(f"Number of edges: {num_edges}")
print("-" * 60)

# --- Example user ---
example_user_id = 0
if 'user_id' in data['user']:
    print("Example User Node:")
    print(f"  Graph Index: {example_user_id}")
    print(f"  Original User ID: {data['user'].user_id[example_user_id].item()}")
    print(f"  UID: {data['user'].uid[example_user_id].item()}")
else:
    print("User IDs not found in graph data.")
print("-" * 60)

# --- Example item ---
example_item_id = 0
if 'x' in data['item']:
    print("Example Item Node:")
    print(f"  Graph Index: {example_item_id}")
    print(f"  Original Item ID: {data['item'].item_id[example_item_id].item()}")
    print(f"  Artist Index: {data['item'].artist_id[example_item_id].item()}")
    print(f"  Album Index: {data['item'].album_id[example_item_id].item()}")
    print(f"  Audio Embedding (first 5 dims): {data['item'].x[example_item_id][:5].tolist()}")
else:
    print("Item features not found in graph data.")
print("-" * 60)

# --- Example edge ---
edge_index = data['user', 'interacts', 'item'].edge_index
edge_attr = data['user', 'interacts', 'item'].edge_attr

# Check edge attribute shape
print(f"Edge Attribute Shape: {edge_attr.shape}")

example_edge_id = 0
src_user = edge_index[0, example_edge_id].item()
dst_item = edge_index[1, example_edge_id].item()
edge_features = edge_attr[example_edge_id]

# Parse attributes using config indices
attr_indices = config.gnn.edge_attr_indices
e_type = edge_features[attr_indices['type']].item()
e_count = edge_features[attr_indices['count']].item()
e_ratio = edge_features[attr_indices['ratio']].item()
e_weight = edge_features[attr_indices['weight']].item()

print("Example Edge:")
print(f"  Edge Index: {example_edge_id}")
print(f"  Source (User Graph Index): {src_user}")
print(f"  Destination (Item Graph Index): {dst_item}")
print(f"  Raw edge_attr: {edge_features.tolist()}")
print(f"  Parsed Attributes:")
print(f"    - Type (1=listen, 2=like, etc): {int(e_type)}")
print(f"    - Interaction Count: {int(e_count)}")
print(f"    - Avg Played Ratio: {e_ratio:.4f}")
print(f"    - Heuristic Weight: {e_weight:.4f}")

print("=" * 60)