import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from models.gcn_model import GCN
import os

LABEL_MAP = {
    0: "safe",
    1: "reentrancy",
    2: "timestamp_dependency",
    3: "integer_overflow"
}

def load_graph(node_path, edge_path, target_dim=250):
    # Step 1: Read node labels and types
    with open(node_path, 'r') as f:
        node_lines = f.readlines()

    node_id_map = {}   # label → index
    node_types = []    # optional: collect types for one-hot
    current_id = 0

    for line in node_lines:
        tokens = line.strip().split()
        if not tokens or len(tokens) < 2:
            continue
        label = tokens[0]
        if label not in node_id_map:
            node_id_map[label] = current_id
            node_types.append(tokens[2] if len(tokens) >= 3 else "UNK")  # use type if available
            current_id += 1

    num_nodes = len(node_id_map)

    # Step 2: Create dummy features (can be replaced with smarter encoding)
    # We'll one-hot encode based on node index for now
    x = torch.eye(num_nodes)

    # Step 3: Pad to fixed input dimension
    if x.shape[1] < target_dim:
        padding = target_dim - x.shape[1]
        x = torch.cat([x, torch.zeros((x.shape[0], padding))], dim=1)

    # Step 4: Load edges and remap labels to indices
    with open(edge_path, 'r') as f:
        edge_lines = f.readlines()

    edges = []
    for line in edge_lines:
        tokens = line.strip().split()
        if len(tokens) < 3:
            continue
        src_label = tokens[0]
        dst_label = tokens[2]

        if src_label not in node_id_map or dst_label not in node_id_map:
            continue  # skip if unknown label

        src = node_id_map[src_label]
        dst = node_id_map[dst_label]
        edges.append([src, dst])

    if not edges:
        raise ValueError(f"❌ No valid edges found in {edge_path}")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)


def predict_vulnerability(model_path, node_file, edge_file):
    data = load_graph(node_file, edge_file)
    input_dim = data.x.shape[1]

    model = GCN(input_dim=input_dim, hidden_dim=64, output_dim=4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        out = model(data)
        pred = out.argmax().item()
        return LABEL_MAP.get(pred, "unknown")

if __name__ == "__main__":
    contract_name = "cross-function-reentrancy-fixed.sol"
    base_path = "GraphDeeSmartContract-master/data/reentrancy/graph_data"
    node_file = os.path.join(base_path, "node", contract_name)
    edge_file = os.path.join(base_path, "edge", contract_name)

    result = predict_vulnerability("GraphDeeSmartContract-master/trained_gcn_model.pt", node_file, edge_file)
    print(f"🛡️ Prediction for {contract_name}: {result}")
