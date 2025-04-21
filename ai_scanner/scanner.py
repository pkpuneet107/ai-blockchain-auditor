# scanner.py (rewritten with GCN model replacing rule-based detection)

import os
import sys
import torch
from torch_geometric.data import Data
GRAPH_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "GraphDeeSmartContract-master", "data", "reentrancy", "graph_data"))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "trained_gcn_model.pt"))
sys.path.append("/Users/pkpuneet/Projects/AI_PROJECT/GraphDeeSmartContract-master")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.gcn_model import GCN
from scripts.log_to_chain import log_vulnerability

LABEL_MAP = {
    0: "safe",
    1: "reentrancy",
    2: "timestamp_dependency",
    3: "integer_overflow"
}

def load_graph(node_path, edge_path, target_dim=250):
    # Read nodes
    with open(node_path, 'r') as f:
        node_lines = f.readlines()

    node_id_map = {}
    node_features = []
    current_id = 0

    for line in node_lines:
        tokens = line.strip().split()
        if not tokens or len(tokens) < 1:
            continue
        label = tokens[0]
        if label not in node_id_map:
            node_id_map[label] = current_id
            current_id += 1

    num_nodes = len(node_id_map)
    x = torch.eye(num_nodes)
    if x.shape[1] < target_dim:
        padding = target_dim - x.shape[1]
        x = torch.cat([x, torch.zeros((x.shape[0], padding))], dim=1)

    with open(edge_path, 'r') as f:
        edge_lines = f.readlines()

    edges = []
    for line in edge_lines:
        tokens = line.strip().split()
        if len(tokens) < 3:
            continue
        src_label, _, dst_label = tokens[0], tokens[1], tokens[2]
        if src_label not in node_id_map or dst_label not in node_id_map:
            continue
        edges.append([node_id_map[src_label], node_id_map[dst_label]])

    if not edges:
        raise ValueError(f"No valid edges found in {edge_path}")

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index)

def predict_vulnerability(model_path, node_path, edge_path):
    data = load_graph(node_path, edge_path)
    input_dim = data.x.shape[1]

    model = GCN(input_dim=input_dim, hidden_dim=64, output_dim=4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        out = model(data)
        pred = out.argmax().item()
    return LABEL_MAP.get(pred, "unknown")

def scan_and_log(contract_name):
    base_path = "GraphDeeSmartContract-master/data/reentrancy/graph_data"
    node_file = os.path.join(GRAPH_DATA_DIR, "node", contract_name)
    edge_file = os.path.join(GRAPH_DATA_DIR, "edge", contract_name)


    result = predict_vulnerability(MODEL_PATH, node_file, edge_file)
    print(f"\n🛡️ Prediction for {contract_name}: {result}")

    target_address = "0x0000000000000000000000000000000000000001"  # placeholder
    log_vulnerability(target_address, result)
    print("✅ Logged to blockchain")

if __name__ == "__main__":
    contract_to_scan = "Reentrance_01.sol"  # Replace or loop over multiple
    scan_and_log(contract_to_scan)