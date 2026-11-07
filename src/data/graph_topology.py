"""
src/data/graph_topology.py

Defines the 8 physiological organ nodes and the 15 directed anatomical feedback edges
spanning the pulmonary, cardiovascular, systemic/inflammatory, and renal-metabolic axes.
Also provides edge configurations for the 4 graph ablation benchmark variants:
1. Clinical (15 directed cascade & feedback arcs)
2. No-Edges (0 edges / self-loops only)
3. Complete (56 directed pairwise edges)
4. Random-15 (15 randomized edges across seeds)
"""
import random
import torch

# 8 Core Physiological Organ Nodes
NODE_FEATURES = [
    "resp_rate",    # 0: Pulmonary
    "spo2",         # 1: Pulmonary
    "heart_rate",   # 2: Cardiovascular
    "sbp",          # 3: Cardiovascular
    "temperature",  # 4: Systemic / Thermal
    "wbc",          # 5: Systemic / Thermal
    "lactate",      # 6: Renal / Metabolic
    "creatinine"    # 7: Renal / Metabolic
]

NUM_NODES = len(NODE_FEATURES)

# Clinical Organ Systems Mapping
ORGAN_SYSTEMS = {
    "Pulmonary": [0, 1],
    "Cardiovascular": [2, 3],
    "Systemic_Thermal": [4, 5],
    "Renal_Metabolic": [6, 7]
}

# 15 Directed Anatomical Cascade & Homeostatic Feedback Edges
EDGES_DIRECTED = [
    (0, 1), # Resp Rate -> SpO2 (Hypoxemic respiratory failure)
    (1, 2), # SpO2 -> Heart Rate (Compensatory tachycardia to hypoxemia)
    (2, 3), # Heart Rate -> SBP (Cardiac output perfusion pressure)
    (3, 7), # SBP -> Creatinine (Renal hypoperfusion / Pre-renal AKI)
    (6, 7), # Lactate -> Creatinine (Metabolic acidosis compounds renal failure)
    (4, 5), # Temperature -> WBC (Systemic inflammatory response / pyrexia)
    (5, 6), # WBC -> Lactate (Microcirculatory hypoperfusion / tissue hypoxia)
    (0, 6), # Resp Rate -> Lactate (Diaphragmatic exhaustion & anaerobic shift)
    (2, 6), # Heart Rate -> Lactate (Circulatory shock cascade)
    (1, 6), # SpO2 -> Lactate (Tissue hypoxia anaerobic glycolysis)
    (1, 0), # SpO2 -> Resp Rate (Carotid body hypoxemic ventilatory reflex)
    (3, 2), # SBP -> Heart Rate (Arterial baroreceptor reflex)
    (6, 0), # Lactate -> Resp Rate (Kussmaul hyperventilation to clear acidosis)
    (7, 3), # Creatinine -> SBP (Renin-angiotensin-aldosterone axis hypertension)
    (6, 3), # Lactate -> SBP (Distributive shock vasodilation)
]

CLINICAL_EDGES_15 = EDGES_DIRECTED
NO_EDGES = []
COMPLETE_EDGES_56 = [(i, j) for i in range(NUM_NODES) for j in range(NUM_NODES) if i != j]

EDGE_INDEX = torch.tensor(EDGES_DIRECTED, dtype=torch.long).t().contiguous()


def generate_random_edges_15(seed: int = 42):
    """Generates 15 random directed pairs for graph topology ablation."""
    rng = random.Random(seed)
    all_possible = [(i, j) for i in range(NUM_NODES) for j in range(NUM_NODES) if i != j]
    return rng.sample(all_possible, 15)


def get_edge_index(variant: str = "clinical", seed: int = 42, device=None):
    """
    Returns the edge_index tensor for a specific graph ablation variant:
    - 'clinical': The 15 directed physiological pathways
    - 'no_edges': Empty edge tensor (isolates temporal TCN)
    - 'complete': All 56 directed edges
    - 'random'  : 15 randomized directed edges
    """
    variant = variant.lower()
    if variant == "clinical":
        edges = CLINICAL_EDGES_15
    elif variant in ["no_edges", "none", "empty"]:
        edges = NO_EDGES
    elif variant in ["complete", "dense"]:
        edges = COMPLETE_EDGES_56
    elif variant == "random":
        edges = generate_random_edges_15(seed)
    else:
        raise ValueError(f"Unknown graph variant '{variant}'. Choose from: clinical, no_edges, complete, random.")

    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    src = [e[0] for e in edges]
    dst = [e[1] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long, device=device)


def get_topology_info(variant: str = "clinical", seed: int = 42):
    """Returns a clinical summary dictionary of the organ graph topology."""
    edge_idx = get_edge_index(variant, seed)
    return {
        "variant": variant,
        "num_nodes": len(NODE_FEATURES),
        "num_edges": edge_idx.shape[1],
        "node_features": NODE_FEATURES,
        "organ_systems": ORGAN_SYSTEMS,
        "edge_index_shape": list(edge_idx.shape)
    }
