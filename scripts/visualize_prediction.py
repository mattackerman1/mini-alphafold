# scripts/visualize_prediction.py
"""
Interactive 3D visualization of predicted vs ground-truth protein backbone.
Usage:
    python scripts/visualize_prediction.py --checkpoint ckpt_phase3.pt --pdb-id 1UBQ
"""

import argparse
import torch
import py3Dmol
from src.data.pdb_parser import parse_structure_file
from src.models.structure_module import build_msa_structure_model
from src.data.msa_encoder import encode_single_sequence

def visualize_prediction(checkpoint_path: str, pdb_id: str):
    # 1. Load ground truth
    gt_data = parse_structure_file(f"{pdb_id}")
    gt_coords = gt_data.coords  # (L, 4, 3)  [N, CA, C, O]

    # 2. Load model + checkpoint
    model = build_msa_structure_model(n_blocks=4, c_m=128, c_z=64, use_triangle=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 3. Prepare input (single sequence + pseudo-MSA for now)
    tokens = encode_single_sequence(gt_data.sequence).unsqueeze(0)  # (1, L)

    with torch.no_grad():
        output = model(tokens)                    # full forward with Evoformer
        pred_ca = output.coords[:, :, 1]          # (1, L, 3) — Cα only

    # 4. Create visualization
    view = py3Dmol.view(width=800, height=600)

    # Ground truth (blue cartoon + CA sticks)
    gt_pdb_str = gt_data.to_pdb_string() if hasattr(gt_data, 'to_pdb_string') else open(f"{pdb_id}.pdb").read()
    view.addModel(gt_pdb_str, "pdb")
    view.setStyle({'cartoon': {'color': 'blue', 'opacity': 0.7}})
    view.addStyle({'stick': {'colorscheme': 'blueCarbon', 'radius': 0.15}}, {'elem': 'CA'})

    # Predicted (red cartoon + CA sticks)
    pred_pdb_str = gt_data.to_pdb_string_with_coords(pred_ca[0])  # you'll need to implement this helper or use a simple one
    view.addModel(pred_pdb_str, "pdb")
    view.setStyle({'cartoon': {'color': 'red', 'opacity': 0.7}}, {'model': -1})
    view.addStyle({'stick': {'colorscheme': 'redCarbon', 'radius': 0.15}}, {'elem': 'CA', 'model': -1})

    view.zoomTo()
    view.show()

    print(f"✅ Visualization ready for {pdb_id}")
    print("   Blue = Ground truth")
    print("   Red  = Model prediction (Cα backbone)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint.pt")
    parser.add_argument("--pdb-id", required=True, help="PDB ID to visualize")
    args = parser.parse_args()
    visualize_prediction(args.checkpoint, args.pdb_id)
