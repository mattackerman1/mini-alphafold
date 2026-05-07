"""
Interactive 3D visualisation of predicted vs ground-truth Ca backbone.

Loads a checkpoint produced by train_end_to_end.py, runs the Evoformer-based
structure model on a query sequence, and opens an HTML page in your default
browser showing:
  - Ground truth backbone  (blue  cartoon/stick)
  - Predicted  backbone    (red   cartoon/stick)

Both backbones are rendered as Ca-only pseudo-PDB strings, so no side-chain
information is needed.

Usage
-----
  # PDB ID -- downloads the structure from RCSB automatically
  python scripts/visualize_prediction.py \\
      --checkpoint ckpt_phase3.pt \\
      --pdb-id 1UBQ

  # Local PDB file
  python scripts/visualize_prediction.py \\
      --checkpoint ckpt_phase3.pt \\
      --pdb-id path/to/my_protein.pdb

Requirements
------------
  pip install py3Dmol biopython torch
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
import webbrowser
from pathlib import Path

import torch

# Ensure repo root is on sys.path when called from any directory
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str) -> dict:
    """Load a .pt checkpoint with flexible key handling.

    Accepts checkpoints saved by train_end_to_end.py in any of three forms:
      1. {'model_state': ..., 'args': ..., ...}  -- new format (our script)
      2. {'model_state_dict': ...}               -- common PyTorch convention
      3. A raw OrderedDict / state_dict           -- bare state dict

    Returns a normalised dict with keys:
      'model_state'  -- state dict ready for model.load_state_dict()
      'args'         -- training args dict (may be empty if not saved)
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(raw, dict):
        if "model_state" in raw:
            return {"model_state": raw["model_state"], "args": raw.get("args", {})}
        if "model_state_dict" in raw:
            return {"model_state": raw["model_state_dict"], "args": raw.get("args", {})}
        # Assume it IS the state dict (all keys are parameter names)
        return {"model_state": raw, "args": {}}

    raise TypeError(f"Unrecognised checkpoint type: {type(raw)}")


# ---------------------------------------------------------------------------
# Model reconstruction from saved args
# ---------------------------------------------------------------------------

def build_model_from_checkpoint(ckpt: dict):
    """Reconstruct the StructureModule using the args stored in the checkpoint.

    Falls back to sensible defaults if args were not saved.
    """
    from src.models.structure_module import build_msa_structure_model

    a = ckpt["args"]
    model = build_msa_structure_model(
        n_blocks=a.get("n_blocks", 2),
        c_m=a.get("c_m", 64),
        c_z=a.get("c_z", 32),
        use_frames=True,
        use_triangle=not a.get("no_triangle", False),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Structure loading (RCSB download or local file)
# ---------------------------------------------------------------------------

def load_structure(pdb_id_or_path: str):
    """Return a StructureData for a PDB ID or a local .pdb path.

    If the argument looks like a 4-character PDB code (no path separators,
    no extension), the file is downloaded from RCSB into a temp directory.
    Otherwise it is treated as a local path.
    """
    from src.data.pdb_parser import parse_structure_file

    p = Path(pdb_id_or_path)

    # Local file -- use directly
    if p.suffix.lower() in (".pdb", ".ent", ".cif", ".mmcif"):
        print(f"  Loading local file: {p}")
        return parse_structure_file(p)

    # Looks like a bare PDB code -- download from RCSB
    pdb_id = pdb_id_or_path.strip().upper()
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {pdb_id} from RCSB...")
    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        return parse_structure_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# MSA generation + model inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_ca(model, sequence: str, n_pseudo: int = 8, seed: int = 0) -> torch.Tensor:
    """Run a forward pass and return predicted Ca coords (L, 3).

    Generates a pseudo-MSA from the query sequence, tokenises it,
    and feeds both through the Evoformer-based StructureModule.
    """
    from src.data.msa_encoder import MSAEncoder, generate_pseudo_msa
    from src.data.protein_dataset import tokenize

    # Tokenise query sequence: (L,) -> (1, L)
    tokens = tokenize(sequence).unsqueeze(0)

    # Build pseudo-MSA and encode: (1, N_seq, L, 45)
    msa_seqs = generate_pseudo_msa(sequence, n_sequences=n_pseudo, seed=seed)
    encoder  = MSAEncoder()
    msa_feat = encoder.encode(msa_seqs)       # MSAFeatures
    msa      = msa_feat.features.unsqueeze(0) # (1, N_seq, L, 45)

    out = model(tokens, msa_features=msa)
    return out.ca_coords[0]  # (L, 3)


# ---------------------------------------------------------------------------
# Coordinate -> PDB string conversion
# ---------------------------------------------------------------------------

def ca_coords_to_pdb(
    ca_xyz: torch.Tensor,
    sequence: str,
    chain_id: str = "A",
) -> str:
    """Convert (L, 3) Ca coordinates to a minimal PDB-format string.

    Produces one ATOM record per residue (Ca only), which is sufficient
    for py3Dmol cartoon and stick rendering.

    Args:
        ca_xyz:   (L, 3) float tensor -- Ca positions in Angstroms.
        sequence: One-letter amino acid string of length L.
        chain_id: Single-character chain identifier.

    Returns:
        Multi-line PDB string ending with 'END'.
    """
    _1TO3 = {
        "A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE",
        "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS", "L": "LEU",
        "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG",
        "S": "SER", "T": "THR", "V": "VAL", "W": "TRP", "Y": "TYR",
    }

    lines = []
    for i, (aa, (x, y, z)) in enumerate(zip(sequence, ca_xyz.tolist()), start=1):
        resname = _1TO3.get(aa, "GLY")
        lines.append(
            f"ATOM  {i:5d}  CA  {resname} {chain_id}{i:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
        )
    lines.append("END")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# py3Dmol visualisation
# ---------------------------------------------------------------------------

def build_viewer(true_pdb: str, pred_pdb: str, title: str = ""):
    """Return a configured py3Dmol view with two labelled models.

    Model 0 = ground truth  (blue)
    Model 1 = prediction    (red)

    Uses 'cartoon' style, which connects residues into a ribbon based on
    the Ca trace (no secondary structure annotation needed).
    """
    try:
        import py3Dmol  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "py3Dmol is required for visualisation.\n"
            "Install it with:  pip install py3Dmol"
        ) from exc

    view = py3Dmol.view(width=900, height=600)

    # Ground truth -- blue cartoon + thin Ca-trace stick
    view.addModel(true_pdb, "pdb")
    view.setStyle(
        {"model": 0},
        {
            "cartoon": {"color": "#2563EB", "opacity": 0.85},
            "stick":   {"color": "#2563EB", "radius": 0.15},
        },
    )

    # Prediction -- red cartoon + thin Ca-trace stick
    view.addModel(pred_pdb, "pdb")
    view.setStyle(
        {"model": 1},
        {
            "cartoon": {"color": "#DC2626", "opacity": 0.85},
            "stick":   {"color": "#DC2626", "radius": 0.15},
        },
    )

    view.zoomTo()
    view.setBackgroundColor("white")

    # Legend via HTML label (py3Dmol doesn't have a native legend API)
    legend_html = (
        "<div style='position:absolute;top:12px;left:12px;"
        "background:rgba(255,255,255,0.85);padding:8px 14px;"
        "border-radius:6px;font-family:sans-serif;font-size:14px;"
        "box-shadow:0 1px 4px rgba(0,0,0,0.2)'>"
        "<b style='color:#2563EB'>&#9632; Ground truth</b>&nbsp;&nbsp;"
        "<b style='color:#DC2626'>&#9632; Predicted</b>"
        + (f"<br><span style='color:#555;font-size:12px'>{title}</span>" if title else "")
        + "</div>"
    )

    return view, legend_html


def show_in_browser(view, legend_html: str, out_html: Path | None = None) -> Path:
    """Write the viewer to an HTML file and open it in the default browser.

    Args:
        view:        py3Dmol view object.
        legend_html: HTML snippet injected into the page body.
        out_html:    Optional explicit output path; defaults to a temp file.

    Returns:
        Path to the written HTML file.
    """
    raw_html: str = view._make_html()
    page = raw_html.replace("</body>", f"{legend_html}\n</body>", 1)

    if out_html is None:
        tmp = tempfile.NamedTemporaryFile(
            suffix=".html", delete=False, prefix="minifold_viz_"
        )
        out_html = Path(tmp.name)
        tmp.close()

    out_html.write_text(page, encoding="utf-8")
    print(f"  Visualisation saved -> {out_html}")
    webbrowser.open(out_html.as_uri())
    return out_html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load checkpoint and reconstruct model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt  = load_checkpoint(args.checkpoint)
    model = build_model_from_checkpoint(ckpt).to(device)
    saved_args = ckpt["args"]
    print(
        f"  Model: {model.count_parameters():,} params  |  "
        f"n_blocks={saved_args.get('n_blocks', '?')}  "
        f"c_m={saved_args.get('c_m', '?')}  "
        f"c_z={saved_args.get('c_z', '?')}"
    )

    # 2. Load ground-truth structure
    print(f"\nLoading structure: {args.pdb_id}")
    struct = load_structure(args.pdb_id)
    print(f"  {struct}")
    sequence = struct.sequence
    true_ca  = struct.coords[:, 1, :]  # CA atom index = 1, shape (L, 3)

    # 3. Predict Ca backbone
    print(f"\nPredicting Ca backbone for {len(sequence)}-residue sequence...")
    pred_ca = predict_ca(
        model.to("cpu"),
        sequence,
        n_pseudo=args.n_pseudo,
        seed=args.seed,
    )
    print(f"  Predicted Ca shape: {tuple(pred_ca.shape)}")

    # 4. Convert to PDB strings
    true_pdb = ca_coords_to_pdb(true_ca.cpu(), sequence, chain_id="A")
    pred_pdb = ca_coords_to_pdb(pred_ca.cpu(), sequence, chain_id="A")

    # 5. Build and display viewer
    pdb_label = Path(args.pdb_id).stem.upper() if Path(args.pdb_id).exists() else args.pdb_id.upper()
    title = f"{pdb_label}  |  L={len(sequence)}"
    print(f"\nOpening 3D viewer ({title})...")

    view, legend = build_viewer(true_pdb, pred_pdb, title=title)
    out_path = Path(args.output) if args.output else None
    show_in_browser(view, legend, out_html=out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise predicted vs ground-truth Ca backbone with py3Dmol.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", required=True,
        help="Path to a .pt checkpoint saved by train_end_to_end.py.",
    )
    p.add_argument(
        "--pdb-id", required=True,
        help=(
            "4-character PDB code (downloaded from RCSB) "
            "or path to a local .pdb / .cif file."
        ),
    )
    p.add_argument(
        "--n-pseudo", type=int, default=8,
        help="Number of pseudo-MSA homologues used during prediction.",
    )
    p.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed for pseudo-MSA generation.",
    )
    p.add_argument(
        "--output", type=str, default=None,
        help="Optional path to save the HTML visualisation (default: temp file).",
    )
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
