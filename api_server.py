"""FastAPI Backend Server for BBBP Molecular Screening & Multi-Compound Comparator.

Provides REST endpoints for:
- /api/health: System health and model availability
- /api/compounds: Searchable catalog with split tags and presets
- /api/screen: Single or multi-compound screening with RDKit SVG diagrams & GAT attention weights
- /api/benchmark: Model leaderboard and performance metrics from summary.json
"""
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

from chem_data import canonical, graph
from predict import ScreeningSystem

app = FastAPI(title="BBBP Screening API", version="1.0.0")

# Enable CORS for frontend local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
system: Optional[ScreeningSystem] = None
catalog_df: Optional[pd.DataFrame] = None
summary_data: dict = {}


def render_mol_svg(smiles: str, width: int = 380, height: int = 240) -> str:
    """Generate dark-mode friendly 2D molecular SVG."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
    opts = drawer.drawOptions()
    opts.clearBackground = True
    opts.bondLineWidth = 2
    # Use clean atom colors for dark background
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def get_gat_attention(smiles: str) -> List[dict]:
    """Calculate GAT Layer-1 incoming attention per atom."""
    if system is None or "gat" not in system.models:
        return []
    try:
        smi = canonical(smiles)
        if not smi:
            return []
        mol = Chem.MolFromSmiles(smi)
        item = graph(smi)
        batch = torch.zeros(item.num_nodes, dtype=torch.long)
        with torch.no_grad():
            _, attns = system.models["gat"](item.x, item.edge_index, batch, return_attention_weights=True)
        edges, weights = attns[0]
        values = np.zeros(item.num_nodes)
        for dst, weight in zip(edges[1].tolist(), weights.mean(dim=1).tolist()):
            values[dst] += weight
        
        result = []
        for i, val in enumerate(values):
            atom_sym = mol.GetAtomWithIdx(i).GetSymbol() if mol and i < mol.GetNumAtoms() else "Atom"
            result.append({
                "atom_index": i,
                "symbol": atom_sym,
                "label": f"{atom_sym}{i}",
                "attention": round(float(val), 4),
            })
        return result
    except Exception as e:
        print(f"Error computing GAT attention: {e}")
        return []


@app.on_event("startup")
def startup_event():
    global system, catalog_df, summary_data
    # 1. Initialize ScreeningSystem
    try:
        system = ScreeningSystem(artifact_dir="artifacts")
        print("ScreeningSystem loaded successfully.")
    except Exception as e:
        print(f"Warning: Failed to load ScreeningSystem: {e}")

    # 2. Load benchmark summary
    summary_path = Path("artifacts/summary.json")
    if summary_path.exists():
        summary_data = json.loads(summary_path.read_text(encoding="utf-8"))

    # 3. Load catalog
    candidates = [
        Path("data_sources/BBBP_with_names.csv"),
        Path("data_sources/bbbp.csv"),
    ]
    cat_path = next((p for p in candidates if p.exists()), None)
    if cat_path:
        df = pd.read_csv(cat_path, encoding="utf-8")
        if "name" not in df.columns:
            df["name"] = [f"Compound #{i+1}" for i in range(len(df))]
        if "num" not in df.columns:
            df["num"] = list(range(1, len(df) + 1))
        if "p_np" not in df.columns and "label" in df.columns:
            df["p_np"] = df["label"]

        # Map scaffold split tags
        splits_dir = Path("artifacts/splits")
        train_set, valid_set, test_set = set(), set(), set()
        if (splits_dir / "train.csv").exists():
            train_set = set(pd.read_csv(splits_dir / "train.csv")["SMILES"])
        if (splits_dir / "valid.csv").exists():
            valid_set = set(pd.read_csv(splits_dir / "valid.csv")["SMILES"])
        if (splits_dir / "test.csv").exists():
            test_set = set(pd.read_csv(splits_dir / "test.csv")["SMILES"])

        def determine_split(smi):
            c = canonical(smi)
            if c in train_set:
                return "Train Set"
            elif c in test_set:
                return "Test Set"
            elif c in valid_set:
                return "Valid Set"
            return "Novel/Other"

        df["split_tag"] = df["smiles"].apply(determine_split)
        catalog_df = df
        print(f"Catalog loaded: {len(catalog_df)} compounds.")


@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "models_loaded": system is not None,
        "catalog_size": len(catalog_df) if catalog_df is not None else 0,
    }


@app.get("/api/compounds")
def get_compounds(
    query: Optional[str] = Query(None, description="Search query by name or SMILES"),
    split: Optional[str] = Query(None, description="Filter by split: 'Train Set', 'Test Set', 'Valid Set'"),
    limit: int = Query(2500, ge=1, le=10000),
    offset: int = Query(0, ge=0),
):
    if catalog_df is None:
        raise HTTPException(status_code=500, detail="Catalog not initialized")
    
    filtered = catalog_df
    if split:
        filtered = filtered[filtered["split_tag"] == split]
    if query and query.strip():
        q = query.strip().lower()
        mask = (
            filtered["name"].astype(str).str.lower().str.contains(q, na=False)
            | filtered["smiles"].astype(str).str.lower().str.contains(q, na=False)
            | filtered["num"].astype(str).str.contains(q, na=False)
        )
        filtered = filtered[mask]

    total = len(filtered)
    rows = filtered.iloc[offset : offset + limit]

    results = []
    for _, r in rows.iterrows():
        raw_p_np = r.get("p_np", None)
        gt_label = None
        if pd.notna(raw_p_np):
            gt_label = "BBB+" if int(raw_p_np) == 1 else "BBB−"

        results.append({
            "num": int(r["num"]),
            "name": str(r["name"]),
            "smiles": str(r["smiles"]).strip(),
            "split_tag": str(r["split_tag"]),
            "ground_truth": gt_label,
            "display_name": f"{r['name']} (#{r['num']}) [{r['split_tag']}]",
        })

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "compounds": results,
    }


@app.get("/api/presets")
def get_presets():
    """Return common curated preset molecules for easy 1-click comparison."""
    if catalog_df is None:
        return {"presets": []}
    
    preset_names = [
        "Propanolol",
        "ondansetron",
        "Etoposide",
        "Atenolol",
        "Diazepam",
        "morphine",
        "haloperidol",
        "caffeine",
    ]
    presets = []
    for p_name in preset_names:
        matches = catalog_df[catalog_df["name"].astype(str).str.lower().str.contains(p_name.lower(), na=False)]
        if not matches.empty:
            r = matches.iloc[0]
            raw_p_np = r.get("p_np", None)
            gt_label = None
            if pd.notna(raw_p_np):
                gt_label = "BBB+" if int(raw_p_np) == 1 else "BBB−"
            presets.append({
                "num": int(r["num"]),
                "name": str(r["name"]),
                "smiles": str(r["smiles"]).strip(),
                "split_tag": str(r["split_tag"]),
                "ground_truth": gt_label,
                "display_name": f"{r['name']} (#{r['num']}) [{r['split_tag']}]",
            })
    return {"presets": presets}


class ScreenRequest(BaseModel):
    smiles: str
    name: Optional[str] = None
    split_tag: Optional[str] = None
    ground_truth: Optional[str] = None


class BatchScreenRequest(BaseModel):
    compounds: List[ScreenRequest]


@app.post("/api/screen")
def screen_compound(req: ScreenRequest):
    if system is None:
        raise HTTPException(status_code=500, detail="Screening system not initialized")

    canonical_smi = canonical(req.smiles)
    if not canonical_smi:
        raise HTTPException(status_code=400, detail=f"Invalid SMILES string: '{req.smiles}'")

    try:
        pred = system.predict(canonical_smi)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    # Ground truth lookup if not provided
    name = req.name or "Custom Molecule"
    split_tag = req.split_tag or "Novel / Custom"
    ground_truth = req.ground_truth

    if catalog_df is not None and not req.ground_truth:
        matches = catalog_df[catalog_df["smiles"].apply(canonical) == canonical_smi]
        if not matches.empty:
            r = matches.iloc[0]
            name = str(r["name"])
            split_tag = str(r["split_tag"])
            raw_p_np = r.get("p_np", None)
            if pd.notna(raw_p_np):
                ground_truth = "BBB+" if int(raw_p_np) == 1 else "BBB−"

    svg_content = render_mol_svg(canonical_smi)
    attention = get_gat_attention(canonical_smi)

    # Explanation note for similarity
    sim = pred["nearest_training_tanimoto"]
    if sim >= 0.999:
        sim_note = "Exact identity match in Training Set (1.00)"
    elif sim < 0.35:
        sim_note = "⚠️ Below 0.35 domain cutoff (Unfamiliar scaffold)"
    else:
        sim_note = f"Unseen test molecule ({sim:.1%} similarity to closest training scaffold)"

    return {
        "smiles": canonical_smi,
        "input_smiles": req.smiles,
        "name": name,
        "split_tag": split_tag,
        "ground_truth": ground_truth,
        "decision": pred["decision"],
        "suggested_class": pred["suggested_class"],
        "ensemble_probability": pred["positive_class_probability"],
        "model_probabilities": {
            "gcn": pred["model_positive_probabilities"]["gcn"],
            "gat": pred["model_positive_probabilities"]["gat"],
            "sage": pred["model_positive_probabilities"]["sage"],
            "rf": pred["model_positive_probabilities"]["rf"],
        },
        "agreement": pred["agreement"],
        "nearest_training_tanimoto": sim,
        "tanimoto_note": sim_note,
        "in_domain": pred["in_domain"],
        "accepted": pred["accepted"],
        "svg": svg_content,
        "attention": attention,
        "notice": pred["notice"],
    }


@app.post("/api/screen-batch")
def screen_batch(req: BatchScreenRequest):
    results = []
    for item in req.compounds:
        try:
            results.append(screen_compound(item))
        except Exception as e:
            results.append({
                "smiles": item.smiles,
                "name": item.name or "Error",
                "error": str(e),
            })
    return {"results": results}


@app.get("/api/benchmark")
def get_benchmark():
    if not summary_data:
        raise HTTPException(status_code=404, detail="Benchmark summary.json not found")
    return summary_data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
